"""Scoped attention backends for upstream SAM-Audio inference."""

from __future__ import annotations

import contextlib
import importlib
import sys
import threading
import types
from collections.abc import Iterator
from typing import Any

import torch
import torch.nn.functional as torch_functional

PYTORCH_ATTENTION = "pytorch"
COMFY_KITCHEN_ATTENTION = "comfy_kitchen"
ATTENTION_BACKENDS = (PYTORCH_ATTENTION, COMFY_KITCHEN_ATTENTION)
_TARGET_MODULE_PREFIXES = ("sam_audio", "core")
_PATCH_LOCK = threading.RLock()
_PYTORCH_SDPA = torch_functional.scaled_dot_product_attention


def _load_comfy_kitchen() -> Any:
    try:
        return importlib.import_module("comfy_kitchen")
    except Exception as error:
        raise RuntimeError(
            "Comfy Kitchen attention is unavailable. Update ComfyUI/comfy-kitchen "
            "or select PyTorch attention."
        ) from error


def validate_attention_backend(backend: str, device: torch.device) -> None:
    if backend not in ATTENTION_BACKENDS:
        raise ValueError(
            f"Unknown attention backend {backend!r}; choose one of {ATTENTION_BACKENDS}"
        )
    if backend == PYTORCH_ATTENTION:
        return

    comfy_kitchen = _load_comfy_kitchen()
    try:
        available = comfy_kitchen.int8_attention_is_available(device)
    except Exception as error:
        raise RuntimeError(
            f"Could not check Comfy Kitchen attention on {device}; select PyTorch attention"
        ) from error
    if not available:
        raise RuntimeError(
            f"Comfy Kitchen INT8 attention is not available on {device}. It requires "
            "a supported comfy-kitchen build and GPU; select PyTorch attention instead."
        )


def _kitchen_sdpa(comfy_kitchen: Any):
    def scaled_dot_product_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        *,
        scale: float | None = None,
        enable_gqa: bool = False,
    ) -> torch.Tensor:
        # Comfy Kitchen is an inference-only, non-causal kernel. Preserve exact
        # PyTorch behavior for uncommon calls outside its supported surface.
        if dropout_p != 0.0 or is_causal:
            return _PYTORCH_SDPA(
                query,
                key,
                value,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                scale=scale,
                enable_gqa=enable_gqa,
            )
        # SAM-Audio creates Q/K/V with permute and RoPE operations that may leave
        # the head dimension strided. Kitchen's INT8 kernel requires stride 1.
        if query.stride(-1) != 1:
            query = query.contiguous()
        if key.stride(-1) != 1:
            key = key.contiguous()
        if value.stride(-1) != 1:
            value = value.contiguous()
        return comfy_kitchen.int8_attention(
            query,
            key,
            value,
            scale=scale,
            attn_mask=attn_mask,
        )

    return scaled_dot_product_attention


class _FunctionalProxy(types.ModuleType):
    def __init__(self, sdpa: Any):
        super().__init__(torch_functional.__name__)
        self.scaled_dot_product_attention = sdpa

    def __getattr__(self, name: str) -> Any:
        return getattr(torch_functional, name)


def _is_target_module(name: str) -> bool:
    return any(
        name == prefix or name.startswith(f"{prefix}.")
        for prefix in _TARGET_MODULE_PREFIXES
    )


@contextlib.contextmanager
def attention_backend_context(backend: str) -> Iterator[None]:
    """Route only SAM-Audio and Perception Models SDPA calls to the backend."""
    if backend == PYTORCH_ATTENTION:
        yield
        return
    if backend != COMFY_KITCHEN_ATTENTION:
        raise ValueError(f"Unknown attention backend {backend!r}")

    comfy_kitchen = _load_comfy_kitchen()
    kitchen_sdpa = _kitchen_sdpa(comfy_kitchen)
    functional_proxy = _FunctionalProxy(kitchen_sdpa)
    patched: list[tuple[types.ModuleType, str, Any]] = []

    # Upstream imports torch.nn.functional as F. Replacing that module-global
    # alias avoids changing torch.nn.functional globally for other ComfyUI nodes.
    with _PATCH_LOCK:
        for name, module in tuple(sys.modules.items()):
            if not _is_target_module(name) or not isinstance(module, types.ModuleType):
                continue
            namespace = vars(module)
            if namespace.get("F") is torch_functional:
                patched.append((module, "F", torch_functional))
                module.F = functional_proxy
            for attribute, value in tuple(namespace.items()):
                if value is _PYTORCH_SDPA:
                    patched.append((module, attribute, value))
                    setattr(module, attribute, kitchen_sdpa)
        try:
            yield
        finally:
            for module, attribute, value in reversed(patched):
                setattr(module, attribute, value)
