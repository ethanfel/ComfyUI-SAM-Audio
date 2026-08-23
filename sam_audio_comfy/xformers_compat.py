"""Narrow xFormers import compatibility for Meta perception_models inference.

perception_models declares xFormers as a training dependency and imports its FMHA
types at module import time. The SAM-Audio inference modules use their PyTorch SDPA
path, so installing a compiled xFormers wheel is unnecessary. This module supplies
only the imported surface when xFormers is absent; it never replaces a real install.
"""

from __future__ import annotations

import contextlib
import sys
import types
from collections.abc import Iterator
from typing import Any

import torch
import torch.nn.functional as F


class AttentionBias:
    """Marker compatible with xformers.ops.AttentionBias type checks."""


def _memory_efficient_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_bias: Any = None,
    p: float = 0.0,
    scale: float | None = None,
    **_: Any,
) -> torch.Tensor:
    """Use PyTorch SDPA for the BSHD tensor convention used by xFormers FMHA."""
    if isinstance(attn_bias, AttentionBias):
        raise NotImplementedError(
            "This xFormers attention-bias type is not used by SAM-Audio inference. "
            "Use the default SDPA attention path."
        )
    mask = attn_bias if isinstance(attn_bias, torch.Tensor) else None
    output = F.scaled_dot_product_attention(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        attn_mask=mask,
        dropout_p=p,
        scale=scale,
    )
    return output.transpose(1, 2).contiguous()


def _compatibility_modules() -> dict[str, types.ModuleType]:
    xformers = types.ModuleType("xformers")
    xformers.__path__ = []
    ops = types.ModuleType("xformers.ops")
    fmha = types.ModuleType("xformers.ops.fmha")
    fmha.memory_efficient_attention = _memory_efficient_attention
    ops.AttentionBias = AttentionBias
    ops.fmha = fmha
    ops.memory_efficient_attention = _memory_efficient_attention
    xformers.ops = ops

    return {
        "xformers": xformers,
        "xformers.ops": ops,
        "xformers.ops.fmha": fmha,
    }


@contextlib.contextmanager
def xformers_import_compatibility() -> Iterator[None]:
    """Isolate perception_models imports from any compiled xFormers package.

    The temporary modules are removed (or prior modules restored) immediately after
    SAM-Audio imports. Perception Models keeps direct references to this narrow SDPA
    surface, while unrelated ComfyUI nodes retain access to their real xFormers install.
    """
    compatibility_modules = _compatibility_modules()
    missing = object()
    previous = {
        name: sys.modules.get(name, missing) for name in compatibility_modules
    }
    sys.modules.update(compatibility_modules)
    try:
        yield
    finally:
        for name, module in previous.items():
            if module is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
