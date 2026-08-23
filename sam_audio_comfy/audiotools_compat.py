"""Import-only compatibility for DACVAE's legacy audiotools base classes."""

from __future__ import annotations

import contextlib
import sys
import types
from collections.abc import Iterator

import torch


class BaseModel(torch.nn.Module):
    """The only audiotools behavior DACVAE's inference model inherits."""

    INTERN: list[str] = []
    EXTERN: list[str] = []


class AudioSignal:
    """Marker used by DACVAE training and file-I/O helpers, not inference."""


class STFTParams:
    """Marker used by DACVAE loss/discriminator helpers, not inference."""


def _compatibility_modules() -> dict[str, types.ModuleType]:
    audiotools = types.ModuleType("audiotools")
    audiotools.__path__ = []
    ml = types.ModuleType("audiotools.ml")
    ml.BaseModel = BaseModel
    audiotools.ml = ml
    audiotools.AudioSignal = AudioSignal
    audiotools.STFTParams = STFTParams
    return {"audiotools": audiotools, "audiotools.ml": ml}


@contextlib.contextmanager
def audiotools_import_compatibility() -> Iterator[None]:
    """Temporarily provide DACVAE's unused legacy import surface."""
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
