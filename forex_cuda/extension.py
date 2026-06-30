from __future__ import annotations

import importlib
from types import ModuleType


_EXTENSION: ModuleType | None = None


def load_extension() -> ModuleType:
    global _EXTENSION
    if _EXTENSION is not None:
        return _EXTENSION

    try:
        import torch  # noqa: F401

        _EXTENSION = importlib.import_module("forex_cuda_ext")
    except ImportError as exc:
        raise ImportError(
            "CUDA extension 'forex_cuda_ext' is not built. "
            "Install this package with `pip install -e .` from the project root "
            "in an environment with PyTorch CUDA and nvcc available."
        ) from exc
    return _EXTENSION
