"""Device-name resolution helpers shared by CLI entrypoints."""

from __future__ import annotations

import sys

import torch

AUTO_DEVICE: str = "auto"


def _mps_is_available() -> bool:
    """Return True when the current PyTorch build supports Apple Metal (MPS)."""
    backend = getattr(torch.backends, "mps", None)
    if backend is None:
        return False
    is_available = getattr(backend, "is_available", None)
    if is_available is None:
        return False
    try:
        return bool(is_available())
    except Exception:
        return False


def resolve_device(device_name: str) -> torch.device:
    """Resolve a user-supplied device name into a concrete ``torch.device``.

    ``"auto"`` selects the best available device in this order: CUDA (NVIDIA
    GPU) > MPS (Apple Silicon GPU) > CPU. Any other value is passed through
    to ``torch.device`` as-is so that explicit requests like ``"cuda"`` or
    ``"mps"`` still fail loudly when the underlying backend is unavailable.
    """
    if device_name == AUTO_DEVICE:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if _mps_is_available():
            print(
                "info: no CUDA device detected; using Apple Metal (MPS).",
                file=sys.stderr,
                flush=True,
            )
            return torch.device("mps")
        print(
            "info: no CUDA or MPS device detected; falling back to CPU "
            "(pass --device cuda or --device mps to override).",
            file=sys.stderr,
            flush=True,
        )
        return torch.device("cpu")
    return torch.device(device_name)
