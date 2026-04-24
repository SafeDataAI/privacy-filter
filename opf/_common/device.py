"""Device-name resolution helpers shared by CLI entrypoints."""

from __future__ import annotations

import sys

import torch

AUTO_DEVICE: str = "auto"


def resolve_device(device_name: str) -> torch.device:
    """Resolve a user-supplied device name into a concrete ``torch.device``.

    ``"auto"`` selects the best available device: CUDA if a GPU is detected,
    otherwise CPU. Any other value is passed through to ``torch.device`` as-is
    so that explicit requests like ``"cuda"`` or ``"cpu"`` still fail loudly
    when the underlying backend is unavailable.
    """
    if device_name == AUTO_DEVICE:
        if torch.cuda.is_available():
            return torch.device("cuda")
        print(
            "info: no CUDA device detected; falling back to CPU "
            "(pass --device cuda to override).",
            file=sys.stderr,
            flush=True,
        )
        return torch.device("cpu")
    return torch.device(device_name)
