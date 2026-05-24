"""Device selection for training scripts."""

from __future__ import annotations

import torch


def resolve_device(request: str = "auto") -> str:
    """
    Resolve training device.

    - ``auto``: cuda if available, else cpu
    - ``cuda`` / ``cuda:0``: GPU (falls back to cpu if unavailable)
    - ``cpu``: CPU only
    """
    req = request.strip().lower()
    if req == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if req.startswith("cuda"):
        if torch.cuda.is_available():
            return request
        return "cpu"
    return "cpu"
