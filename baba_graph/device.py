"""Device selection and CUDA / T4-oriented training settings."""

from __future__ import annotations

import torch

# Defaults tuned for 16 GB T4 (batch fits comfortably; AMP saves headroom).
T4_VQ_BATCH_SIZE = 512
T4_VQ_STEPS_PER_EPOCH = 200
T4_PREDICTOR_GRAD_ACCUM = 8


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


def is_cuda_device(device: str) -> bool:
    return device.startswith("cuda") and torch.cuda.is_available()


def configure_cuda_training(device: str) -> None:
    """cuDNN + matmul settings suited to NVIDIA T4 / datacenter GPUs."""
    if not is_cuda_device(device):
        return
    torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def resolve_amp(use_amp: str, device: str) -> bool:
    """``auto`` → AMP on CUDA (recommended for T4)."""
    if use_amp == "auto":
        return is_cuda_device(device)
    return use_amp in ("1", "true", "yes", "on")


def t4_vq_batch_size(device: str, requested: int | None) -> int:
    if requested is not None:
        return requested
    return T4_VQ_BATCH_SIZE if is_cuda_device(device) else 256


def t4_grad_accum(device: str, requested: int | None) -> int:
    if requested is not None:
        return max(1, requested)
    return T4_PREDICTOR_GRAD_ACCUM if is_cuda_device(device) else 1


def as_torch_device(device: str | torch.device) -> torch.device:
    return device if isinstance(device, torch.device) else torch.device(device)


def infer_device_from_tensors(*tensors: torch.Tensor | None) -> torch.device:
    """Device of first tensor found (including empty tensors)."""
    for t in tensors:
        if isinstance(t, torch.Tensor):
            return t.device
    return torch.device("cpu")


def module_device(module: torch.nn.Module) -> torch.device:
    """Canonical device for a ``nn.Module`` (after ``.to(cuda)``)."""
    return next(module.parameters()).device


def resolve_reference_device(
    ref: str | torch.device | torch.nn.Module | torch.Tensor,
) -> torch.device:
    """Map module, tensor, or device string to a ``torch.device``."""
    if isinstance(ref, torch.nn.Module):
        return module_device(ref)
    if isinstance(ref, torch.Tensor):
        return ref.device
    return as_torch_device(ref)


def action_tensor(
    action: int | torch.Tensor,
    device: str | torch.device | torch.nn.Module | torch.Tensor,
) -> torch.Tensor:
    """Scalar action index on the same device as the model / reference tensors."""
    dev = resolve_reference_device(device)
    if isinstance(action, torch.Tensor):
        return action.detach().long().to(dev).reshape(())
    return torch.tensor(int(action), dtype=torch.long, device=dev)


_SNAPSHOT_TENSOR_KEYS = (
    "text_x",
    "text_edge_index",
    "text_token_ids",
    "text_node_mask",
    "physical_x",
    "physical_edge_index",
    "physical_token_ids",
    "physical_node_mask",
    "codebook_size",
)


def snap_tensors_to_device(
    snap_tensors: dict,
    device: str | torch.device | torch.nn.Module | torch.Tensor,
) -> dict:
    """Move all tensor entries in a snapshot dict onto ``device`` (in-place copy)."""
    dev = resolve_reference_device(device)
    out = dict(snap_tensors)
    for key in _SNAPSHOT_TENSOR_KEYS:
        val = out.get(key)
        if isinstance(val, torch.Tensor) and val.device != dev:
            out[key] = val.to(dev, non_blocking=(dev.type == "cuda"))
    return out
