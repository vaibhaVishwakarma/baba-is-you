"""VQ codebook training on perception visual vectors."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import torch

from baba_graph.device import T4_VQ_STEPS_PER_EPOCH, is_cuda_device
from baba_graph.vq.config import VQConfig
from baba_graph.vq.data import VisualBatch, collect_visual_vectors
from baba_graph.vq.quantizer import DynamicVectorQuantizer


@dataclass
class TrainStats:
    epoch: int
    loss: float
    commitment: float
    codebook: float
    perplexity: float
    num_codes: int
    expansions: int


def train_vq(
    quantizer: DynamicVectorQuantizer,
    data: VisualBatch,
    *,
    epochs: int = 30,
    batch_size: int = 256,
    steps_per_epoch: int = 100,
    device: str = "cpu",
    use_amp: bool = False,
) -> list[TrainStats]:
    """Train VQ codebook via EMA updates (Phase 1 visual vectors are fixed)."""
    quantizer.to(device)
    quantizer.train()
    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if use_amp and is_cuda_device(device)
        else nullcontext()
    )

    history: list[TrainStats] = []
    if len(data) == 0:
        return history

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_commit = 0.0
        epoch_code = 0.0
        epoch_perp = 0.0
        for _ in range(steps_per_epoch):
            z = data.sample_torch(batch_size, device=device)
            with amp_ctx:
                out = quantizer(z)
            epoch_loss += float((out.commitment_loss + out.codebook_loss).item())
            epoch_commit += float(out.commitment_loss.item())
            epoch_code += float(out.codebook_loss.item())
            epoch_perp += float(out.perplexity.item())

        history.append(
            TrainStats(
                epoch=epoch + 1,
                loss=epoch_loss / steps_per_epoch,
                commitment=epoch_commit / steps_per_epoch,
                codebook=epoch_code / steps_per_epoch,
                perplexity=epoch_perp / steps_per_epoch,
                num_codes=quantizer.num_codes,
                expansions=quantizer.expansions_total,
            )
        )

    return history


def collect_and_train(
    *,
    maps: list[str] | None = None,
    vq_config: VQConfig | None = None,
    epochs: int = 30,
    batch_size: int = 256,
    episodes_per_map: int = 50,
    max_steps: int = 100,
    policy: str = "biased",
    device: str = "cpu",
    use_amp: bool = False,
    steps_per_epoch: int | None = None,
) -> tuple[DynamicVectorQuantizer, VisualBatch, list[TrainStats]]:
    from baba_graph.vision.config import VisualEncoderConfig

    vq_config = vq_config or VQConfig()
    enc_device = device if is_cuda_device(device) else "cpu"
    data = collect_visual_vectors(
        maps=maps,
        episodes_per_map=episodes_per_map,
        max_steps=max_steps,
        policy=policy,
        config=VisualEncoderConfig(device=enc_device),
    )
    quantizer = DynamicVectorQuantizer(vq_config).to(device)
    spe = steps_per_epoch if steps_per_epoch is not None else (
        T4_VQ_STEPS_PER_EPOCH if is_cuda_device(device) else 100
    )
    history = train_vq(
        quantizer,
        data,
        epochs=epochs,
        batch_size=batch_size,
        steps_per_epoch=spe,
        device=device,
        use_amp=use_amp,
    )
    quantizer.eval()
    return quantizer, data, history


def save_quantizer(quantizer: DynamicVectorQuantizer, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": quantizer.config,
            "state": {
                "embedding": quantizer.embedding[: quantizer.num_codes].cpu(),
                "ema_cluster_size": quantizer.ema_cluster_size[: quantizer.num_codes].cpu(),
                "ema_embedding": quantizer.ema_embedding[: quantizer.num_codes].cpu(),
                "num_codes": quantizer.num_codes,
                "expansions_total": quantizer.expansions_total,
            },
        },
        path,
    )
    return path


def load_quantizer(path: str | Path, device: str = "cpu") -> DynamicVectorQuantizer:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    config = ckpt["config"]
    q = DynamicVectorQuantizer(config).to(device)
    s = ckpt["state"]
    k = s["num_codes"]
    q._num_codes = k
    q.embedding[:k] = s["embedding"]
    q.ema_cluster_size[:k] = s["ema_cluster_size"]
    q.ema_embedding[:k] = s["ema_embedding"]
    q.expansions_total = s.get("expansions_total", 0)
    q.eval()
    return q
