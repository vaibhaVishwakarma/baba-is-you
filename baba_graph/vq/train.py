"""VQ codebook training on perception visual vectors."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

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
) -> list[TrainStats]:
    """Train VQ codebook via EMA updates (Phase 1 visual vectors are fixed)."""
    quantizer.to(device)
    quantizer.train()

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
) -> tuple[DynamicVectorQuantizer, VisualBatch, list[TrainStats]]:
    vq_config = vq_config or VQConfig()
    data = collect_visual_vectors(
        maps=maps,
        episodes_per_map=episodes_per_map,
        max_steps=max_steps,
        policy=policy,
    )
    quantizer = DynamicVectorQuantizer(vq_config).to(device)
    history = train_vq(quantizer, data, epochs=epochs, batch_size=batch_size, device=device)
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
