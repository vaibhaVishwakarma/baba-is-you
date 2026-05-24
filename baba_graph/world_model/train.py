"""Minimal training utilities for Phase 3 (representation warmup)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from baba_graph.world_model.config import WorldModelConfig
from baba_graph.world_model.data import DynamicsTransition
from baba_graph.world_model.model import DualGraphWorldModel, snapshot_to_tensors


@dataclass
class TrainStepStats:
    loss: float
    num_transitions: int


def contrastive_next_state_loss(
    model: DualGraphWorldModel,
    transition: DynamicsTransition,
    *,
    device: str = "cpu",
    codebook_size: int | None = None,
) -> torch.Tensor:
    """
    Lightweight self-supervised signal: next physical embeddings should be
    predictable from (state, action) encoding.

    Phase 4 will replace this with categorical token prediction.
    """
    cb = codebook_size or model.config.codebook_size
    s = snapshot_to_tensors(transition.state, device=device, codebook_size=cb)
    ns = snapshot_to_tensors(transition.next_state, device=device, codebook_size=cb)
    out = model.from_snapshot_tensors(s, transition.action)
    target = ns["physical_x"].detach()
    if target.numel() == 0 or out.physical_h.numel() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    dim = min(out.physical_h.size(-1), target.size(-1))
    n = min(out.physical_h.size(0), target.size(0))
    if n == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    return nn.functional.mse_loss(
        out.physical_h[:n, :dim],
        target[:n, :dim],
    )


def train_representation_warmup(
    model: DualGraphWorldModel,
    transitions: list[DynamicsTransition],
    *,
    epochs: int = 5,
    lr: float = 1e-3,
    device: str = "cpu",
) -> list[TrainStepStats]:
    """Few epochs of MSE warmup before Phase 4 discrete head."""
    model.to(device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history: list[TrainStepStats] = []

    for epoch in range(epochs):
        total = 0.0
        count = 0
        for tr in transitions:
            opt.zero_grad()
            loss = contrastive_next_state_loss(model, tr, device=device)
            if loss.requires_grad:
                loss.backward()
                opt.step()
            total += float(loss.item())
            count += 1
        history.append(
            TrainStepStats(
                loss=total / max(count, 1),
                num_transitions=count,
            )
        )
    model.eval()
    return history
