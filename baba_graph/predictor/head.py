"""Dual-headed transition predictor: identity + spatial movement."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from baba_graph.predictor.movement import NUM_ADJACENCY_CLASSES


@dataclass
class DualHeadOutput:
    identity_logits: torch.Tensor  # (N, codebook_size)
    movement_logits: torch.Tensor  # (N, 5) STAY/UP/DOWN/LEFT/RIGHT


class DualTransitionPredictorHead(nn.Module):
    """
    Phase 4 dual head:
      A) Identity — next VQ token (cross-entropy)
      B) Movement — adjacency class (cross-entropy)
    """

    def __init__(
        self,
        hidden_dim: int,
        codebook_size: int,
        *,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.codebook_size = codebook_size
        trunk = [
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        ]
        self.trunk = nn.Sequential(*trunk)
        self.identity_head = nn.Linear(hidden_dim, codebook_size)
        self.movement_head = nn.Linear(hidden_dim, NUM_ADJACENCY_CLASSES)

    def forward(self, physical_h: torch.Tensor) -> DualHeadOutput:
        if physical_h.numel() == 0:
            dev = physical_h.device
            return DualHeadOutput(
                identity_logits=torch.zeros(0, self.codebook_size, device=dev),
                movement_logits=torch.zeros(0, NUM_ADJACENCY_CLASSES, device=dev),
            )
        h = self.trunk(physical_h)
        return DualHeadOutput(
            identity_logits=self.identity_head(h),
            movement_logits=self.movement_head(h),
        )


# Back-compat alias
TransitionPredictorHead = DualTransitionPredictorHead
