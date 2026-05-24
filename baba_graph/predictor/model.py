"""Full pipeline: Phase 3 dynamics + Phase 4 dual transition head."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from baba_graph.device import action_tensor
from baba_graph.predictor.config import PredictorConfig
from baba_graph.predictor.head import DualHeadOutput, DualTransitionPredictorHead
from baba_graph.world_model.config import WorldModelConfig
from baba_graph.world_model.model import DualGraphWorldModel, WorldModelOutput


@dataclass
class PredictorOutput:
    dynamics: WorldModelOutput
    identity_logits: torch.Tensor  # (N_phys, codebook_size)
    movement_logits: torch.Tensor  # (N_phys, 5)

    @property
    def logits(self) -> torch.Tensor:
        """Backward-compatible alias for identity head."""
        return self.identity_logits


class BabaTransitionModel(nn.Module):
    """Dual-graph world model + dual-headed transition predictor."""

    def __init__(
        self,
        world_config: WorldModelConfig | None = None,
        predictor_config: PredictorConfig | None = None,
    ) -> None:
        super().__init__()
        self.world_config = world_config or WorldModelConfig()
        self.predictor_config = predictor_config or PredictorConfig()
        self.predictor_config.codebook_size = self.world_config.codebook_size

        self.dynamics = DualGraphWorldModel(self.world_config)
        self.predictor = DualTransitionPredictorHead(
            self.world_config.hidden_dim,
            self.predictor_config.codebook_size,
            dropout=self.predictor_config.dropout,
        )

    def forward(
        self,
        snap_tensors: dict,
        action: torch.Tensor | int,
    ) -> PredictorOutput:
        action = action_tensor(action, self)
        dyn = self.dynamics.from_snapshot_tensors(snap_tensors, action)
        heads = self.predictor(dyn.physical_h)
        return PredictorOutput(
            dynamics=dyn,
            identity_logits=heads.identity_logits,
            movement_logits=heads.movement_logits,
        )

    def predict_heads(self, snap_tensors: dict, action: int | torch.Tensor) -> DualHeadOutput:
        """Return raw dual-head output without dynamics wrapper."""
        out = self.forward(snap_tensors, action)
        return DualHeadOutput(out.identity_logits, out.movement_logits)
