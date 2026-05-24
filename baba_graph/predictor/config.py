"""Phase 4 discrete transition predictor configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PredictorConfig:
    codebook_size: int = 512
    hidden_dim: int = 256
    dropout: float = 0.1
    label_smoothing: float = 0.05
    movement_loss_weight: float = 1.0
