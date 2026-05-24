"""VQ-VAE configuration (Phase 2)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class VQConfig:
    """Vector quantizer hyperparameters."""

    embed_dim: int = 128
    num_codes: int = 512
    max_codes: int = 2048
    commitment_beta: float = 0.25
    ema_decay: float = 0.99
    ema_epsilon: float = 1e-5
    # L2 distance above this triggers a new codebook entry (dynamic expansion).
    expansion_threshold: float = 1.5
    enable_expansion: bool = True
    # Dead-code revival: reset codes with EMA cluster size below this fraction of mean.
    dead_code_threshold: float = 0.01

    @property
    def codebook_dim(self) -> int:
        return self.embed_dim
