"""Phase 3 dual-graph world model configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class WorldModelConfig:
    """Hyperparameters for rule GNN + physical MPNN."""

    embed_dim: int = 128
    hidden_dim: int = 256
    rule_layers: int = 2
    physical_layers: int = 3
    action_dim: int = 4
    dropout: float = 0.1
    codebook_size: int = 512
    rule_aggregation: str = "attention"  # attention | max | sum
    use_vq_tokens: bool = True  # if False, use continuous visual vectors
