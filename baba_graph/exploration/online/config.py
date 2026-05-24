"""Phase 5 online exploration / few-shot adaptation configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OnlineExplorationConfig:
    """Mixed-batch adaptation to avoid catastrophic forgetting."""

    new_data_fraction: float = 0.2
    replay_fraction: float = 0.8
    adapt_steps: int = 8
    adapt_lr: float = 5e-4
    sandbox_max_steps: int = 80
    sandbox_episodes: int = 3
    replay_buffer_capacity: int = 5000
    min_replay_for_adapt: int = 32

    def __post_init__(self) -> None:
        if abs(self.new_data_fraction + self.replay_fraction - 1.0) > 1e-6:
            raise ValueError("new_data_fraction + replay_fraction must equal 1.0")
