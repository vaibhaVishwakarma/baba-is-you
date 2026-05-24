"""Structured transition types (canonical training interface)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TransitionSample:
    """Single causal transition (canonical training record)."""

    state: np.ndarray
    action: int
    next_state: np.ndarray
    done: bool
    step_id: int
    episode_id: int
    rules_before: tuple[str, ...] = ()
    rules_after: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "action": int(self.action),
            "next_state": self.next_state,
            "done": bool(self.done),
            "step_id": int(self.step_id),
            "episode_id": int(self.episode_id),
            "rules_before": self.rules_before,
            "rules_after": self.rules_after,
        }


@dataclass
class Transition:
    """Alias-style transition without episode metadata."""

    state: np.ndarray
    action: int
    next_state: np.ndarray
    done: bool = False


@dataclass
class Episode:
    """Full episode as a list of transitions."""

    map_name: str
    transitions: list[TransitionSample] = field(default_factory=list)
    episode_id: int = 0

    def __len__(self) -> int:
        return len(self.transitions)

    @property
    def done(self) -> bool:
        return bool(self.transitions) and self.transitions[-1].done
