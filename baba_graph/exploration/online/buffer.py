"""Historical replay buffer for mixed-batch Phase 5 updates."""

from __future__ import annotations

import random
from collections import deque

from baba_graph.world_model.data import DynamicsTransition


class TransitionReplayBuffer:
    """FIFO buffer of dynamics transitions (known-object experience)."""

    def __init__(self, capacity: int = 5000) -> None:
        self.capacity = capacity
        self._buf: deque[DynamicsTransition] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._buf)

    def push(self, transition: DynamicsTransition) -> None:
        self._buf.append(transition)

    def extend(self, transitions: list[DynamicsTransition]) -> None:
        for tr in transitions:
            self.push(tr)

    def sample(self, n: int, *, rng: random.Random | None = None) -> list[DynamicsTransition]:
        rng = rng or random.Random()
        if not self._buf:
            return []
        n = min(n, len(self._buf))
        return rng.sample(list(self._buf), n)

    def clear(self) -> None:
        self._buf.clear()
