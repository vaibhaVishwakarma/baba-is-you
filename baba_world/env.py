"""Minimal world-transition environment (no pygame, gym, or rewards)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyBaba

from baba_world.actions import decode_action
from baba_world.paths import map_path
from baba_world.rules import extract_active_rules
from baba_world.state import StateExtractor
from baba_world.types import TransitionSample


class BabaTransitionEnv:
    """
    World-transition-centric simulator wrapper.

    Only exposes reset/step and structured (state, action, next_state, done).
    """

    def __init__(self, map_name: str | Path) -> None:
        self._map_path = map_path(str(map_name))
        self._map_key = self._map_path.stem
        self._extractor = StateExtractor()
        self._game: pyBaba.Game | None = None
        self._step_id = 0
        self._episode_id = 0
        self._last_rules: tuple[str, ...] = ()

    @property
    def map_name(self) -> str:
        return self._map_key

    @property
    def state_shape(self) -> tuple[int, int, int]:
        assert self._game is not None
        return self._extractor.shape(self._game)

    def reset(self, *, episode_id: int | None = None) -> np.ndarray:
        if episode_id is not None:
            self._episode_id = episode_id
        self._game = pyBaba.Game(str(self._map_path))
        self._game.Reset()
        self._step_id = 0
        self._last_rules = extract_active_rules(self._game)
        return self._extractor.extract(self._game)

    def step(self, action: int) -> TransitionSample:
        if self._game is None:
            raise RuntimeError("Call reset() before step()")

        state = self._extractor.extract(self._game)
        rules_before = self._last_rules

        self._game.MovePlayer(decode_action(action))
        next_state = self._extractor.extract(self._game)
        rules_after = extract_active_rules(self._game)
        self._last_rules = rules_after

        play_state = self._game.GetPlayState()
        done = play_state in (pyBaba.PlayState.WON, pyBaba.PlayState.LOST)

        sample = TransitionSample(
            state=state,
            action=int(action),
            next_state=next_state,
            done=done,
            step_id=self._step_id,
            episode_id=self._episode_id,
            rules_before=rules_before,
            rules_after=rules_after,
        )
        self._step_id += 1
        return sample

    def clone_state(self) -> np.ndarray:
        """Current state tensor (e.g. for validation)."""
        if self._game is None:
            raise RuntimeError("Call reset() before clone_state()")
        return self._extractor.extract(self._game)
