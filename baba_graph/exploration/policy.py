"""Biased exploration targeting text blocks and rule mutations."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import pyBaba

from baba_graph.exploration.adjacency import (
    adjacency_changed,
    rules_changed,
    text_adjacency_signature,
)
from baba_graph.perception.types import PerceptionSnapshot
from baba_world.actions import ACTION_COUNT

# action id -> (dx, dy)
_ACTION_DELTA = {
    0: (0, -1),  # UP
    1: (0, 1),   # DOWN
    2: (-1, 0),  # LEFT
    3: (1, 0),   # RIGHT
}


@dataclass
class ExplorerStats:
    steps: int = 0
    adjacency_changes: int = 0
    rule_changes: int = 0
    text_pushes_attempted: int = 0


@dataclass
class BiasedTextExplorer:
    """
    Exploration policy for data collection (VQ + world model).

    Prefers moves that approach text, push text blocks, and (when observed)
    recently yielded adjacency or rule changes.
    """

    rng: random.Random
    epsilon_random: float = 0.08
    rule_change_boost: float = 3.0
    adjacency_change_boost: float = 2.0
    stats: ExplorerStats = field(default_factory=ExplorerStats)
    _last_adj: frozenset | None = None
    _last_rules: tuple[str, ...] = ()
    _boost_actions: set[int] = field(default_factory=set)

    def reset(self) -> None:
        self._last_adj = None
        self._last_rules = ()
        self._boost_actions.clear()

    def observe_snapshot(
        self,
        snap: PerceptionSnapshot,
        *,
        rules: tuple[str, ...] = (),
    ) -> None:
        """Call after each env step with the new perception + rules."""
        adj = text_adjacency_signature(snap)
        if self._last_adj is not None and adjacency_changed(self._last_adj, adj):
            self.stats.adjacency_changes += 1
        if self._last_rules and rules_changed(self._last_rules, rules):
            self.stats.rule_changes += 1
        self._last_adj = adj
        self._last_rules = rules

    def choose_action(
        self,
        game: pyBaba.Game,
        snap: PerceptionSnapshot | None = None,
    ) -> int:
        self.stats.steps += 1
        if self.rng.random() < self.epsilon_random:
            return self.rng.randint(0, ACTION_COUNT - 1)

        positions = _player_positions(game)
        if not positions:
            return self.rng.randint(0, ACTION_COUNT - 1)

        px, py = positions[0]
        text_cells = _text_cells(game, snap)
        if not text_cells:
            return self.rng.randint(0, ACTION_COUNT - 1)

        scores = [
            self._score_action(game, px, py, action, text_cells)
            for action in range(ACTION_COUNT)
        ]
        best = max(scores)
        top = [a for a, s in enumerate(scores) if s >= best - 1e-6]
        if self._boost_actions:
            boosted = [a for a in top if a in self._boost_actions]
            if boosted:
                top = boosted
        return self.rng.choice(top)

    def _score_action(
        self,
        game: pyBaba.Game,
        px: int,
        py: int,
        action: int,
        text_cells: set[tuple[int, int]],
    ) -> float:
        dx, dy = _ACTION_DELTA[action]
        nx, ny = px + dx, py + dy
        score = 0.0

        # Approach nearest text (Manhattan)
        if text_cells:
            before = min(abs(px - tx) + abs(py - ty) for tx, ty in text_cells)
            after = min(abs(nx - tx) + abs(ny - ty) for tx, ty in text_cells)
            score += float(before - after) * 2.0

        # Push opportunity: text in target cell
        if (nx, ny) in text_cells:
            score += 8.0
            self.stats.text_pushes_attempted += 1
            # Bonus if push would align text with another text (rule chain forming)
            for tx, ty in text_cells:
                if (tx, ty) == (nx, ny):
                    continue
                if abs((nx + dx) - tx) + abs((ny + dy) - ty) <= 1:
                    score += 12.0

        # Avoid walking into STOP walls when detectable
        if _cell_has_stop(game, nx, ny):
            score -= 15.0

        if action in self._boost_actions:
            score += self.adjacency_change_boost

        return score

    def register_successful_action(self, action: int, *, rule_change: bool) -> None:
        """Bias toward actions that recently changed rules/adjacency."""
        self._boost_actions.add(action)
        if rule_change:
            self._boost_actions.add(action)
        if len(self._boost_actions) > 4:
            self._boost_actions = set(list(self._boost_actions)[-4:])


def _player_positions(game: pyBaba.Game) -> list[tuple[int, int]]:
    icon = game.GetPlayerIcon()
    raw = game.GetMap().GetPositions(icon)
    if not raw:
        return []
    return [(int(p[0]), int(p[1])) for p in raw]


def _text_cells(
    game: pyBaba.Game,
    snap: PerceptionSnapshot | None,
) -> set[tuple[int, int]]:
    if snap is not None:
        return {(n.x, n.y) for n in snap.text.nodes}
    cells: set[tuple[int, int]] = set()
    m = game.GetMap()
    for y in range(m.GetHeight()):
        for x in range(m.GetWidth()):
            for t in m.At(x, y).GetTypes():
                if pyBaba.IsTextType(t):
                    cells.add((x, y))
    return cells


def _cell_has_stop(game: pyBaba.Game, x: int, y: int) -> bool:
    m = game.GetMap()
    if x < 0 or y < 0 or x >= m.GetWidth() or y >= m.GetHeight():
        return True
    for t in m.At(x, y).GetTypes():
        if t == pyBaba.ObjectType.WALL:
            return True
    return False
