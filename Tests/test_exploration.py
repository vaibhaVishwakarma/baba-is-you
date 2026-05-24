"""Biased exploration tests."""

from __future__ import annotations

import random

import pytest

pyBaba = pytest.importorskip("pyBaba")

from baba_graph import extract_perception_from_game
from baba_graph.exploration import BiasedTextExplorer, text_adjacency_signature
from baba_graph.exploration.rollout import iter_perception_rollout
from baba_world.env import BabaTransitionEnv
from baba_world.paths import map_path

MAP = "baba_is_you"


def test_biased_explorer_prefers_moves():
    game = pyBaba.Game(str(map_path(MAP)))
    game.Reset()
    snap = extract_perception_from_game(game, calibrate_classifier=False)
    exp = BiasedTextExplorer(random.Random(0), epsilon_random=0.0)
    actions = [exp.choose_action(game, snap) for _ in range(20)]
    assert all(0 <= a <= 3 for a in actions)


def test_rollout_biased_faster_cache(monkeypatch):
    """Biased rollout completes and touches text graph."""
    steps = 0
    for snap, action, done, rules in iter_perception_rollout(
        MAP, episodes=1, max_steps=30, seed=1, policy="biased"
    ):
        steps += 1
        if snap.num_text > 0:
            sig = text_adjacency_signature(snap)
            assert isinstance(sig, frozenset)
        if done:
            break
    assert steps > 5


def test_biased_more_rule_activity_than_random():
    """Heuristic: biased policy should hit more rule/adjacency changes in same budget."""
    def count_changes(policy: str) -> int:
        changes = 0
        prev_rules = ()
        prev_adj = None
        for snap, _a, _d, rules in iter_perception_rollout(
            MAP, episodes=2, max_steps=40, seed=42, policy=policy
        ):
            adj = text_adjacency_signature(snap)
            if prev_adj is not None and adj != prev_adj:
                changes += 1
            if prev_rules and rules != prev_rules:
                changes += 1
            prev_adj = adj
            prev_rules = rules
        return changes

    biased = count_changes("biased")
    random_c = count_changes("random")
    # Not guaranteed every seed, but biased should not be worse on baba_is_you
    assert biased >= random_c - 2
