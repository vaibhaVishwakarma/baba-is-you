"""Phase 5 online exploration tests."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("scipy")
pyBaba = pytest.importorskip("pyBaba")

from baba_graph import extract_perception_from_game
from baba_graph.exploration.online import (
    OnlineExplorationConfig,
    TransitionReplayBuffer,
    build_mixed_batch,
    few_shot_adapt,
    seed_replay_from_rollout,
    TokenExpansionTracker,
)
from baba_graph.perception.properties_view import win_positions, you_positions
from baba_graph.predictor import BabaTransitionModel, PredictorConfig
from baba_graph.predictor.agent import score_predicted_state
from baba_graph.world_model import WorldModelConfig, snapshot_to_tensors
from baba_world.paths import map_path

MAP = "baba_is_you"


def test_you_win_from_properties_not_noun():
    game = pyBaba.Game(str(map_path(MAP)))
    game.Reset()
    snap = extract_perception_from_game(game, calibrate_classifier=False)
    you = you_positions(snap)
    win = win_positions(snap)
    assert len(you) >= 1
    assert len(win) >= 1


def test_mixed_batch_ratio():
    import random

    class Tr:
        pass

    buf = TransitionReplayBuffer(100)
    new = [Tr() for _ in range(10)]
    for _ in range(50):
        buf.push(Tr())
    rng = random.Random(0)
    batch = build_mixed_batch(new, buf, batch_size=20, new_fraction=0.2, rng=rng)
    assert len(batch) == 20
    n_new = sum(1 for x in batch if x in new)
    assert 2 <= n_new <= 6


def test_token_tracker():
    tr = TokenExpansionTracker()
    assert tr.observe_tokens(np.array([1, 2, 2])) == [1, 2]
    assert tr.observe_tokens(np.array([2, 3])) == [3]


def test_few_shot_adapt_runs():
    from baba_graph.predictor.data import collect_tokenized_transitions

    transitions = collect_tokenized_transitions(MAP, episodes=1, max_steps=12, seed=0)
    if len(transitions) < 4:
        pytest.skip("need transitions")
    buf = TransitionReplayBuffer(200)
    buf.extend(transitions)
    model = BabaTransitionModel(
        WorldModelConfig(hidden_dim=32, codebook_size=64, rule_layers=1, physical_layers=1),
        PredictorConfig(codebook_size=64),
    )
    result = few_shot_adapt(
        model,
        None,
        [99],
        buf,
        transitions[:4],
        config=OnlineExplorationConfig(adapt_steps=2, min_replay_for_adapt=4),
    )
    assert result.steps >= 1
    assert result.replay_samples > 0


def test_agent_scores_with_win_property():
    game = pyBaba.Game(str(map_path(MAP)))
    game.Reset()
    snap = extract_perception_from_game(game, calibrate_classifier=False)
    model = BabaTransitionModel(
        WorldModelConfig(hidden_dim=32, codebook_size=64, rule_layers=1, physical_layers=1),
        PredictorConfig(codebook_size=64),
    )
    t = snapshot_to_tensors(snap, codebook_size=64)
    s = score_predicted_state(model, snap, t, 0)
    assert isinstance(s, float)
