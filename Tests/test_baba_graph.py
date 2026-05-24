"""Phase 1 visual graph extraction tests."""

from __future__ import annotations

import numpy as np
import pytest

pyBaba = pytest.importorskip("pyBaba")
torch = pytest.importorskip("torch")

from baba_graph import (
    VisualEncoderConfig,
    extract_graph_from_game,
    extract_perception_from_game,
    game_to_pyg_data,
    rollout_graph_episode,
    run_graph_validations,
)
from baba_graph.perception.build import physical_feature_dim, text_feature_dim
from baba_graph.vision.patches import PatchRenderer
from baba_world.env import BabaTransitionEnv
from baba_world.paths import map_path

MAP = "baba_is_you"


def test_feature_dim_layout():
    cfg = VisualEncoderConfig(embed_dim=64)
    assert text_feature_dim(cfg.embed_dim) == 64 + 2
    assert physical_feature_dim(cfg.embed_dim) == 64 + 2 + 12


def test_graph_node_count():
    game = pyBaba.Game(str(map_path(MAP)))
    game.Reset()
    snap = extract_graph_from_game(game)
    assert snap.graph.num_nodes > 0
    assert snap.graph.x.shape == (snap.graph.num_nodes, snap.graph.feature_dim)
    assert snap.graph.visual_dim == 128


def test_visual_block_nonzero():
    snap = extract_graph_from_game(pyBaba.Game(str(map_path(MAP))))
    visual = snap.graph.x[:, : snap.graph.visual_dim]
    assert np.abs(visual).sum() > 0


def test_open_vocabulary_unknown_type():
    from baba_graph.vision.patches import PatchRenderer
    from baba_graph.vision.perception_encoder import get_perception_encoder

    enc = get_perception_encoder(VisualEncoderConfig(), force_new=True)
    renderer = PatchRenderer()
    p = renderer.render("COMPLETELY_NEW_THING", is_text=False)
    e = enc.encode_visual(p[np.newaxis, ...])[0]
    assert e.shape == (128,)
    assert np.isfinite(e).all()


def test_pyg_data_conversion():
    pytest.importorskip("torch_geometric")
    game = pyBaba.Game(str(map_path(MAP)))
    game.Reset()
    data = game_to_pyg_data(game)
    assert data.x.shape[0] == data.num_nodes
    assert data.edge_index.shape[0] == 2
    assert hasattr(data, "visual_dim")


def test_deterministic_extraction():
    game = pyBaba.Game(str(map_path(MAP)))
    game.Reset()
    a = extract_perception_from_game(game, calibrate_classifier=False)
    b = extract_perception_from_game(game, calibrate_classifier=False)
    np.testing.assert_allclose(a.text.visual, b.text.visual)
    np.testing.assert_allclose(a.physical.visual, b.physical.visual)


def test_graph_rollout():
    transitions = rollout_graph_episode(MAP, max_steps=10, seed=1)
    assert len(transitions) >= 1


def test_validation_suite():
    report = run_graph_validations(MAP)
    assert report.passed, report.summary()


def test_env_step_graph_shapes():
    env = BabaTransitionEnv(MAP)
    env.reset()
    game = env._game
    before = extract_graph_from_game(game).graph.x.shape
    env.step(0)
    after = extract_graph_from_game(game).graph.x.shape
    assert before == after
