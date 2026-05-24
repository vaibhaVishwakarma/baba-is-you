"""Phase 4 dual-head predictor tests."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("scipy")
pyBaba = pytest.importorskip("pyBaba")

from baba_graph import extract_perception_from_game
from baba_graph.predictor import (
    BabaTransitionModel,
    PredictorConfig,
    transition_loss,
)
from baba_graph.predictor.data import collect_tokenized_transitions
from baba_graph.predictor.movement import AdjacencyMove, movement_label
from baba_graph.world_model import WorldModelConfig
from baba_graph.world_model.model import snapshot_to_tensors
from baba_world.paths import map_path

MAP = "baba_is_you"


def test_dual_head_shapes():
    game = pyBaba.Game(str(map_path(MAP)))
    game.Reset()
    snap = extract_perception_from_game(game, calibrate_classifier=False)
    cfg = WorldModelConfig(hidden_dim=32, codebook_size=64)
    model = BabaTransitionModel(cfg, PredictorConfig(codebook_size=64))
    t = snapshot_to_tensors(snap, codebook_size=64)
    out = model(t, 0)
    assert out.identity_logits.shape == (snap.num_physical, 64)
    assert out.movement_logits.shape == (snap.num_physical, 5)


def test_forward_cuda_action_on_same_device():
    if not torch.cuda.is_available():
        pytest.skip("cuda not available")
    game = pyBaba.Game(str(map_path(MAP)))
    game.Reset()
    snap = extract_perception_from_game(game, calibrate_classifier=False)
    model = BabaTransitionModel(
        WorldModelConfig(hidden_dim=32, codebook_size=64, rule_layers=1, physical_layers=1),
        PredictorConfig(codebook_size=64),
    ).cuda()
    t = snapshot_to_tensors(snap, device="cuda", codebook_size=64)
    out = model(t, 0)
    assert out.identity_logits.device.type == "cuda"
    assert out.movement_logits.device.type == "cuda"


def test_movement_label_cardinal():
    import pyBaba
    from baba_graph.types import ObjectNode

    a = ObjectNode(0, 2, 2, pyBaba.ObjectType.ICON_ROCK, "ROCK", False, True, False)
    b = ObjectNode(1, 2, 1, pyBaba.ObjectType.ICON_ROCK, "ROCK", False, True, False)
    assert movement_label(a, b) == int(AdjacencyMove.UP)


def test_dual_loss_finite():
    tr = collect_tokenized_transitions(MAP, episodes=1, max_steps=20, seed=0)
    if not tr:
        pytest.skip("no transitions")
    model = BabaTransitionModel(
        WorldModelConfig(hidden_dim=32, codebook_size=128, rule_layers=1, physical_layers=1),
        PredictorConfig(codebook_size=128),
    )
    loss, metrics = transition_loss(model, tr[0])
    assert torch.isfinite(loss)
    assert metrics["num_pairs"] >= 0


def test_transition_loss_cuda():
    if not torch.cuda.is_available():
        pytest.skip("cuda not available")
    tr = collect_tokenized_transitions(MAP, episodes=1, max_steps=20, seed=0)
    if not tr:
        pytest.skip("no transitions")
    model = BabaTransitionModel(
        WorldModelConfig(hidden_dim=32, codebook_size=64, rule_layers=1, physical_layers=1),
        PredictorConfig(codebook_size=64),
    ).cuda()
    loss, metrics = transition_loss(model, tr[0], device="cuda", use_amp=True)
    assert loss.device.type == "cuda"
    assert torch.isfinite(loss)
    assert metrics["num_pairs"] > 0, "need aligned pairs to exercise CUDA target gather"


def test_train_one_epoch():
    tr = collect_tokenized_transitions(MAP, episodes=1, max_steps=15, seed=1)
    model = BabaTransitionModel(
        WorldModelConfig(hidden_dim=32, codebook_size=64, rule_layers=1, physical_layers=1),
        PredictorConfig(codebook_size=64),
    )
    from baba_graph.predictor import train_predictor

    h = train_predictor(model, tr, epochs=1)
    assert h[0].num_pairs >= 0
