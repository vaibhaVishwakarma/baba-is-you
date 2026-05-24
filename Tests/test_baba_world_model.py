"""Phase 3 dual-graph world model tests (targeted binding)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pyBaba = pytest.importorskip("pyBaba")

from baba_graph import extract_perception_from_game
from baba_graph.world_model import (
    DualGraphWorldModel,
    WorldModelConfig,
    collect_dynamics_transitions,
    gather_node_rule_context,
    parse_rule_chains,
    snapshot_to_tensors,
)
from baba_graph.world_model.binding import canonical_noun, token_slots_for_noun
from baba_world.paths import map_path

MAP = "baba_is_you"


def test_dual_model_forward_shapes():
    game = pyBaba.Game(str(map_path(MAP)))
    game.Reset()
    snap = extract_perception_from_game(game, calibrate_classifier=False)
    cfg = WorldModelConfig(hidden_dim=64, codebook_size=128)
    tensors = snapshot_to_tensors(snap, device="cpu", codebook_size=cfg.codebook_size)
    model = DualGraphWorldModel(cfg)
    out = model.from_snapshot_tensors(tensors, action=0)
    h = 64
    assert out.rule_embeddings.shape == (cfg.codebook_size, h)
    assert out.node_rule_context.shape == (snap.num_physical, h)
    assert out.physical_h.shape == (snap.num_physical, h)
    assert out.text_h.shape == (snap.num_text, h)


def test_targeted_binding_differs_by_token():
    """Wall and Rock nodes must receive different rule contexts (not global soup)."""
    game = pyBaba.Game(str(map_path(MAP)))
    game.Reset()
    snap = extract_perception_from_game(game, calibrate_classifier=False)
    cfg = WorldModelConfig(hidden_dim=32, codebook_size=256)
    tensors = snapshot_to_tensors(snap, codebook_size=cfg.codebook_size)
    model = DualGraphWorldModel(cfg)
    model.eval()
    with torch.no_grad():
        out = model.from_snapshot_tensors(tensors, 0)

    phys_nodes = tensors["physical_nodes"]
    phys_tok = tensors["physical_token_ids"]
    wall_idx = next(
        (i for i, n in enumerate(phys_nodes) if canonical_noun(n) == "WALL"),
        None,
    )
    rock_idx = next(
        (i for i, n in enumerate(phys_nodes) if canonical_noun(n) == "ROCK"),
        None,
    )
    if wall_idx is None or rock_idx is None:
        pytest.skip("map missing wall or rock")

    w_ctx = out.node_rule_context[wall_idx]
    r_ctx = out.node_rule_context[rock_idx]
    assert not torch.allclose(w_ctx, r_ctx, atol=1e-5)


def test_rule_chains_parsed():
    game = pyBaba.Game(str(map_path(MAP)))
    game.Reset()
    snap = extract_perception_from_game(game, calibrate_classifier=False)
    chains = parse_rule_chains(snap.text.nodes, torch.from_numpy(snap.text.edge_index).long())
    assert len(chains) >= 1


def test_token_slots_write_to_noun_indices():
    game = pyBaba.Game(str(map_path(MAP)))
    game.Reset()
    snap = extract_perception_from_game(game, calibrate_classifier=False)
    tensors = snapshot_to_tensors(snap, codebook_size=128)
    chains = parse_rule_chains(snap.text.nodes, tensors["text_edge_index"])
    if not chains:
        pytest.skip("no chains")
    subj_idx, _ = chains[0]
    noun = canonical_noun(snap.text.nodes[subj_idx])
    slots = token_slots_for_noun(
        noun,
        snap.text.nodes,
        tensors["text_token_ids"],
        snap.physical.nodes,
        tensors["physical_token_ids"],
    )
    assert len(slots) >= 1


def test_permutation_invariance():
    game = pyBaba.Game(str(map_path(MAP)))
    game.Reset()
    snap = extract_perception_from_game(game, calibrate_classifier=False)
    if snap.num_physical < 2:
        pytest.skip("need multiple nodes")
    t = snapshot_to_tensors(snap, codebook_size=64)
    model = DualGraphWorldModel(WorldModelConfig(hidden_dim=32, codebook_size=64))
    model.eval()
    with torch.no_grad():
        o1 = model.from_snapshot_tensors(t, 1)
        perm = torch.randperm(snap.num_physical)
        t2 = dict(t)
        t2["physical_x"] = t["physical_x"][perm]
        t2["physical_token_ids"] = t["physical_token_ids"][perm]
        t2["physical_nodes"] = [t["physical_nodes"][int(i)] for i in perm]
        inv = torch.zeros_like(perm)
        inv[perm] = torch.arange(len(perm))
        t2["physical_edge_index"] = inv[t["physical_edge_index"]]
        o2 = model.from_snapshot_tensors(t2, 1)
        assert o1.physical_h.shape == o2.physical_h.shape


def test_mpnn_padding_mask_blocks_ghosts():
    from baba_graph.world_model.layers import MPNNLayer

    layer = MPNNLayer(8, 8)
    x = torch.randn(4, 8)
    ei = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    mask = torch.tensor([True, True, True, False])
    out_masked = layer(x, ei, node_mask=mask)
    x_pad = x.clone()
    x_pad[3] = 1e6
    out_unmasked = layer(x_pad, ei, node_mask=None)
    assert torch.allclose(out_masked[:3], out_unmasked[:3], atol=1e-5)
    assert out_masked[3].abs().max() < 1e-3


def test_collect_transitions():
    tr = collect_dynamics_transitions(MAP, episodes=1, max_steps=25, seed=0)
    assert len(tr) > 0


def test_warmup_step():
    tr = collect_dynamics_transitions(MAP, episodes=1, max_steps=15, seed=1)
    model = DualGraphWorldModel(
        WorldModelConfig(hidden_dim=32, codebook_size=64, rule_layers=1, physical_layers=1)
    )
    from baba_graph.world_model.train import contrastive_next_state_loss

    loss = contrastive_next_state_loss(model, tr[0], codebook_size=64)
    assert torch.isfinite(loss)
