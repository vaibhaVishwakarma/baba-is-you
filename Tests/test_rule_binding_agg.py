"""Multi-property rule binding aggregation tests."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from baba_graph.world_model.binding import MultiPropertySlotAggregator


def test_scatter_max_composes_multiple_rules():
    agg = MultiPropertySlotAggregator(8, mode="max")
    default = torch.zeros(1, 8)
    fused = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    slot_ids = torch.tensor([3, 3, 3])
    out = agg(slot_ids, fused, codebook_size=8, default=default)
    # slot 3 must retain max per dim from all three — not last vector only
    assert out[3, 0] == pytest.approx(1.0)
    assert out[3, 1] == pytest.approx(2.0)
    assert out[3, 2] == pytest.approx(0.5)


def test_attention_differs_from_last_write():
    agg = MultiPropertySlotAggregator(16, mode="attention")
    default = torch.zeros(1, 16)
    a = torch.randn(16)
    b = torch.randn(16)
    c = torch.randn(16)
    fused = torch.stack([a, b, c])
    slot_ids = torch.tensor([1, 1, 1])
    out = agg(slot_ids, fused, codebook_size=4, default=default)
    last_only = b  # if overwrite
    assert not torch.allclose(out[1], last_only, atol=1e-4)


def test_distinct_slots_independent():
    agg = MultiPropertySlotAggregator(8, mode="max")
    default = torch.zeros(1, 8)
    fused = torch.ones(4, 8)
    slot_ids = torch.tensor([0, 0, 1, 1])
    out = agg(slot_ids, fused, codebook_size=4, default=default)
    assert out[0].abs().sum() > 0
    assert out[1].abs().sum() > 0
    assert out[2].abs().sum() == 0
