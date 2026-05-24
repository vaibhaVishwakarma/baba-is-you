"""Device helpers: action + snapshot tensors on CUDA."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from baba_graph.device import action_tensor, snap_tensors_to_device


def test_snap_tensors_to_device_moves_cpu_tensors():
    snap = {
        "physical_x": torch.randn(3, 4),
        "text_nodes": [],
    }
    out = snap_tensors_to_device(snap, torch.device("cpu"))
    assert out["physical_x"].device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda not available")
def test_action_tensor_matches_module_cuda():
    emb = torch.nn.Embedding(4, 8).cuda()
    act = action_tensor(2, emb.weight)
    assert act.device.type == "cuda"
    assert emb(act).device.type == "cuda"
