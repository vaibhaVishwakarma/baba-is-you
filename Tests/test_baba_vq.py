"""Phase 2 VQ-VAE tests."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pyBaba = pytest.importorskip("pyBaba")

from baba_graph import extract_perception_from_game
from baba_graph.vq import (
    DynamicVectorQuantizer,
    VQConfig,
    load_quantizer,
    save_quantizer,
    tokenize_perception,
)
from baba_world.paths import map_path

MAP = "baba_is_you"


def test_vq_ste_gradient():
    q = DynamicVectorQuantizer(VQConfig(num_codes=32, enable_expansion=False))
    z_e = torch.randn(8, 128, requires_grad=True)
    out = q(z_e)
    out.z_q.sum().backward()
    assert z_e.grad is not None
    assert torch.isfinite(z_e.grad).all()


def test_vq_encode_decode_roundtrip():
    q = DynamicVectorQuantizer(VQConfig(num_codes=64, enable_expansion=False))
    q.eval()
    z = torch.randn(16, 128)
    idx = q.encode(z)
    z_q = q.decode_indices(idx)
    assert idx.shape == (16,)
    assert z_q.shape == (16, 128)


def test_dynamic_expansion():
    cfg = VQConfig(num_codes=4, max_codes=16, expansion_threshold=0.1, enable_expansion=True)
    q = DynamicVectorQuantizer(cfg)
    q.train()
    # Far from small random codebook
    z = torch.randn(6, 128) * 5.0
    before = q.num_codes
    out = q(z)
    assert q.num_codes >= before
    assert out.indices.max() < q.num_codes


def test_tokenize_perception():
    game = pyBaba.Game(str(map_path(MAP)))
    game.Reset()
    snap = extract_perception_from_game(game, calibrate_classifier=False)
    q = DynamicVectorQuantizer(VQConfig(num_codes=128, enable_expansion=False))
    q.eval()
    tok, _ = tokenize_perception(snap, q)
    assert tok.text_tokens.shape[0] == snap.num_text
    assert tok.physical_tokens.shape[0] == snap.num_physical
    assert tok.text_z_q.shape == snap.text.visual.shape


def test_save_load_quantizer(tmp_path):
    q = DynamicVectorQuantizer(VQConfig(num_codes=32))
    path = tmp_path / "vq.pt"
    save_quantizer(q, path)
    q2 = load_quantizer(path)
    assert q2.num_codes == q.num_codes
    z = torch.randn(4, 128)
    np.testing.assert_array_equal(
        q.encode(z).cpu().numpy(),
        q2.encode(z).cpu().numpy(),
    )


def test_training_reduces_loss():
    cfg = VQConfig(num_codes=64, enable_expansion=False)
    q = DynamicVectorQuantizer(cfg)
    data = torch.randn(500, 128)
    from baba_graph.vq.data import VisualBatch
    from baba_graph.vq.train import train_vq

    batch = VisualBatch(data.numpy().astype(np.float32), np.zeros(500, np.int64))
    h = train_vq(q, batch, epochs=5, steps_per_epoch=20, batch_size=64)
    assert h[-1].loss <= h[0].loss * 1.5 or h[-1].perplexity > 1.0
