"""Visual embedding cache tests."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pyBaba = pytest.importorskip("pyBaba")

from baba_graph import extract_perception_from_game
from baba_graph.vision.cache import VisualEmbeddingCache, get_visual_embedding_cache, patch_hash
from baba_graph.vision.config import VisualEncoderConfig
from baba_graph.vision.patches import PatchRenderer
from baba_graph.vision.perception_encoder import get_perception_encoder
from baba_world.paths import map_path

MAP = "baba_is_you"


def test_patch_hash_stable():
    r = PatchRenderer(patch_size=32)
    p = r.render("ROCK", is_text=True)
    assert patch_hash(p) == patch_hash(p.copy())


def test_cache_avoids_reencode():
    cfg = VisualEncoderConfig()
    enc = get_perception_encoder(cfg, force_new=True)
    cache = VisualEmbeddingCache()
    r = PatchRenderer(patch_size=cfg.patch_size)
    patches = np.stack(
        [r.render("BABA", is_text=True), r.render("IS", is_text=True)] * 5,
        axis=0,
    )
    cache.encode_visual(enc, patches)
    assert cache.encoder_calls == 2
    cache.encode_visual(enc, patches)
    assert cache.encoder_calls == 2
    assert cache.hits == 10


def test_extract_uses_cache_across_steps():
    cache = VisualEmbeddingCache()
    get_visual_embedding_cache(force_new=True)
    game = pyBaba.Game(str(map_path(MAP)))
    game.Reset()
    snap1 = extract_perception_from_game(game, calibrate_classifier=False)
    snap2 = extract_perception_from_game(game, calibrate_classifier=False)
    shared = get_visual_embedding_cache()
    assert shared.hits > 0 or shared.encoder_calls < snap1.num_physical + snap2.num_physical
