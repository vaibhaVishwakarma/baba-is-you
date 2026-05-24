"""Cache frozen visual embeddings keyed by patch pixel hash."""

from __future__ import annotations

import hashlib
from collections import OrderedDict

import numpy as np

from baba_graph.vision.perception_encoder import PerceptionEncoder

_default_cache: VisualEmbeddingCache | None = None


def patch_hash(patch: np.ndarray) -> str:
    """
    Stable hash of patch pixels (3, H, W) in [0, 1].

    Quantizes to uint8 so hashing is deterministic across platforms.
    """
    arr = patch
    if arr.dtype != np.uint8:
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return hashlib.blake2b(arr.tobytes(), digest_size=16).hexdigest()


class VisualEmbeddingCache:
    """
    O(1) lookup for repeated object patches across timesteps.

    Baba sprites/text tiles are static; only positions change. After the first
    frame, encoding cost is ~0 for known patches.
    """

    def __init__(self, *, max_entries: int = 8192) -> None:
        self.max_entries = max_entries
        self._store: OrderedDict[str, np.ndarray] = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.encoder_calls = 0

    def clear(self) -> None:
        self._store.clear()
        self.hits = 0
        self.misses = 0
        self.encoder_calls = 0

    @property
    def size(self) -> int:
        return len(self._store)

    def encode_visual(
        self,
        encoder: PerceptionEncoder,
        patches: np.ndarray,
    ) -> np.ndarray:
        """
        Encode (N, 3, H, W) patches; run CNN only on cache misses.

        Returns (N, embed_dim) float32.
        """
        if patches.size == 0:
            return np.zeros((0, encoder.embed_dim), dtype=np.float32)

        n = patches.shape[0]
        dim = encoder.embed_dim
        out = np.zeros((n, dim), dtype=np.float32)
        # Unique misses per batch (duplicate patches share one forward pass)
        batch_miss: dict[str, list[int]] = {}

        for i in range(n):
            key = patch_hash(patches[i])
            hit = self._store.get(key)
            if hit is not None:
                out[i] = hit
                self.hits += 1
                self._store.move_to_end(key)
            else:
                batch_miss.setdefault(key, []).append(i)
                self.misses += 1

        if batch_miss:
            unique_patches = []
            unique_keys: list[str] = []
            for key, indices in batch_miss.items():
                unique_keys.append(key)
                unique_patches.append(patches[indices[0]])

            encoded = encoder.encode_visual(np.stack(unique_patches, axis=0))
            self.encoder_calls += len(unique_patches)
            for key, emb in zip(unique_keys, encoded):
                vec = emb.astype(np.float32)
                self._put(key, vec)
                for idx in batch_miss[key]:
                    out[idx] = vec

        return out

    def _put(self, key: str, embedding: np.ndarray) -> None:
        if key in self._store:
            self._store.move_to_end(key)
            self._store[key] = embedding
            return
        self._store[key] = embedding
        while len(self._store) > self.max_entries:
            self._store.popitem(last=False)


def get_visual_embedding_cache(*, force_new: bool = False) -> VisualEmbeddingCache:
    global _default_cache
    if force_new or _default_cache is None:
        _default_cache = VisualEmbeddingCache()
    return _default_cache
