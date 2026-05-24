"""Collect visual vectors for VQ training."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from baba_graph.exploration.rollout import iter_perception_rollout
from baba_graph.vision.config import VisualEncoderConfig
from baba_graph.vision.perception_encoder import get_perception_encoder
from baba_world.paths import SUPPORTED_MAPS


@dataclass
class VisualBatch:
    """Flattened visual vectors from many snapshots."""

    vectors: np.ndarray  # (N, D)
    modality: np.ndarray  # (N,) 0=physical 1=text

    def __len__(self) -> int:
        return self.vectors.shape[0]

    def sample_torch(self, batch_size: int, device: str = "cpu"):
        import torch

        n = len(self)
        if n == 0:
            return torch.zeros(0, self.vectors.shape[1], device=device)
        idx = np.random.randint(0, n, size=min(batch_size, n))
        return torch.from_numpy(self.vectors[idx]).float().to(device)


def collect_visual_vectors(
    maps: list[str] | None = None,
    *,
    episodes_per_map: int = 50,
    max_steps: int = 100,
    seed: int = 0,
    config: VisualEncoderConfig | None = None,
    policy: str = "biased",
) -> VisualBatch:
    """
    Roll out with biased text exploration and patch-hash visual cache.

    Default policy targets text blocks and rule adjacency changes instead of
    uniform random walks.
    """
    maps = list(maps or SUPPORTED_MAPS)
    cfg = config or VisualEncoderConfig()
    get_perception_encoder(cfg, force_new=True)

    chunks: list[np.ndarray] = []
    mods: list[np.ndarray] = []

    for map_name in maps:
        for snap, _action, _done, _rules in iter_perception_rollout(
            map_name,
            episodes=episodes_per_map,
            max_steps=max_steps,
            seed=seed,
            policy="biased" if policy == "biased" else "random",
            config=cfg,
            calibrate_classifier=False,
        ):
            _append_group(snap.text.visual, 1, chunks, mods)
            _append_group(snap.physical.visual, 0, chunks, mods)

    if not chunks:
        d = cfg.embed_dim
        return VisualBatch(np.zeros((0, d), np.float32), np.zeros(0, np.int64))

    return VisualBatch(
        vectors=np.vstack(chunks).astype(np.float32),
        modality=np.concatenate(mods).astype(np.int64),
    )


def _append_group(
    visual: np.ndarray,
    mod: int,
    chunks: list[np.ndarray],
    mods: list[np.ndarray],
) -> None:
    if visual.size == 0:
        return
    chunks.append(visual)
    mods.append(np.full(visual.shape[0], mod, dtype=np.int64))
