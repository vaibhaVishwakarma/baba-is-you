"""Offline transition dataset writer and loader."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from baba_world.types import Episode, TransitionSample


@dataclass
class TransitionDataset:
    """In-memory transition dataset (fixed map shape)."""

    state: np.ndarray
    action: np.ndarray
    next_state: np.ndarray
    done: np.ndarray
    episode_id: np.ndarray
    step_id: np.ndarray
    map_name: str
    rules_before: list[tuple[str, ...]] | None = None
    rules_after: list[tuple[str, ...]] | None = None

    @property
    def num_transitions(self) -> int:
        return int(self.action.shape[0])

    @property
    def state_shape(self) -> tuple[int, ...]:
        return tuple(self.state.shape[1:])

    def __len__(self) -> int:
        return self.num_transitions


def samples_to_arrays(
    samples: list[TransitionSample],
    *,
    include_rules: bool = False,
) -> TransitionDataset:
    if not samples:
        raise ValueError("Cannot build dataset from empty sample list")

    map_name = "unknown"
    states = np.stack([s.state for s in samples], axis=0)
    next_states = np.stack([s.next_state for s in samples], axis=0)
    actions = np.asarray([s.action for s in samples], dtype=np.int8)
    dones = np.asarray([s.done for s in samples], dtype=np.bool_)
    episode_ids = np.asarray([s.episode_id for s in samples], dtype=np.int32)
    step_ids = np.asarray([s.step_id for s in samples], dtype=np.int32)

    rules_before = None
    rules_after = None
    if include_rules:
        rules_before = [s.rules_before for s in samples]
        rules_after = [s.rules_after for s in samples]

    return TransitionDataset(
        state=states,
        action=actions,
        next_state=next_states,
        done=dones,
        episode_id=episode_ids,
        step_id=step_ids,
        map_name=map_name,
        rules_before=rules_before,
        rules_after=rules_after,
    )


def episodes_to_dataset(
    episodes: list[Episode],
    *,
    include_rules: bool = True,
) -> TransitionDataset:
    if not episodes:
        raise ValueError("Cannot build dataset from empty episode list")

    samples = [t for ep in episodes for t in ep.transitions]
    dataset = samples_to_arrays(samples, include_rules=include_rules)
    return TransitionDataset(
        state=dataset.state,
        action=dataset.action,
        next_state=dataset.next_state,
        done=dataset.done,
        episode_id=dataset.episode_id,
        step_id=dataset.step_id,
        map_name=episodes[0].map_name,
        rules_before=dataset.rules_before,
        rules_after=dataset.rules_after,
    )


def save_dataset(dataset: TransitionDataset, path: str | Path) -> Path:
    """Save dataset as .npz plus metadata JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stem = path.with_suffix("") if path.suffix == ".npz" else path
    npz_path = stem.with_suffix(".npz")
    meta_path = stem.with_suffix(".meta.json")

    np.savez_compressed(
        npz_path,
        state=dataset.state,
        action=dataset.action,
        next_state=dataset.next_state,
        done=dataset.done,
        episode_id=dataset.episode_id,
        step_id=dataset.step_id,
    )

    meta: dict[str, Any] = {
        "map_name": dataset.map_name,
        "num_transitions": dataset.num_transitions,
        "state_shape": list(dataset.state_shape),
        "version": 1,
    }
    if dataset.rules_before is not None:
        meta["rules_before"] = [list(r) for r in dataset.rules_before]
        meta["rules_after"] = [list(r) for r in dataset.rules_after]

    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return npz_path


def load_dataset(path: str | Path) -> TransitionDataset:
    """Load dataset from .npz (+ optional .meta.json)."""
    path = Path(path)
    npz_path = path if path.suffix == ".npz" else path.with_suffix(".npz")
    meta_path = npz_path.with_suffix(".meta.json")

    with np.load(npz_path) as data:
        arrays = {key: data[key] for key in data.files}

    map_name = "unknown"
    rules_before = None
    rules_after = None
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        map_name = meta.get("map_name", map_name)
        if "rules_before" in meta:
            rules_before = [tuple(r) for r in meta["rules_before"]]
            rules_after = [tuple(r) for r in meta["rules_after"]]

    return TransitionDataset(
        state=arrays["state"],
        action=arrays["action"],
        next_state=arrays["next_state"],
        done=arrays["done"],
        episode_id=arrays["episode_id"],
        step_id=arrays["step_id"],
        map_name=map_name,
        rules_before=rules_before,
        rules_after=rules_after,
    )


class TransitionDatasetLoader:
    """PyTorch-style batch iterator (optional dependency)."""

    def __init__(self, dataset: TransitionDataset, batch_size: int = 128) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self._indices = np.arange(len(dataset))

    def __len__(self) -> int:
        n = len(self.dataset)
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        rng = np.random.default_rng()
        order = rng.permutation(self._indices)
        for start in range(0, len(order), self.batch_size):
            idx = order[start : start + self.batch_size]
            yield {
                "state": self.dataset.state[idx],
                "action": self.dataset.action[idx],
                "next_state": self.dataset.next_state[idx],
                "done": self.dataset.done[idx],
            }
