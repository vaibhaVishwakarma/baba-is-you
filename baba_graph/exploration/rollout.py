"""Shared perception rollout loop for VQ / world-model data."""

from __future__ import annotations

import random
from typing import Callable, Iterator, Literal

import numpy as np

from baba_graph.exploration.policy import BiasedTextExplorer
from baba_graph.perception.extract import extract_perception_from_game
from baba_graph.perception.types import PerceptionSnapshot
from baba_graph.vision.cache import VisualEmbeddingCache, get_visual_embedding_cache
from baba_graph.vision.config import VisualEncoderConfig
from baba_graph.vision.perception_encoder import get_perception_encoder
from baba_world.env import BabaTransitionEnv
from baba_world.paths import SUPPORTED_MAPS, map_path

PolicyName = Literal["biased", "random"]


def iter_perception_rollout(
    map_name: str,
    *,
    episodes: int = 1,
    max_steps: int = 100,
    seed: int = 0,
    policy: PolicyName = "biased",
    config: VisualEncoderConfig | None = None,
    visual_cache: VisualEmbeddingCache | None = None,
    calibrate_classifier: bool = False,
) -> Iterator[tuple[PerceptionSnapshot, int, bool, tuple[str, ...]]]:
    """
    Yield (snapshot, action, done, active_rules) per step.

    Uses patch-hash visual cache and biased text exploration by default.
    """
    path = map_path(map_name)
    if not path.is_file():
        return

    cfg = config or VisualEncoderConfig()
    cache = visual_cache or get_visual_embedding_cache()
    get_perception_encoder(cfg, force_new=False)

    rng = random.Random(seed)
    explorer = BiasedTextExplorer(rng) if policy == "biased" else None

    for ep in range(episodes):
        env = BabaTransitionEnv(map_name)
        env.reset(episode_id=ep)
        game = env._game
        assert game is not None

        if explorer:
            explorer.reset()

        snap = extract_perception_from_game(
            game,
            config=cfg,
            calibrate_classifier=calibrate_classifier,
        )
        rules = snap.active_rules or env._last_rules
        yield snap, -1, False, rules

        for _ in range(max_steps):
            if explorer:
                action = explorer.choose_action(game, snap)
            else:
                action = rng.randint(0, 3)

            rules_before = env._last_rules
            adj_before = None
            if explorer:
                from baba_graph.exploration.adjacency import text_adjacency_signature

                adj_before = text_adjacency_signature(snap)

            sample = env.step(action)
            snap = extract_perception_from_game(
                game,
                config=cfg,
                calibrate_classifier=calibrate_classifier,
            )
            rules = sample.rules_after

            if explorer:
                explorer.observe_snapshot(snap, rules=rules)
                from baba_graph.exploration.adjacency import (
                    adjacency_changed,
                    rules_changed,
                    text_adjacency_signature,
                )

                if adj_before is not None and adjacency_changed(
                    adj_before, text_adjacency_signature(snap)
                ):
                    explorer.register_successful_action(
                        action, rule_change=rules_changed(rules_before, rules)
                    )
                elif rules_changed(rules_before, rules):
                    explorer.register_successful_action(action, rule_change=True)

            yield snap, action, sample.done, rules
            if sample.done:
                break


def collect_snapshots(
    maps: list[str] | None = None,
    *,
    episodes_per_map: int = 50,
    max_steps: int = 100,
    seed: int = 0,
    policy: PolicyName = "biased",
    on_snapshot: Callable[[PerceptionSnapshot], None],
) -> dict[str, int]:
    """Run rollouts and invoke `on_snapshot` for each perception frame."""
    maps = list(maps or SUPPORTED_MAPS)
    counts: dict[str, int] = {}
    for map_name in maps:
        n = 0
        for snap, _, _, _ in iter_perception_rollout(
            map_name,
            episodes=episodes_per_map,
            max_steps=max_steps,
            seed=seed + hash(map_name) % 10_000,
            policy=policy,
        ):
            on_snapshot(snap)
            n += 1
        counts[map_name] = n
    return counts
