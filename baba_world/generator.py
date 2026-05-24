"""Episode and transition generation."""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence

import numpy as np

from baba_world.actions import ACTION_COUNT
from baba_world.env import BabaTransitionEnv
from baba_world.types import Episode, TransitionSample


ActionPolicy = Callable[[np.ndarray, int], int]


def random_policy(_state: np.ndarray, _step: int, rng: random.Random) -> int:
    return rng.randint(0, ACTION_COUNT - 1)


def generate_episode(
    map_name: str,
    *,
    max_steps: int = 200,
    policy: ActionPolicy | None = None,
    seed: int | None = None,
    episode_id: int = 0,
) -> Episode:
    """Roll out one episode and return structured transitions."""
    rng = random.Random(seed)
    policy = policy or (lambda s, t: random_policy(s, t, rng))

    env = BabaTransitionEnv(map_name)
    state = env.reset(episode_id=episode_id)
    episode = Episode(map_name=env.map_name, episode_id=episode_id)

    for step in range(max_steps):
        action = policy(state, step)
        sample = env.step(action)
        episode.transitions.append(sample)
        state = sample.next_state
        if sample.done:
            break

    return episode


def generate_episodes(
    map_name: str,
    num_episodes: int,
    *,
    max_steps: int = 200,
    seed: int = 0,
) -> list[Episode]:
    """Generate multiple episodes with deterministic per-episode seeds."""
    episodes: list[Episode] = []
    for ep_id in range(num_episodes):
        episodes.append(
            generate_episode(
                map_name,
                max_steps=max_steps,
                seed=seed + ep_id,
                episode_id=ep_id,
            )
        )
    return episodes


def episodes_to_samples(episodes: Sequence[Episode]) -> list[TransitionSample]:
    """Flatten episodes into a list of transition samples."""
    return [t for ep in episodes for t in ep.transitions]
