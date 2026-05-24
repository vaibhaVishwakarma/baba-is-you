"""Tokenized transition dataset for Phase 4."""

from __future__ import annotations

from baba_graph.world_model.data import DynamicsTransition, collect_dynamics_transitions
from baba_world.paths import SUPPORTED_MAPS


def collect_tokenized_transitions(
    map_name: str,
    *,
    episodes: int = 30,
    max_steps: int = 150,
    seed: int = 0,
) -> list[DynamicsTransition]:
    """Biased rollouts packaged as DynamicsTransition list."""
    return collect_dynamics_transitions(
        map_name,
        episodes=episodes,
        max_steps=max_steps,
        seed=seed,
    )


def collect_multi_map(
    maps: list[str] | None = None,
    **kwargs,
) -> list[DynamicsTransition]:
    maps = list(maps or SUPPORTED_MAPS)
    out: list[DynamicsTransition] = []
    for i, name in enumerate(maps):
        out.extend(
            collect_tokenized_transitions(name, seed=kwargs.get("seed", 0) + i * 997, **kwargs)
        )
    return out
