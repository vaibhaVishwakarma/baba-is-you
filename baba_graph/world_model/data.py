"""Collect (s, a, s') transitions for world-model training."""

from __future__ import annotations

from dataclasses import dataclass

from baba_graph.exploration.rollout import iter_perception_rollout
from baba_graph.perception.types import PerceptionSnapshot
from baba_graph.vq.quantizer import DynamicVectorQuantizer
from baba_graph.vq.tokenize import tokenize_perception
from baba_world.paths import SUPPORTED_MAPS


@dataclass
class DynamicsTransition:
    """One step of perception dynamics (pre-Phase-4 token prediction)."""

    state: PerceptionSnapshot
    action: int
    next_state: PerceptionSnapshot
    done: bool
    rules_before: tuple[str, ...]
    rules_after: tuple[str, ...]


def collect_dynamics_transitions(
    map_name: str,
    *,
    episodes: int = 20,
    max_steps: int = 150,
    seed: int = 0,
    quantizer: DynamicVectorQuantizer | None = None,
) -> list[DynamicsTransition]:
    """
    Biased rollouts with rule/adjacency-rich exploration.

    Optionally attach VQ tokenization later in the training loop via quantizer.
    """
    transitions: list[DynamicsTransition] = []
    prev_snap: PerceptionSnapshot | None = None
    prev_rules: tuple[str, ...] = ()

    for snap, action, done, rules in iter_perception_rollout(
        map_name,
        episodes=episodes,
        max_steps=max_steps,
        seed=seed,
        policy="biased",
        calibrate_classifier=False,
    ):
        if action < 0:
            prev_snap = snap
            prev_rules = rules
            continue
        if prev_snap is not None:
            transitions.append(
                DynamicsTransition(
                    state=prev_snap,
                    action=action,
                    next_state=snap,
                    done=done,
                    rules_before=prev_rules,
                    rules_after=rules,
                )
            )
        prev_snap = snap
        prev_rules = rules
        if done:
            prev_snap = None

    if quantizer is not None:
        # Touch tokenization path so callers can verify VQ integration
        for tr in transitions[:1]:
            tokenize_perception(tr.state, quantizer)

    return transitions


def collect_multi_map_transitions(
    maps: list[str] | None = None,
    *,
    seed: int = 0,
    **kwargs,
) -> list[DynamicsTransition]:
    maps = list(maps or SUPPORTED_MAPS)
    out: list[DynamicsTransition] = []
    for i, name in enumerate(maps):
        out.extend(
            collect_dynamics_transitions(name, seed=seed + i * 997, **kwargs)
        )
    return out
