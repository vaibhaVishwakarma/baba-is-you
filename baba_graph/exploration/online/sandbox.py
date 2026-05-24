"""Isolated sandbox rollouts targeting novel tokens."""

from __future__ import annotations

import random

from baba_graph.exploration.online.config import OnlineExplorationConfig
from baba_graph.exploration.rollout import iter_perception_rollout
from baba_graph.perception.extract import extract_perception_from_game
from baba_graph.vq.tokenize import tokenize_perception
from baba_graph.world_model.data import DynamicsTransition
from baba_world.env import BabaTransitionEnv
from baba_world.paths import map_path


def collect_sandbox_transitions(
    map_name: str,
    target_token_ids: set[int],
    *,
    config: OnlineExplorationConfig | None = None,
    seed: int = 0,
    quantizer=None,
) -> list[DynamicsTransition]:
    """
    Biased rollouts in a sandbox map; keep steps that touch `target_token_ids`.

    Simulates pushing/interacting after ontology change (e.g. new VQ entry).
    """
    cfg = config or OnlineExplorationConfig()
    if not map_path(map_name).is_file():
        return []

    rng = random.Random(seed)
    kept: list[DynamicsTransition] = []
    prev_snap = None
    prev_rules: tuple[str, ...] = ()

    for ep in range(cfg.sandbox_episodes):
        env = BabaTransitionEnv(map_name)
        env.reset(episode_id=ep)
        game = env._game
        assert game is not None

        prev_snap = extract_perception_from_game(game, calibrate_classifier=False)
        prev_rules = env._last_rules

        for _ in range(cfg.sandbox_max_steps):
            from baba_graph.exploration.policy import BiasedTextExplorer

            explorer = BiasedTextExplorer(rng)
            action = explorer.choose_action(game, prev_snap)
            sample = env.step(action)
            snap = extract_perception_from_game(game, calibrate_classifier=False)

            if quantizer is not None:
                tok, _ = tokenize_perception(snap, quantizer)
                phys_toks = set(tok.physical_tokens.tolist())
            else:
                from baba_graph.world_model.tokens import snapshot_token_ids

                _, arr = snapshot_token_ids(snap, codebook_size=512)
                phys_toks = set(arr.tolist())

            touches_new = bool(phys_toks & target_token_ids)
            if prev_snap is not None and touches_new:
                kept.append(
                    DynamicsTransition(
                        state=prev_snap,
                        action=action,
                        next_state=snap,
                        done=sample.done,
                        rules_before=prev_rules,
                        rules_after=sample.rules_after,
                    )
                )

            prev_snap = snap
            prev_rules = sample.rules_after
            if sample.done:
                break

    return kept


def seed_replay_from_rollout(
    map_name: str,
    buffer,
    *,
    episodes: int = 20,
    max_steps: int = 100,
    seed: int = 0,
) -> int:
    """Fill replay buffer with known-object transitions (pre-expansion data)."""
    from baba_graph.exploration.online.buffer import TransitionReplayBuffer

    assert isinstance(buffer, TransitionReplayBuffer)
    count = 0
    prev = None
    prev_rules = ()
    for snap, action, done, rules in iter_perception_rollout(
        map_name,
        episodes=episodes,
        max_steps=max_steps,
        seed=seed,
        policy="biased",
    ):
        if action < 0:
            prev = snap
            prev_rules = rules
            continue
        if prev is not None:
            buffer.push(
                DynamicsTransition(
                    state=prev,
                    action=action,
                    next_state=snap,
                    done=done,
                    rules_before=prev_rules,
                    rules_after=rules,
                )
            )
            count += 1
        prev = snap
        prev_rules = rules
    return count
