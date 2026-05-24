"""Graph-level transitions paired with baba_world rollouts."""

from __future__ import annotations

from dataclasses import dataclass

from baba_graph.extractor import extract_graph_from_game
from baba_graph.types import GraphSnapshot
from baba_graph.vision.config import VisualEncoderConfig
from baba_graph.vision.encoder import FrozenVisualEncoder, get_visual_encoder
from baba_world.env import BabaTransitionEnv


@dataclass
class GraphTransition:
    state: GraphSnapshot
    action: int
    next_state: GraphSnapshot
    done: bool
    step_id: int
    episode_id: int


def rollout_graph_episode(
    map_name: str,
    *,
    max_steps: int = 200,
    seed: int | None = None,
    episode_id: int = 0,
    encoder: FrozenVisualEncoder | None = None,
    config: VisualEncoderConfig | None = None,
) -> list[GraphTransition]:
    import random

    from baba_world.actions import ACTION_COUNT

    rng = random.Random(seed)
    cfg = config or VisualEncoderConfig()
    enc = encoder or get_visual_encoder(cfg)
    env = BabaTransitionEnv(map_name)
    env.reset(episode_id=episode_id)
    game = env._game
    assert game is not None

    transitions: list[GraphTransition] = []
    for _ in range(max_steps):
        state_snap = extract_graph_from_game(
            game, encoder=enc, config=cfg, map_name=env.map_name
        )
        action = rng.randint(0, ACTION_COUNT - 1)
        sample = env.step(action)
        next_snap = extract_graph_from_game(
            game, encoder=enc, config=cfg, map_name=env.map_name
        )
        transitions.append(
            GraphTransition(
                state=state_snap,
                action=sample.action,
                next_state=next_snap,
                done=sample.done,
                step_id=sample.step_id,
                episode_id=sample.episode_id,
            )
        )
        if sample.done:
            break
    return transitions
