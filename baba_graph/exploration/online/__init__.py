"""Phase 5: online exploration and few-shot token binding."""

from baba_graph.exploration.online.adapt import (
    AdaptationResult,
    build_mixed_batch,
    few_shot_adapt,
    on_new_vq_tokens,
    run_online_adaptation_loop,
)
from baba_graph.exploration.online.buffer import TransitionReplayBuffer
from baba_graph.exploration.online.config import OnlineExplorationConfig
from baba_graph.exploration.online.sandbox import collect_sandbox_transitions, seed_replay_from_rollout
from baba_graph.exploration.online.trigger import TokenExpansionTracker

__all__ = [
    "AdaptationResult",
    "OnlineExplorationConfig",
    "TokenExpansionTracker",
    "TransitionReplayBuffer",
    "build_mixed_batch",
    "collect_sandbox_transitions",
    "few_shot_adapt",
    "on_new_vq_tokens",
    "run_online_adaptation_loop",
    "seed_replay_from_rollout",
]
