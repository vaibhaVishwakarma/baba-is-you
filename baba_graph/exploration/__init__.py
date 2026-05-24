"""Biased exploration for ontology-rich rollouts."""

from baba_graph.exploration.adjacency import (
    adjacency_changed,
    rules_changed,
    text_adjacency_signature,
)
from baba_graph.exploration.policy import BiasedTextExplorer, ExplorerStats

try:
    from baba_graph.exploration.online import (
        OnlineExplorationConfig,
        TokenExpansionTracker,
        TransitionReplayBuffer,
        on_new_vq_tokens,
    )
    _ONLINE = [
        "OnlineExplorationConfig",
        "TokenExpansionTracker",
        "TransitionReplayBuffer",
        "on_new_vq_tokens",
    ]
except ImportError:
    _ONLINE = []

__all__ = [
    "BiasedTextExplorer",
    "ExplorerStats",
    "adjacency_changed",
    "rules_changed",
    "text_adjacency_signature",
    *_ONLINE,
]
