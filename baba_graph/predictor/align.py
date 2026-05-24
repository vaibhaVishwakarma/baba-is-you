"""Bipartite node matching between timesteps (Hungarian algorithm)."""

from __future__ import annotations

import numpy as np

from baba_graph.types import ObjectNode

try:
    from scipy.optimize import linear_sum_assignment

    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


def _position_cost(
    prev_nodes: list[ObjectNode],
    next_nodes: list[ObjectNode],
    prev_tokens: np.ndarray,
    next_tokens: np.ndarray,
    *,
    token_mismatch_penalty: float = 1e4,
) -> np.ndarray:
    """Cost matrix (N_prev, N_next): L2 grid distance + token mismatch penalty."""
    n, m = len(prev_nodes), len(next_nodes)
    cost = np.zeros((n, m), dtype=np.float64)
    for i, pn in enumerate(prev_nodes):
        for j, nn in enumerate(next_nodes):
            dpos = np.hypot(pn.x - nn.x, pn.y - nn.y)
            penalty = 0.0 if int(prev_tokens[i]) == int(next_tokens[j]) else token_mismatch_penalty
            cost[i, j] = dpos + penalty
    return cost


def align_nodes_hungarian(
    prev_nodes: list[ObjectNode],
    prev_tokens: np.ndarray,
    next_nodes: list[ObjectNode],
    next_tokens: np.ndarray,
    *,
    max_match_cost: float = 50.0,
) -> list[tuple[int, int]]:
    """
    Optimal 1-to-1 matching via Hungarian algorithm on L2 position + token penalty.

    Rejects assignments whose cost exceeds `max_match_cost` (unmatched nodes).
    """
    if not prev_nodes or not next_nodes:
        return []

    if not _HAS_SCIPY:
        raise ImportError(
            "scipy is required for Hungarian alignment. Install with: pip install scipy"
        )

    cost = _position_cost(prev_nodes, next_nodes, prev_tokens, next_tokens)
    row_ind, col_ind = linear_sum_assignment(cost)

    pairs: list[tuple[int, int]] = []
    for r, c in zip(row_ind, col_ind):
        if cost[r, c] <= max_match_cost:
            pairs.append((int(r), int(c)))
    return pairs


# Canonical training alignment
align_nodes_by_token = align_nodes_hungarian
