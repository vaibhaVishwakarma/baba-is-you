"""Spatial dynamics labels for Phase 4 movement head."""

from __future__ import annotations

from enum import IntEnum

import numpy as np

from baba_graph.types import ObjectNode


class AdjacencyMove(IntEnum):
    STAY = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4


NUM_ADJACENCY_CLASSES = 5

# (dx, dy) in grid coordinates (x right, y down)
_DELTA_BY_CLASS: dict[int, tuple[int, int]] = {
    AdjacencyMove.STAY: (0, 0),
    AdjacencyMove.UP: (0, -1),
    AdjacencyMove.DOWN: (0, 1),
    AdjacencyMove.LEFT: (-1, 0),
    AdjacencyMove.RIGHT: (1, 0),
}


def delta_to_adjacency(dx: int, dy: int) -> int | None:
    """Map integer cell delta to adjacency class, or None if not cardinal/stay."""
    for cls, (cdx, cdy) in _DELTA_BY_CLASS.items():
        if dx == cdx and dy == cdy:
            return int(cls)
    return None


def movement_label(prev: ObjectNode, nxt: ObjectNode) -> int | None:
    return delta_to_adjacency(nxt.x - prev.x, nxt.y - prev.y)


def apply_adjacency(x: int, y: int, adj_class: int) -> tuple[int, int]:
    dx, dy = _DELTA_BY_CLASS[int(adj_class)]
    return x + dx, y + dy


def movement_labels_for_pairs(
    prev_nodes: list[ObjectNode],
    next_nodes: list[ObjectNode],
    pairs: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build movement targets for aligned pairs.

    Returns:
        labels: (P,) int64 adjacency class
        valid: (P,) bool — False where delta is not cardinal (skip in loss)
    """
    labels = np.zeros(len(pairs), dtype=np.int64)
    valid = np.zeros(len(pairs), dtype=bool)
    for k, (i, j) in enumerate(pairs):
        lab = movement_label(prev_nodes[i], next_nodes[j])
        if lab is not None:
            labels[k] = lab
            valid[k] = True
    return labels, valid
