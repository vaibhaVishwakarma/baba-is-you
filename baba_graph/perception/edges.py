"""Modality-specific edge construction."""

from __future__ import annotations

import numpy as np

from baba_graph.edges import build_edge_index
from baba_graph.types import ObjectNode

# Directed reading order for rule chains (Phase 3 Rule Graph).
_RIGHT = (1, 0)   # left → right
_DOWN = (0, 1)   # top → bottom


def build_text_rule_edges(nodes: list[ObjectNode]) -> np.ndarray:
    """
    Directed text adjacency for rule parsing only.

    Edges follow reading order:
      - horizontal: (x, y) → (x+1, y)
      - vertical:   (x, y) → (x, y+1)

    No reverse edges, so YOU ← IS ← BABA is not implied from BABA → IS → YOU.
    """
    if len(nodes) <= 1:
        return np.zeros((2, 0), dtype=np.int64)

    index_by_cell = {(n.x, n.y): n.node_id for n in nodes}
    edges: list[tuple[int, int]] = []

    for n in nodes:
        right = index_by_cell.get((n.x + _RIGHT[0], n.y + _RIGHT[1]))
        if right is not None:
            edges.append((n.node_id, right))
        down = index_by_cell.get((n.x + _DOWN[0], n.y + _DOWN[1]))
        if down is not None:
            edges.append((n.node_id, down))

    if not edges:
        return np.zeros((2, 0), dtype=np.int64)
    src, dst = zip(*edges)
    return np.asarray([src, dst], dtype=np.int64)


def build_physical_edges(nodes: list[ObjectNode]) -> np.ndarray:
    """
    Spatial collision graph: overlap (same cell) + undirected 4-neighbor adjacency.

    Includes icons and text blocks (text leads a double life in the physical graph).
    """
    return build_edge_index(nodes)


def is_directed_rule_edge(edge_index: np.ndarray, nodes: list[ObjectNode]) -> bool:
    """True if every edge moves rightward or downward (never left/up)."""
    if edge_index.size == 0:
        return True
    by_id = {n.node_id: n for n in nodes}
    for src, dst in edge_index.T:
        a, b = by_id[int(src)], by_id[int(dst)]
        forward = (b.x > a.x and b.y == a.y) or (b.y > a.y and b.x == a.x)
        if not forward:
            return False
    return True
