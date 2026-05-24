"""Edge construction: same-cell (overlap) + 4-neighbor spatial adjacency."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from baba_graph.types import ObjectNode

_NEIGHBORS_4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def build_edge_index(nodes: list[ObjectNode]) -> np.ndarray:
    """
    Build undirected edges as bidirectional COO pairs.

    - Overlap: all pairs of nodes sharing (x, y)
    - Adjacent: all pairs where cells are 4-neighbors
    """
    if len(nodes) <= 1:
        return np.zeros((2, 0), dtype=np.int64)

    by_cell: dict[tuple[int, int], list[int]] = defaultdict(list)
    for node in nodes:
        by_cell[(node.x, node.y)].append(node.node_id)

    edge_set: set[tuple[int, int]] = set()

    def add_edge(i: int, j: int) -> None:
        if i != j:
            edge_set.add((i, j))
            edge_set.add((j, i))

    for cell_nodes in by_cell.values():
        for i in range(len(cell_nodes)):
            for j in range(i + 1, len(cell_nodes)):
                add_edge(cell_nodes[i], cell_nodes[j])

    for (x, y), cell_nodes in by_cell.items():
        for dx, dy in _NEIGHBORS_4:
            neighbor = by_cell.get((x + dx, y + dy))
            if not neighbor:
                continue
            for i in cell_nodes:
                for j in neighbor:
                    add_edge(i, j)

    if not edge_set:
        return np.zeros((2, 0), dtype=np.int64)

    src, dst = zip(*sorted(edge_set))
    return np.asarray([src, dst], dtype=np.int64)
