"""Graph data structures for object-centric states."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyBaba


@dataclass(frozen=True)
class ObjectNode:
    """Single entity on the grid (one object type at one cell)."""

    node_id: int
    x: int
    y: int
    object_type: pyBaba.ObjectType
    type_name: str
    is_text: bool
    is_icon: bool
    in_rule: bool


@dataclass
class ObjectGraph:
    """
    Object-centric graph for one timestep.

    x: (num_nodes, feature_dim) — [visual_embed | pos | properties]
    edge_index: (2, num_edges) COO format
    """

    nodes: list[ObjectNode]
    x: np.ndarray
    edge_index: np.ndarray
    grid_width: int
    grid_height: int
    feature_dim: int
    visual_dim: int
    map_name: str = ""

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def num_edges(self) -> int:
        return 0 if self.edge_index.size == 0 else self.edge_index.shape[1]

    def node_positions(self) -> np.ndarray:
        """(num_nodes, 2) integer grid coordinates."""
        return np.asarray([[n.x, n.y] for n in self.nodes], dtype=np.int64)


@dataclass
class GraphSnapshot:
    """Graph plus metadata (for transitions and validation)."""

    graph: ObjectGraph
    active_rules: tuple[str, ...] = ()
    play_state: pyBaba.PlayState | None = None
