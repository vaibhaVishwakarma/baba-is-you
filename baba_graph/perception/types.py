"""V3 perception split: separate text and physical node groups."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyBaba

from baba_graph.types import ObjectNode


@dataclass
class NodeGroup:
    """
    One modality (text or physical) at a single timestep.

    x layout: [visual_embed (D) | norm_x | norm_y | (physical only: properties)]
    """

    modality: str  # "text" | "physical"
    nodes: list[ObjectNode]
    x: np.ndarray
    edge_index: np.ndarray
    visual: np.ndarray  # (N, D) continuous vectors only
    feature_dim: int
    visual_dim: int

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def num_edges(self) -> int:
        return 0 if self.edge_index.size == 0 else self.edge_index.shape[1]

    def positions(self) -> np.ndarray:
        return np.asarray([[n.x, n.y] for n in self.nodes], dtype=np.int64)


@dataclass
class PerceptionSnapshot:
    """
    Full Phase 1 output: two explicit node lists + classifier diagnostics.

    Text blocks lead a double life:
      - `text`: rule-reading graph (directed L→R / T→B edges)
      - `physical`: collision graph (icons + mirrored text with PUSH, spatial edges)
    """

    text: NodeGroup
    physical: NodeGroup
    grid_width: int
    grid_height: int
    active_rules: tuple[str, ...] = ()
    play_state: pyBaba.PlayState | None = None
    map_name: str = ""
    # Classifier vs simulator ground-truth (is_text from engine)
    classifier_accuracy: float = 1.0
    classifier_preds: np.ndarray | None = None  # (total_nodes,) 0=physical 1=text

    @property
    def num_text(self) -> int:
        return self.text.num_nodes

    @property
    def num_physical(self) -> int:
        return self.physical.num_nodes

    @property
    def num_collision_nodes(self) -> int:
        """Icons + text blocks in the spatial interaction graph."""
        return self.physical.num_nodes

    @property
    def num_icon_nodes(self) -> int:
        return sum(1 for n in self.physical.nodes if not n.is_text)

    def combined_nodes(self) -> list[ObjectNode]:
        """All spatially interactive entities (physical graph is authoritative)."""
        return list(self.physical.nodes)

    def text_nodes_in_physical(self) -> list[ObjectNode]:
        return [n for n in self.physical.nodes if n.is_text]
