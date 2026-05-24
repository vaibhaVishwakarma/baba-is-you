"""Bridge V3 PerceptionSnapshot to legacy ObjectGraph."""

from __future__ import annotations

import numpy as np

from baba_graph.edges import build_edge_index
from baba_graph.perception.types import PerceptionSnapshot
from baba_graph.types import ObjectGraph, GraphSnapshot


def perception_to_object_graph(snap: PerceptionSnapshot) -> ObjectGraph:
    """Merge text + physical groups into one graph (backward compatibility)."""
    nodes = snap.combined_nodes()
    if not nodes:
        x = np.zeros((0, 0), dtype=np.float32)
        ei = np.zeros((2, 0), dtype=np.int64)
    else:
        full_dim = snap.physical.feature_dim
        text_x = snap.text.x
        if text_x.shape[1] < full_dim:
            pad = full_dim - text_x.shape[1]
            text_x = np.pad(text_x, ((0, 0), (0, pad)))
        x = np.vstack([text_x, snap.physical.x])
        ei = build_edge_index(nodes)

    return ObjectGraph(
        nodes=nodes,
        x=x,
        edge_index=ei,
        grid_width=snap.grid_width,
        grid_height=snap.grid_height,
        feature_dim=x.shape[1] if x.size else 0,
        visual_dim=snap.text.visual_dim,
        map_name=snap.map_name,
    )


def perception_to_graph_snapshot(snap: PerceptionSnapshot) -> GraphSnapshot:
    return GraphSnapshot(
        graph=perception_to_object_graph(snap),
        active_rules=snap.active_rules,
        play_state=snap.play_state,
    )
