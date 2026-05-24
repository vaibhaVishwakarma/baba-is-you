"""Extract object-centric graphs with open-vocabulary visual node features."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyBaba

from baba_graph.perception.compat import perception_to_graph_snapshot
from baba_graph.perception.extract import extract_perception_from_game, extract_perception_from_tensor
from baba_graph.types import GraphSnapshot, ObjectGraph
from baba_graph.vision.config import VisualEncoderConfig
from baba_graph.vision.encoder import FrozenVisualEncoder

if TYPE_CHECKING:
    from torch_geometric.data import Data


def extract_graph_from_game(
    game: pyBaba.Game,
    *,
    encoder: FrozenVisualEncoder | None = None,
    config: VisualEncoderConfig | None = None,
    map_name: str = "",
) -> GraphSnapshot:
    """Legacy unified graph; prefer extract_perception_from_game for V3."""
    del encoder  # V3 uses PerceptionEncoder; kept for API compat
    cfg = config or VisualEncoderConfig()
    snap = extract_perception_from_game(
        game, config=cfg, map_name=map_name, calibrate_classifier=True
    )
    return perception_to_graph_snapshot(snap)


def extract_graph_from_tensor(
    tensor: np.ndarray,
    *,
    encoder: FrozenVisualEncoder | None = None,
    config: VisualEncoderConfig | None = None,
    map_name: str = "",
    active_rules: tuple[str, ...] = (),
) -> GraphSnapshot:
    """Legacy unified graph from tensor."""
    del encoder
    cfg = config or VisualEncoderConfig()
    snap = extract_perception_from_tensor(
        tensor, config=cfg, map_name=map_name, active_rules=active_rules
    )
    return perception_to_graph_snapshot(snap)


def game_to_pyg_data(
    game: pyBaba.Game,
    *,
    encoder: FrozenVisualEncoder | None = None,
    config: VisualEncoderConfig | None = None,
    map_name: str = "",
) -> Data:
    snapshot = extract_graph_from_game(
        game, encoder=encoder, config=config, map_name=map_name
    )
    return graph_to_pyg_data(snapshot.graph, snapshot=snapshot)


def tensor_to_pyg_data(
    tensor: np.ndarray,
    *,
    encoder: FrozenVisualEncoder | None = None,
    config: VisualEncoderConfig | None = None,
    map_name: str = "",
    active_rules: tuple[str, ...] = (),
) -> Data:
    snapshot = extract_graph_from_tensor(
        tensor,
        encoder=encoder,
        config=config,
        map_name=map_name,
        active_rules=active_rules,
    )
    return graph_to_pyg_data(snapshot.graph, snapshot=snapshot)


def graph_to_pyg_data(
    graph: ObjectGraph,
    *,
    snapshot: GraphSnapshot | None = None,
) -> Data:
    import torch
    from torch_geometric.data import Data

    data = Data(
        x=torch.from_numpy(graph.x),
        edge_index=torch.from_numpy(graph.edge_index),
        pos=torch.tensor(graph.node_positions(), dtype=torch.long),
        num_nodes=graph.num_nodes,
    )
    data.grid_width = graph.grid_width
    data.grid_height = graph.grid_height
    data.map_name = graph.map_name
    data.visual_dim = graph.visual_dim
    if snapshot is not None:
        data.active_rules = list(snapshot.active_rules)
        if snapshot.play_state is not None:
            data.play_state = int(snapshot.play_state)
    return data


def batch_graphs(graphs: list[ObjectGraph]) -> Data:
    from torch_geometric.data import Batch

    return Batch.from_data_list([graph_to_pyg_data(g) for g in graphs])
