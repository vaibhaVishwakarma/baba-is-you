"""Query YOU / WIN / … positions from perception snapshots (ontology-agnostic)."""

from __future__ import annotations

import pyBaba

from baba_graph.perception.types import PerceptionSnapshot, NodeGroup
from baba_graph.vocab import OBJECT_PROPERTY_TYPES


def _property_column(group: NodeGroup, prop: pyBaba.ObjectType) -> int:
    idx = OBJECT_PROPERTY_TYPES.index(prop)
    return group.visual_dim + 2 + idx


def node_indices_with_property(group: NodeGroup, prop: pyBaba.ObjectType) -> list[int]:
    col = _property_column(group, prop)
    if group.x.shape[1] <= col:
        return []
    active = group.x[:, col] > 0.5
    return [i for i, v in enumerate(active) if v]


def cell_positions_for_property(
    snap: PerceptionSnapshot,
    prop: pyBaba.ObjectType,
    *,
    modality: str = "physical",
) -> list[tuple[int, int]]:
    """Grid cells of nodes carrying `prop` in the collision graph."""
    group = snap.physical if modality == "physical" else snap.text
    return [
        (group.nodes[i].x, group.nodes[i].y)
        for i in node_indices_with_property(group, prop)
    ]


def you_positions(snap: PerceptionSnapshot) -> list[tuple[int, int]]:
    return cell_positions_for_property(snap, pyBaba.ObjectType.YOU)


def win_positions(snap: PerceptionSnapshot) -> list[tuple[int, int]]:
    return cell_positions_for_property(snap, pyBaba.ObjectType.WIN)
