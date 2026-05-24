"""Node enumeration (structure only; features built in features.py)."""

from __future__ import annotations

import pyBaba

from baba_graph.types import ObjectNode
from baba_graph.vocab import IGNORED_NODE_TYPES, type_name
from baba_world.rules import _cell_rule_string


def _rule_participation_mask(game: pyBaba.Game) -> set[tuple[int, int]]:
    m = game.GetMap()
    cells: set[tuple[int, int]] = set()
    for y in range(m.GetHeight()):
        for x in range(m.GetWidth()):
            for dx, dy in ((1, 0), (0, 1)):
                if _cell_rule_string(game, x, y, dx, dy) is not None:
                    cells.add((x, y))
                    cells.add((x + dx, y + dy))
                    cells.add((x + 2 * dx, y + 2 * dy))
    return cells


def enumerate_nodes(game: pyBaba.Game) -> list[ObjectNode]:
    """One node per (cell, object_type), excluding ICON_EMPTY."""
    m = game.GetMap()
    rule_cells = _rule_participation_mask(game)
    nodes: list[ObjectNode] = []
    node_id = 0

    for y in range(m.GetHeight()):
        for x in range(m.GetWidth()):
            for obj_type in m.At(x, y).GetTypes():
                if obj_type in IGNORED_NODE_TYPES:
                    continue
                nodes.append(
                    ObjectNode(
                        node_id=node_id,
                        x=x,
                        y=y,
                        object_type=obj_type,
                        type_name=type_name(obj_type),
                        is_text=bool(pyBaba.IsTextType(obj_type)),
                        is_icon=not bool(pyBaba.IsTextType(obj_type)),
                        in_rule=(x, y) in rule_cells,
                    )
                )
                node_id += 1

    return nodes
