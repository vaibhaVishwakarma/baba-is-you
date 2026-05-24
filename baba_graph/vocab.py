"""Shared constants for graph extraction (no categorical node IDs in Phase 1)."""

from __future__ import annotations

import pyBaba

# Properties that can appear in active rules (multi-hot tail of node features).
OBJECT_PROPERTY_TYPES: tuple[pyBaba.ObjectType, ...] = (
    pyBaba.ObjectType.YOU,
    pyBaba.ObjectType.STOP,
    pyBaba.ObjectType.PUSH,
    pyBaba.ObjectType.PULL,
    pyBaba.ObjectType.WIN,
    pyBaba.ObjectType.DEFEAT,
    pyBaba.ObjectType.SINK,
    pyBaba.ObjectType.HOT,
    pyBaba.ObjectType.MELT,
    pyBaba.ObjectType.OPEN,
    pyBaba.ObjectType.SHUT,
    pyBaba.ObjectType.MOVE,
)

# Types we never emit as nodes.
IGNORED_NODE_TYPES = frozenset({pyBaba.ObjectType.ICON_EMPTY})


def type_name(obj_type: pyBaba.ObjectType) -> str:
    return str(obj_type).split(".")[-1]


# Back-compat alias
_type_name = type_name
