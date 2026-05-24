"""Per-object property flags from active rules."""

from __future__ import annotations

import pyBaba

from baba_graph.vocab import OBJECT_PROPERTY_TYPES, type_name
from baba_world.rules import extract_active_rules


def _noun_for_object(obj_type: pyBaba.ObjectType) -> str | None:
    if pyBaba.IsTextType(obj_type) and pyBaba.IsNounType(obj_type):
        return type_name(obj_type)
    name = type_name(obj_type)
    if name.startswith("ICON_"):
        try:
            return type_name(pyBaba.ConvertIconToText(obj_type))
        except Exception:
            return None
    return None


def property_multi_hot(
    game: pyBaba.Game,
    obj_type: pyBaba.ObjectType,
) -> list[float]:
    return property_multi_hot_from_rules(extract_active_rules(game), obj_type)


def property_multi_hot_from_rules(
    active_rules: tuple[str, ...],
    obj_type: pyBaba.ObjectType,
) -> list[float]:
    noun = _noun_for_object(obj_type)
    lookup: dict[str, set[str]] = {}
    for rule in active_rules:
        parts = rule.split()
        if len(parts) == 3 and parts[1] == "IS":
            lookup.setdefault(parts[0], set()).add(parts[2])
    active = lookup.get(noun, set()) if noun else set()
    return [1.0 if type_name(p) in active else 0.0 for p in OBJECT_PROPERTY_TYPES]


def _property_index(prop: pyBaba.ObjectType) -> int:
    return OBJECT_PROPERTY_TYPES.index(prop)


def text_physics_properties(
    game: pyBaba.Game,
    obj_type: pyBaba.ObjectType,
    *,
    active_rules: tuple[str, ...] | None = None,
) -> list[float]:
    """
    Physics-facing properties for text blocks in the collision graph.

    Text is pushable by default in Baba Is You (unless STOP is active via rules).
    Rule-derived properties (YOU, WIN, etc.) are merged on top.
    """
    rules = active_rules if active_rules is not None else extract_active_rules(game)
    vec = property_multi_hot_from_rules(rules, obj_type)

    push_i = _property_index(pyBaba.ObjectType.PUSH)
    stop_i = _property_index(pyBaba.ObjectType.STOP)

    if vec[stop_i] < 0.5:
        vec[push_i] = 1.0
    return vec
