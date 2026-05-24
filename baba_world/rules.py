"""Active rule extraction by scanning the map (mirrors C++ ParseRule)."""

from __future__ import annotations

import pyBaba


def _primary_type_name(obj: pyBaba.Object) -> str:
    types = obj.GetTypes()
    if not types:
        return "EMPTY"
    return str(types[0]).split(".")[-1]


def _cell_rule_string(game: pyBaba.Game, x: int, y: int, dx: int, dy: int) -> str | None:
    """Return 'NOUN IS PROPERTY' if a valid rule starts at (x, y), else None."""
    m = game.GetMap()
    width = m.GetWidth()
    height = m.GetHeight()
    x2, y2 = x + dx, y + dy
    x3, y3 = x + 2 * dx, y + 2 * dy

    if x3 >= width or y3 >= height:
        return None

    a = m.At(x, y)
    b = m.At(x2, y2)
    c = m.At(x3, y3)

    if not a.HasNounType() or not b.HasVerbType():
        return None
    if not (c.HasNounType() or c.HasPropertyType()):
        return None

    return f"{_primary_type_name(a)} {_primary_type_name(b)} {_primary_type_name(c)}"


def extract_active_rules(game: pyBaba.Game) -> tuple[str, ...]:
    """Return sorted unique active rules from the current map layout."""
    m = game.GetMap()
    width = m.GetWidth()
    height = m.GetHeight()
    seen: set[str] = set()
    rules: list[str] = []

    for y in range(height):
        for x in range(width):
            for dx, dy in ((1, 0), (0, 1)):
                text = _cell_rule_string(game, x, y, dx, dy)
                if text and text not in seen:
                    seen.add(text)
                    rules.append(text)

    return tuple(sorted(rules))


def rule_count_matches_manager(game: pyBaba.Game) -> bool:
    """True if scanned rules match RuleManager.GetNumRules()."""
    return len(extract_active_rules(game)) == game.GetRuleManager().GetNumRules()
