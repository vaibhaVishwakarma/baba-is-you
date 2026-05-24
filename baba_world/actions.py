"""Fixed action encoding (frozen before training)."""

from __future__ import annotations

import pyBaba

# Integer action id -> simulator direction (do not change after dataset creation).
ACTION_TO_DIRECTION: dict[int, pyBaba.Direction] = {
    0: pyBaba.Direction.UP,
    1: pyBaba.Direction.DOWN,
    2: pyBaba.Direction.LEFT,
    3: pyBaba.Direction.RIGHT,
}

DIRECTION_TO_ACTION: dict[pyBaba.Direction, int] = {
    direction: action for action, direction in ACTION_TO_DIRECTION.items()
}

ACTION_COUNT = len(ACTION_TO_DIRECTION)
ACTION_NAMES = ("UP", "DOWN", "LEFT", "RIGHT")


def encode_action(direction: pyBaba.Direction) -> int:
    """Map simulator direction to integer action id."""
    try:
        return DIRECTION_TO_ACTION[direction]
    except KeyError as exc:
        raise ValueError(f"Direction {direction} is not a player move") from exc


def decode_action(action: int) -> pyBaba.Direction:
    """Map integer action id to simulator direction."""
    try:
        return ACTION_TO_DIRECTION[int(action)]
    except (KeyError, ValueError) as exc:
        raise ValueError(
            f"Action must be in {{0, 1, 2, 3}}, got {action!r}"
        ) from exc
