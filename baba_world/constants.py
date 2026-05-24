"""Tensor channel semantics (mirrors C++ Preprocess::TENSOR_DIM_MAP)."""

from __future__ import annotations

import pyBaba

# Channel index -> ObjectType (C++ TENSOR_DIM_MAP in Preprocess.cpp).
TENSOR_DIM_MAP: dict[pyBaba.ObjectType, int] = {
    pyBaba.ObjectType.BABA: 0,
    pyBaba.ObjectType.IS: 1,
    pyBaba.ObjectType.YOU: 2,
    pyBaba.ObjectType.ICON_EMPTY: 3,
    pyBaba.ObjectType.FLAG: 4,
    pyBaba.ObjectType.WIN: 5,
    pyBaba.ObjectType.ICON_WALL: 6,
    pyBaba.ObjectType.ICON_ROCK: 7,
    pyBaba.ObjectType.ICON_BABA: 8,
    pyBaba.ObjectType.ICON_FLAG: 9,
    pyBaba.ObjectType.WALL: 10,
    pyBaba.ObjectType.STOP: 11,
    pyBaba.ObjectType.ROCK: 12,
    pyBaba.ObjectType.PUSH: 13,
}

# Channel index -> ObjectType encoded in that channel (inverse of TENSOR_DIM_MAP).
TENSOR_CHANNEL_TYPES: dict[int, pyBaba.ObjectType] = {
    0: pyBaba.ObjectType.BABA,
    1: pyBaba.ObjectType.IS,
    2: pyBaba.ObjectType.YOU,
    3: pyBaba.ObjectType.ICON_EMPTY,
    4: pyBaba.ObjectType.FLAG,
    5: pyBaba.ObjectType.WIN,
    6: pyBaba.ObjectType.ICON_WALL,
    7: pyBaba.ObjectType.ICON_ROCK,
    8: pyBaba.ObjectType.ICON_BABA,
    9: pyBaba.ObjectType.ICON_FLAG,
    10: pyBaba.ObjectType.WALL,
    11: pyBaba.ObjectType.STOP,
    12: pyBaba.ObjectType.ROCK,
    13: pyBaba.ObjectType.PUSH,
}

CHANNEL_HAS_TEXT = 14
CHANNEL_IS_RULE = 15

TENSOR_DIM = pyBaba.Preprocess.TENSOR_DIM

# Object types present on map cells but not assigned a dedicated tensor channel.
UNCODED_OBJECT_TYPES = frozenset(
    {
        pyBaba.ObjectType.ICON_TILE,
        pyBaba.ObjectType.ICON_WATER,
        pyBaba.ObjectType.ICON_GRASS,
        pyBaba.ObjectType.ICON_LAVA,
        pyBaba.ObjectType.ICON_SKULL,
        pyBaba.ObjectType.ICON_FLOWER,
        pyBaba.ObjectType.WATER,
        pyBaba.ObjectType.SINK,
        pyBaba.ObjectType.LAVA,
        pyBaba.ObjectType.MELT,
        pyBaba.ObjectType.HOT,
        pyBaba.ObjectType.SKULL,
        pyBaba.ObjectType.DEFEAT,
    }
)
