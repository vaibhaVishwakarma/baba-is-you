"""Hungarian alignment tests."""

from __future__ import annotations

import numpy as np
import pytest

scipy = pytest.importorskip("scipy")

from baba_graph.predictor.align import align_nodes_hungarian
from baba_graph.types import ObjectNode


def _node(nid: int, x: int, y: int, name: str = "ROCK") -> ObjectNode:
    import pyBaba

    ot = getattr(pyBaba.ObjectType, name, pyBaba.ObjectType.ROCK)
    return ObjectNode(
        node_id=nid,
        x=x,
        y=y,
        object_type=ot,
        type_name=name,
        is_text=False,
        is_icon=True,
        in_rule=False,
    )


def test_hungarian_one_to_one_with_two_rocks():
    prev = [_node(0, 0, 0), _node(1, 5, 5)]
    next_n = [_node(0, 0, 1), _node(1, 5, 6)]
    tok = np.array([7, 7])
    pairs = align_nodes_hungarian(prev, tok, next_n, tok)
    assert len(pairs) == 2
    used_next = {j for _, j in pairs}
    assert len(used_next) == 2


def test_hungarian_prefers_nearby_not_crossed():
    prev = [_node(0, 0, 0), _node(1, 10, 0)]
    next_n = [_node(0, 0, 1), _node(1, 10, 1)]
    tok = np.array([3, 3])
    pairs = align_nodes_hungarian(prev, tok, next_n, tok)
    mapping = dict(pairs)
    assert mapping[0] == 0
    assert mapping[1] == 1
