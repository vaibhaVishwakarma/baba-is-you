"""V3 perception split tests."""

from __future__ import annotations

import numpy as np
import pytest

pyBaba = pytest.importorskip("pyBaba")
pytest.importorskip("torch")

from baba_graph import extract_perception_from_game, run_graph_validations
from baba_graph.perception.edges import is_directed_rule_edge
from baba_graph.properties import _property_index
from baba_world.paths import map_path

MAP = "baba_is_you"


def test_text_rule_graph_only_text():
    snap = extract_perception_from_game(pyBaba.Game(str(map_path(MAP))))
    assert snap.num_text > 0
    assert all(n.is_text for n in snap.text.nodes)


def test_text_mirrored_into_physical():
    snap = extract_perception_from_game(pyBaba.Game(str(map_path(MAP))))
    assert len(snap.text_nodes_in_physical()) == snap.num_text
    assert snap.num_physical == snap.num_text + snap.num_icon_nodes
    assert all(n.is_text for n in snap.text_nodes_in_physical())


def test_text_has_push_in_physical_graph():
    snap = extract_perception_from_game(pyBaba.Game(str(map_path(MAP))))
    push_i = _property_index(pyBaba.ObjectType.PUSH)
    stop_i = _property_index(pyBaba.ObjectType.STOP)
    offset = snap.physical.visual_dim + 2
    for i, node in enumerate(snap.physical.nodes):
        if not node.is_text:
            continue
        if snap.physical.x[i, offset + stop_i] > 0.5:
            continue  # e.g. WALL IS STOP
        assert snap.physical.x[i, offset + push_i] > 0.5


def test_text_edges_are_directed():
    snap = extract_perception_from_game(pyBaba.Game(str(map_path(MAP))))
    assert is_directed_rule_edge(snap.text.edge_index, snap.text.nodes)
    if snap.text.num_edges > 0:
        assert snap.text.num_edges <= snap.num_text * 2


def test_directed_edges_not_symmetric():
    snap = extract_perception_from_game(pyBaba.Game(str(map_path(MAP))))
    ei = snap.text.edge_index
    if ei.shape[1] == 0:
        pytest.skip("no text edges on this map")
    forward = {tuple(e) for e in ei.T}
    backward = {(int(d), int(s)) for s, d in ei.T}
    assert forward.isdisjoint(backward), "text edges must not be bidirectional"


def test_separate_visual_tensors():
    snap = extract_perception_from_game(pyBaba.Game(str(map_path(MAP))))
    assert snap.text.visual.shape[0] == snap.num_text
    assert snap.physical.visual.shape[0] == snap.num_physical


def test_classifier_accuracy():
    game = pyBaba.Game(str(map_path(MAP)))
    game.Reset()
    snap = extract_perception_from_game(game, calibrate_classifier=True)
    assert snap.classifier_accuracy >= 0.85


def test_v3_validation():
    report = run_graph_validations(MAP)
    assert report.passed, report.summary()
