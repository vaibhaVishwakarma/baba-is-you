"""Fingerprints for text rule adjacency and active rules."""

from __future__ import annotations

from baba_graph.perception.types import PerceptionSnapshot


def text_adjacency_signature(snap: PerceptionSnapshot) -> frozenset[tuple[tuple[int, int], tuple[int, int]]]:
    """
    Undirected cell pairs linked by directed text rule edges.

    Used to detect when pushing text changes rule-reading topology.
    """
    if snap.text.num_edges == 0:
        return frozenset()

    by_id = {n.node_id: (n.x, n.y) for n in snap.text.nodes}
    pairs: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for src, dst in snap.text.edge_index.T:
        a = by_id[int(src)]
        b = by_id[int(dst)]
        pairs.add(tuple(sorted((a, b))))
    return frozenset(pairs)


def adjacency_changed(before: frozenset, after: frozenset) -> bool:
    return before != after


def rules_changed(before: tuple[str, ...], after: tuple[str, ...]) -> bool:
    return before != after
