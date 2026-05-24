"""VQ token assignment for targeted rule binding."""

from __future__ import annotations

import numpy as np

from baba_graph.perception.types import PerceptionSnapshot
from baba_graph.types import ObjectNode
from baba_graph.vq.quantizer import DynamicVectorQuantizer
from baba_graph.vq.tokenize import tokenize_perception
from baba_graph.world_model.binding import canonical_noun


def stable_type_token(type_label: str, codebook_size: int) -> int:
    """Deterministic fallback slot when VQ checkpoint is unavailable."""
    return abs(hash(type_label)) % max(codebook_size, 1)


def token_ids_for_nodes(
    nodes: list[ObjectNode],
    *,
    codebook_size: int,
    explicit: np.ndarray | None = None,
) -> np.ndarray:
    """
    Per-node codebook index.

    Prefer VQ `explicit` indices; otherwise hash canonical noun / type name.
    """
    if explicit is not None and len(explicit) == len(nodes):
        return explicit.astype(np.int64)

    out = np.zeros(len(nodes), dtype=np.int64)
    for i, n in enumerate(nodes):
        noun = canonical_noun(n)
        label = noun if noun else n.type_name
        out[i] = stable_type_token(label, codebook_size)
    return out


def snapshot_token_ids(
    snap: PerceptionSnapshot,
    *,
    codebook_size: int,
    quantizer: DynamicVectorQuantizer | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (text_token_ids, physical_token_ids) aligned with node lists."""
    if quantizer is not None:
        tok, _ = tokenize_perception(snap, quantizer)
        return tok.text_tokens, tok.physical_tokens

    text_ids = token_ids_for_nodes(snap.text.nodes, codebook_size=codebook_size)
    phys_ids = token_ids_for_nodes(snap.physical.nodes, codebook_size=codebook_size)
    return text_ids, phys_ids
