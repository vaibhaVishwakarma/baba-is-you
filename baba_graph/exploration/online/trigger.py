"""Detect novel VQ tokens and codebook expansion events."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from baba_graph.vq.quantizer import DynamicVectorQuantizer, VQOutput


@dataclass
class TokenExpansionTracker:
    """Tracks known codebook indices; reports newly seen token IDs."""

    known_tokens: set[int] = field(default_factory=set)

    def observe_tokens(self, token_ids: np.ndarray) -> list[int]:
        new_ids: list[int] = []
        for t in token_ids.reshape(-1).tolist():
            tid = int(t)
            if tid not in self.known_tokens:
                new_ids.append(tid)
                self.known_tokens.add(tid)
        return new_ids

    def observe_vq_output(
        self,
        quantizer: DynamicVectorQuantizer,
        output: VQOutput,
    ) -> list[int]:
        """Return token IDs added by a dynamic expansion step."""
        if output.expanded <= 0:
            return []
        k = quantizer.num_codes
        start = k - output.expanded
        new_ids = list(range(start, k))
        for tid in new_ids:
            self.known_tokens.add(tid)
        return new_ids


def transitions_with_tokens(
    transitions: list,
    token_ids: set[int],
) -> list:
    """Filter transitions where next state contains any of the token ids."""
    from baba_graph.world_model.tokens import snapshot_token_ids

    out = []
    for tr in transitions:
        _, phys = snapshot_token_ids(tr.next_state, codebook_size=max(token_ids) + 1)
        if any(int(t) in token_ids for t in phys):
            out.append(tr)
    return out
