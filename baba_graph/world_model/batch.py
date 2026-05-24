"""Padded graph batches with masks (avoids ghost-node message bleed)."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class PaddedGraphBatch:
    """
    Fixed-size batch of variable-size graphs.

    Real nodes are True in `node_mask`; padding nodes must not send/receive messages.
    """

    x: torch.Tensor  # (B, N_max, F)
    edge_index: torch.Tensor  # (B, 2, E_max) with -1 padding on edges
    node_mask: torch.Tensor  # (B, N_max) bool
    edge_mask: torch.Tensor  # (B, E_max) bool
    batch_size: int
    max_nodes: int

    @staticmethod
    def from_graph_list(
        graphs: list[dict[str, torch.Tensor]],
        *,
        max_nodes: int | None = None,
    ) -> PaddedGraphBatch:
        if not graphs:
            raise ValueError("empty graph list")

        b = len(graphs)
        max_nodes = max_nodes or max(g["x"].size(0) for g in graphs)
        feat = graphs[0]["x"].size(-1)
        device = graphs[0]["x"].device

        x = torch.zeros(b, max_nodes, feat, device=device)
        node_mask = torch.zeros(b, max_nodes, dtype=torch.bool, device=device)

        edge_chunks: list[torch.Tensor] = []
        max_e = 0
        for bi, g in enumerate(graphs):
            n = g["x"].size(0)
            x[bi, :n] = g["x"]
            node_mask[bi, :n] = True
            ei = g["edge_index"]
            if ei.numel():
                max_e = max(max_e, ei.size(1))
                edge_chunks.append((bi, ei))

        if max_e == 0:
            edge_index = torch.full((b, 2, 1), -1, dtype=torch.long, device=device)
            edge_mask = torch.zeros(b, 1, dtype=torch.bool, device=device)
        else:
            edge_index = torch.full((b, 2, max_e), -1, dtype=torch.long, device=device)
            edge_mask = torch.zeros(b, max_e, dtype=torch.bool, device=device)
            for bi, ei in edge_chunks:
                e = ei.size(1)
                edge_index[bi, :, :e] = ei
                edge_index[bi, 0, :e] += bi * max_nodes  # batched offset for disjoint union style
                edge_mask[bi, :e] = True

        return PaddedGraphBatch(
            x=x,
            edge_index=edge_index,
            node_mask=node_mask,
            edge_mask=edge_mask,
            batch_size=b,
            max_nodes=max_nodes,
        )
