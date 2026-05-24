"""Pure-PyTorch message passing with optional padding masks."""

from __future__ import annotations

import torch
import torch.nn as nn


def scatter_add_messages(
    messages: torch.Tensor,
    index: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """Aggregate incoming messages to each destination node."""
    dim = messages.size(-1)
    out = torch.zeros(num_nodes, dim, device=messages.device, dtype=messages.dtype)
    return out.index_add_(0, index, messages)


class MPNNLayer(nn.Module):
    """One round of message passing over edge_index (2, E) with optional node_mask."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.msg = nn.Linear(in_dim * 2, out_dim)
        self.self_lin = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        node_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if node_mask is not None:
            x = x * node_mask.unsqueeze(-1).to(x.dtype)

        if edge_index.numel() == 0:
            out = torch.relu(self.self_lin(x))
            if node_mask is not None:
                out = out * node_mask.unsqueeze(-1).to(out.dtype)
            return self.norm(out)

        src, dst = edge_index[0], edge_index[1]
        if node_mask is not None:
            valid = node_mask[src] & node_mask[dst]
            if not valid.any():
                out = torch.relu(self.self_lin(x))
                out = out * node_mask.unsqueeze(-1).to(out.dtype)
                return self.norm(out)
            src, dst = src[valid], dst[valid]

        pair = torch.cat([x[src], x[dst]], dim=-1)
        messages = self.msg(pair)
        agg = scatter_add_messages(messages, dst, x.size(0))
        out = torch.relu(self.self_lin(x) + agg)
        if node_mask is not None:
            out = out * node_mask.unsqueeze(-1).to(out.dtype)
        return self.norm(out)
