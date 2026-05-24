"""Dual-graph dynamics encoder with targeted symbolic rule binding."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from baba_graph.perception.types import PerceptionSnapshot
from baba_graph.device import action_tensor, infer_device_from_tensors
from baba_graph.world_model.binding import RuleBindingHead, gather_node_rule_context
from baba_graph.world_model.config import WorldModelConfig
from baba_graph.world_model.layers import MPNNLayer
from baba_graph.world_model.tokens import snapshot_token_ids


@dataclass
class WorldModelOutput:
    """Contextualized node embeddings after dynamics encoding."""

    text_h: torch.Tensor  # (N_text, H)
    physical_h: torch.Tensor  # (N_phys, H)
    rule_embeddings: torch.Tensor  # (codebook_size, H) token-indexed rule table
    node_rule_context: torch.Tensor  # (N_phys, H) per-node fetched rule context


class RuleGraphEncoder(nn.Module):
    """
    Directed MPNN on text + targeted binding into (codebook_size, H) rule table.

    No global soup: each VQ token slot receives only rules that govern that noun.
    """

    def __init__(self, config: WorldModelConfig) -> None:
        super().__init__()
        self.config = config
        h = config.hidden_dim
        self.in_proj = nn.Linear(config.embed_dim + 2, h)
        self.layers = nn.ModuleList(
            [MPNNLayer(h, h) for _ in range(config.rule_layers)]
        )
        self.binding = RuleBindingHead(
            h,
            config.codebook_size,
            aggregation=config.rule_aggregation,  # type: ignore[arg-type]
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        text_nodes: list,
        text_token_ids: torch.Tensor,
        physical_nodes: list,
        physical_token_ids: torch.Tensor,
        node_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if x.numel() == 0:
            k, h = self.config.codebook_size, self.config.hidden_dim
            device = edge_index.device if edge_index.numel() else x.device
            empty = torch.zeros(0, h, device=device)
            table = self.binding.default_slot.expand(k, -1).clone()
            return empty, table

        h = self.in_proj(x)
        for layer in self.layers:
            h = layer(h, edge_index, node_mask=node_mask)

        rule_embeddings = self.binding(
            h,
            text_nodes,
            text_token_ids,
            physical_nodes,
            physical_token_ids,
            edge_index,
        )
        return h, rule_embeddings


class PhysicalGraphMPNN(nn.Module):
    """
    Action-conditioned MPNN with per-node rule context from VQ token lookup.

    Each physical node concatenates only rule_embeddings[its_token_id].
    """

    def __init__(self, config: WorldModelConfig) -> None:
        super().__init__()
        self.config = config
        h = config.hidden_dim
        phys_in = config.embed_dim + 2 + 12
        self.action_emb = nn.Embedding(config.action_dim, h)
        self.in_proj = nn.Linear(phys_in + h + h, h)
        self.layers = nn.ModuleList(
            [MPNNLayer(h, h) for _ in range(config.physical_layers)]
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        action: torch.Tensor,
        rule_embeddings: torch.Tensor,
        physical_token_ids: torch.Tensor,
        node_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.numel() == 0:
            return torch.zeros(0, self.config.hidden_dim, device=rule_embeddings.device)

        node_rule = gather_node_rule_context(rule_embeddings, physical_token_ids)

        dev = x.device
        act = action_tensor(action, dev)
        a = self.action_emb(act.view(-1)[0])
        n = x.size(0)
        action_broadcast = a.unsqueeze(0).expand(n, -1)
        inp = torch.cat([x, action_broadcast, node_rule], dim=-1)
        h = self.in_proj(inp)
        h = self.dropout(h)
        for layer in self.layers:
            h = layer(h, edge_index, node_mask=node_mask)
        return h


class DualGraphWorldModel(nn.Module):
    """Phase 3 dynamics encoder with targeted symbolic binding."""

    def __init__(self, config: WorldModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or WorldModelConfig()
        self.rule_encoder = RuleGraphEncoder(self.config)
        self.physical_encoder = PhysicalGraphMPNN(self.config)

    def forward(
        self,
        text_x: torch.Tensor,
        text_edge_index: torch.Tensor,
        physical_x: torch.Tensor,
        physical_edge_index: torch.Tensor,
        action: torch.Tensor,
        text_token_ids: torch.Tensor,
        physical_token_ids: torch.Tensor,
        text_nodes: list,
        physical_nodes: list,
        *,
        text_node_mask: torch.Tensor | None = None,
        physical_node_mask: torch.Tensor | None = None,
    ) -> WorldModelOutput:
        action = action_tensor(
            action, infer_device_from_tensors(physical_x, text_x)
        )
        text_h, rule_embeddings = self.rule_encoder(
            text_x,
            text_edge_index,
            text_nodes,
            text_token_ids,
            physical_nodes,
            physical_token_ids,
            node_mask=text_node_mask,
        )
        node_rule = gather_node_rule_context(rule_embeddings, physical_token_ids)
        physical_h = self.physical_encoder(
            physical_x,
            physical_edge_index,
            action,
            rule_embeddings,
            physical_token_ids,
            node_mask=physical_node_mask,
        )
        return WorldModelOutput(
            text_h=text_h,
            physical_h=physical_h,
            rule_embeddings=rule_embeddings,
            node_rule_context=node_rule,
        )

    def from_snapshot_tensors(
        self,
        snap_tensors: dict[str, torch.Tensor | list],
        action: int | torch.Tensor,
    ) -> WorldModelOutput:
        dev = infer_device_from_tensors(
            snap_tensors["physical_x"],  # type: ignore[arg-type]
            snap_tensors["text_x"],  # type: ignore[arg-type]
        )
        action = action_tensor(action, dev)
        return self.forward(
            snap_tensors["text_x"],  # type: ignore[arg-type]
            snap_tensors["text_edge_index"],  # type: ignore[arg-type]
            snap_tensors["physical_x"],  # type: ignore[arg-type]
            snap_tensors["physical_edge_index"],  # type: ignore[arg-type]
            action,
            snap_tensors["text_token_ids"],  # type: ignore[arg-type]
            snap_tensors["physical_token_ids"],  # type: ignore[arg-type]
            snap_tensors["text_nodes"],  # type: ignore[arg-type]
            snap_tensors["physical_nodes"],  # type: ignore[arg-type]
            text_node_mask=snap_tensors.get("text_node_mask"),  # type: ignore[arg-type]
            physical_node_mask=snap_tensors.get("physical_node_mask"),  # type: ignore[arg-type]
        )


def snapshot_to_tensors(
    snap: PerceptionSnapshot,
    *,
    device: str = "cpu",
    codebook_size: int | None = None,
    quantizer=None,
    use_vq: bool = False,
    text_z_q=None,
    physical_z_q=None,
) -> dict[str, torch.Tensor | list]:
    """
    Build model inputs including VQ token ids for targeted rule binding.

    Pass `quantizer` for real VQ indices; otherwise stable type-hash slots.
    """
    cfg_embed = snap.text.visual_dim
    cb = codebook_size or (quantizer.num_codes if quantizer is not None else 512)

    text_tok, phys_tok = snapshot_token_ids(snap, codebook_size=cb, quantizer=quantizer)

    def _build_group(group, z_q_override, token_ids):
        n = group.num_nodes
        if n == 0:
            return (
                torch.zeros(0, cfg_embed + 2, device=device),
                torch.zeros(2, 0, dtype=torch.long, device=device),
                torch.zeros(0, dtype=torch.long, device=device),
                [],
                torch.zeros(0, dtype=torch.bool, device=device),
            )
        visual = z_q_override if z_q_override is not None else group.visual
        x_np = group.x.copy()
        x_np[:, :cfg_embed] = visual
        x = torch.from_numpy(x_np).float().to(device)
        ei = torch.from_numpy(group.edge_index).long().to(device)
        tids = torch.from_numpy(token_ids).long().to(device)
        mask = torch.ones(n, dtype=torch.bool, device=device)
        return x, ei, tids, list(group.nodes), mask

    text_x, text_ei, text_tids, text_nodes, text_mask = _build_group(
        snap.text, text_z_q if use_vq else None, text_tok
    )
    phys_x, phys_ei, phys_tids, phys_nodes, phys_mask = _build_group(
        snap.physical, physical_z_q if use_vq else None, phys_tok
    )

    return {
        "text_x": text_x,
        "text_edge_index": text_ei,
        "text_token_ids": text_tids,
        "text_nodes": text_nodes,
        "text_node_mask": text_mask,
        "physical_x": phys_x,
        "physical_edge_index": phys_ei,
        "physical_token_ids": phys_tids,
        "physical_nodes": phys_nodes,
        "physical_node_mask": phys_mask,
        "codebook_size": torch.tensor(cb, device=device),
    }
