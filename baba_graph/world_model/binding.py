"""Targeted symbolic binding: rule vectors keyed by VQ token, not global soup."""

from __future__ import annotations

from typing import Literal

import pyBaba
import torch
import torch.nn as nn
import torch.nn.functional as F

from baba_graph.types import ObjectNode
from baba_graph.vocab import type_name


def canonical_noun(node: ObjectNode) -> str | None:
    """Noun label shared by text ROCK and icon ROCK."""
    if pyBaba.IsTextType(node.object_type) and pyBaba.IsNounType(node.object_type):
        return node.type_name
    if node.type_name.startswith("ICON_"):
        try:
            return type_name(pyBaba.ConvertIconToText(node.object_type))
        except Exception:
            return None
    return None


def parse_rule_chains(
    nodes: list[ObjectNode],
    edge_index: torch.Tensor,
) -> list[tuple[int, int]]:
    """
    Parse directed text graph into (subject_idx, property_idx) chains.

    Pattern: NOUN -[edge]-> IS -[edge]-> PROPERTY (horizontal or vertical).
    """
    if not nodes or edge_index.numel() == 0:
        return []

    is_indices = {i for i, n in enumerate(nodes) if n.type_name == "IS"}

    pred: dict[int, list[int]] = {i: [] for i in range(len(nodes))}
    succ: dict[int, list[int]] = {i: [] for i in range(len(nodes))}
    for src, dst in edge_index.t().tolist():
        s, d = int(src), int(dst)
        if s < len(nodes) and d < len(nodes):
            succ[s].append(d)
            pred[d].append(s)

    chains: list[tuple[int, int]] = []
    for i_is in is_indices:
        for i_subj in pred.get(i_is, []):
            for i_prop in succ.get(i_is, []):
                n_subj = nodes[i_subj]
                n_prop = nodes[i_prop]
                if canonical_noun(n_subj) is None:
                    continue
                if not pyBaba.IsPropertyType(n_prop.object_type):
                    continue
                chains.append((i_subj, i_prop))
    return chains


def token_slots_for_noun(
    noun: str,
    text_nodes: list[ObjectNode],
    text_token_ids: torch.Tensor,
    physical_nodes: list[ObjectNode],
    physical_token_ids: torch.Tensor,
) -> list[int]:
    """All codebook indices that refer to this noun (text + icon tokens)."""
    slots: set[int] = set()
    for i, n in enumerate(text_nodes):
        if canonical_noun(n) == noun:
            slots.add(int(text_token_ids[i].item()))
    for i, n in enumerate(physical_nodes):
        if canonical_noun(n) == noun:
            slots.add(int(physical_token_ids[i].item()))
    return sorted(slots)


class MultiPropertySlotAggregator(nn.Module):
    """
    Fuse multiple rule vectors targeting the same codebook slot.

    Modes:
      - attention: learned softmax pool (default; composes YOU + FLOAT + …)
      - max: scatter_max per dimension (deterministic, no overwrite by last write)
      - sum: scatter_add + LayerNorm
    """

    def __init__(
        self,
        hidden_dim: int,
        *,
        mode: Literal["attention", "max", "sum"] = "attention",
    ) -> None:
        super().__init__()
        self.mode = mode
        self.hidden_dim = hidden_dim
        if mode == "attention":
            self.slot_query = nn.Parameter(torch.randn(1, hidden_dim) * 0.02)
            self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        elif mode == "sum":
            self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        slot_ids: torch.Tensor,
        fused: torch.Tensor,
        codebook_size: int,
        default: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            slot_ids: (M,) codebook indices (duplicates allowed)
            fused: (M, H) rule vectors to aggregate per slot
            default: (1, H) fallback for empty slots

        Returns:
            (codebook_size, H)
        """
        k, h = codebook_size, fused.size(-1)
        device = fused.device
        out = default.expand(k, -1).clone()

        if fused.numel() == 0:
            return out

        if self.mode == "max":
            return self._scatter_max(slot_ids, fused, k, default)

        if self.mode == "sum":
            return self._scatter_sum(slot_ids, fused, k, default)

        return self._attention_pool(slot_ids, fused, k, default)

    def _scatter_max(
        self,
        slot_ids: torch.Tensor,
        fused: torch.Tensor,
        k: int,
        default: torch.Tensor,
    ) -> torch.Tensor:
        out = default.expand(k, -1).clone()
        for slot in slot_ids.unique().tolist():
            mask = slot_ids == slot
            out[slot] = torch.amax(fused[mask], dim=0)
        return out

    def _scatter_sum(
        self,
        slot_ids: torch.Tensor,
        fused: torch.Tensor,
        k: int,
        default: torch.Tensor,
    ) -> torch.Tensor:
        out = default.expand(k, -1).clone()
        for slot in slot_ids.unique().tolist():
            mask = slot_ids == slot
            out[slot] = self.scale * fused[mask].sum(dim=0)
        return out

    def _attention_pool(
        self,
        slot_ids: torch.Tensor,
        fused: torch.Tensor,
        k: int,
        default: torch.Tensor,
    ) -> torch.Tensor:
        out = default.expand(k, -1).clone()
        scale = self.hidden_dim**-0.5
        for slot in slot_ids.unique().tolist():
            mask = slot_ids == slot
            vecs = fused[mask]
            if vecs.size(0) == 1:
                out[slot] = self.out_proj(vecs[0])
            else:
                logits = (vecs @ self.slot_query.T).squeeze(-1) * scale
                weights = F.softmax(logits, dim=0)
                pooled = (weights.unsqueeze(-1) * vecs).sum(dim=0)
                out[slot] = self.out_proj(pooled)
        return out


class RuleBindingHead(nn.Module):
    """
    Parse rule chains and write into (codebook_size, H) with multi-property aggregation.

    Multiple chains (BABA IS YOU, BABA IS WEAK, …) targeting the same token slot are
    merged via MultiPropertySlotAggregator — never last-write-wins overwrite.
    """

    def __init__(
        self,
        hidden_dim: int,
        codebook_size: int,
        *,
        aggregation: Literal["attention", "max", "sum"] = "attention",
    ) -> None:
        super().__init__()
        self.codebook_size = codebook_size
        self.fuse = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.default_slot = nn.Parameter(torch.zeros(1, hidden_dim))
        self.aggregator = MultiPropertySlotAggregator(hidden_dim, mode=aggregation)

    def forward(
        self,
        text_h: torch.Tensor,
        text_nodes: list[ObjectNode],
        text_token_ids: torch.Tensor,
        physical_nodes: list[ObjectNode],
        physical_token_ids: torch.Tensor,
        text_edge_index: torch.Tensor,
    ) -> torch.Tensor:
        k = self.codebook_size
        device = text_h.device
        default = self.default_slot

        if text_h.numel() == 0:
            return default.expand(k, -1).clone()

        chains = parse_rule_chains(text_nodes, text_edge_index)
        slot_ids_list: list[int] = []
        fused_list: list[torch.Tensor] = []

        for subj_idx, prop_idx in chains:
            noun = canonical_noun(text_nodes[subj_idx])
            if noun is None:
                continue
            fused = self.fuse(
                torch.cat([text_h[subj_idx], text_h[prop_idx]], dim=-1)
            )
            slots = token_slots_for_noun(
                noun,
                text_nodes,
                text_token_ids,
                physical_nodes,
                physical_token_ids,
            )
            for slot in slots:
                if 0 <= slot < k:
                    slot_ids_list.append(slot)
                    fused_list.append(fused)

        if not fused_list:
            return default.expand(k, -1).clone()

        slot_ids = torch.tensor(slot_ids_list, device=device, dtype=torch.long)
        fused_stack = torch.stack(fused_list, dim=0)
        return self.aggregator(slot_ids, fused_stack, k, default)


def gather_node_rule_context(
    rule_embeddings: torch.Tensor,
    physical_token_ids: torch.Tensor,
) -> torch.Tensor:
    """Per-physical-node rule context: (N_phys, H)."""
    if physical_token_ids.numel() == 0:
        return torch.zeros(0, rule_embeddings.size(-1), device=rule_embeddings.device)
    ids = physical_token_ids.long().to(rule_embeddings.device)
    ids = ids.clamp(0, rule_embeddings.size(0) - 1)
    return rule_embeddings[ids]
