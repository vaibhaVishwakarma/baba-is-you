"""EMA vector quantizer with STE and dynamic codebook expansion."""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from baba_graph.vq.config import VQConfig


class VQOutput(NamedTuple):
    z_q: torch.Tensor
    z_e: torch.Tensor
    indices: torch.Tensor
    commitment_loss: torch.Tensor
    codebook_loss: torch.Tensor
    perplexity: torch.Tensor
    expanded: int


class DynamicVectorQuantizer(nn.Module):
    """
    Vector quantizer with EMA codebook updates, straight-through estimator,
    and optional dynamic codebook expansion for open-vocabulary objects.
    """

    def __init__(self, config: VQConfig | None = None) -> None:
        super().__init__()
        self.config = config or VQConfig()
        c = self.config

        embed = torch.randn(c.num_codes, c.embed_dim) * 0.02
        self.register_buffer("embedding", embed)
        self.register_buffer("ema_cluster_size", torch.ones(c.num_codes))
        self.register_buffer("ema_embedding", embed.clone())
        self._num_codes = c.num_codes
        self.expansions_total = 0

    @property
    def num_codes(self) -> int:
        return self._num_codes

    @property
    def embedding_dim(self) -> int:
        return self.config.embed_dim

    def _pairwise_distances(self, z_e: torch.Tensor) -> torch.Tensor:
        """Squared L2 distances (N, K)."""
        z2 = (z_e**2).sum(dim=1, keepdim=True)
        e2 = (self.embedding[: self._num_codes] ** 2).sum(dim=1)
        ze = z_e @ self.embedding[: self._num_codes].T
        return z2 + e2.unsqueeze(0) - 2 * ze

    def _encode_indices(self, z_e: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        distances = self._pairwise_distances(z_e)
        min_dist, indices = distances.min(dim=1)
        return indices, min_dist

    @torch.no_grad()
    def _ema_update(self, z_e: torch.Tensor, indices: torch.Tensor) -> None:
        c = self.config
        k = self._num_codes
        one_hot = F.one_hot(indices, k).float()
        cluster_size = one_hot.sum(dim=0)
        embed_sum = one_hot.T @ z_e

        self.ema_cluster_size[:k] = (
            c.ema_decay * self.ema_cluster_size[:k] + (1 - c.ema_decay) * cluster_size
        )
        self.ema_embedding[:k] = (
            c.ema_decay * self.ema_embedding[:k] + (1 - c.ema_decay) * embed_sum
        )

        n = self.ema_cluster_size[:k].sum()
        normalized = (self.ema_cluster_size[:k] + c.ema_epsilon) / (n + k * c.ema_epsilon) * n
        self.embedding[:k] = self.ema_embedding[:k] / normalized.unsqueeze(1)

    def _expand_codes(self, z_e: torch.Tensor, mask: torch.Tensor) -> int:
        """Add new codebook entries for vectors far from all existing codes."""
        c = self.config
        if not c.enable_expansion or not mask.any():
            return 0

        newcomers = z_e[mask]
        n_new = newcomers.shape[0]
        room = c.max_codes - self._num_codes
        if room <= 0:
            return 0
        n_add = min(n_new, room)
        if n_add < n_new:
            newcomers = newcomers[:n_add]

        old_k = self._num_codes
        new_k = old_k + n_add

        new_emb = torch.cat([self.embedding[:old_k], newcomers.detach()], dim=0)
        new_cluster = torch.cat(
            [self.ema_cluster_size[:old_k], torch.ones(n_add, device=z_e.device)],
            dim=0,
        )
        new_ema_emb = torch.cat(
            [self.ema_embedding[:old_k], newcomers.detach()], dim=0
        )

        pad = c.max_codes - new_k
        if pad > 0:
            d = self.embedding.shape[1]
            new_emb = torch.cat(
                [new_emb, torch.zeros(pad, d, device=z_e.device)], dim=0
            )
            new_cluster = torch.cat(
                [new_cluster, torch.zeros(pad, device=z_e.device)], dim=0
            )
            new_ema_emb = torch.cat(
                [new_ema_emb, torch.zeros(pad, d, device=z_e.device)], dim=0
            )

        self.embedding = new_emb
        self.ema_cluster_size = new_cluster
        self.ema_embedding = new_ema_emb
        self._num_codes = new_k
        self.expansions_total += n_add
        return n_add

    def forward(self, z_e: torch.Tensor) -> VQOutput:
        """
        Args:
            z_e: (N, D) continuous encoder outputs

        Returns:
            VQOutput with STE-quantized z_q and scalar losses
        """
        if z_e.dim() != 2:
            raise ValueError(f"Expected (N, D), got {z_e.shape}")

        indices, min_dist = self._encode_indices(z_e)
        expanded = 0

        if self.training and self.config.enable_expansion:
            far = min_dist > self.config.expansion_threshold**2
            expanded = self._expand_codes(z_e, far)
            if expanded > 0:
                indices, min_dist = self._encode_indices(z_e)

        z_q = self.embedding[indices]

        # Straight-through estimator
        z_q_st = z_e + (z_q - z_e).detach()

        commitment = F.mse_loss(z_e, z_q.detach())
        codebook = F.mse_loss(z_q, z_e.detach())
        commitment_loss = self.config.commitment_beta * commitment
        codebook_loss = codebook

        if self.training:
            self._ema_update(z_e.detach(), indices)

        # Perplexity proxy (code usage diversity)
        k = self._num_codes
        one_hot = F.one_hot(indices, k).float()
        avg_prob = one_hot.mean(dim=0)
        perplexity = torch.exp(-torch.sum(avg_prob * torch.log(avg_prob + 1e-10)))

        return VQOutput(
            z_q=z_q_st,
            z_e=z_e,
            indices=indices,
            commitment_loss=commitment_loss,
            codebook_loss=codebook_loss,
            perplexity=perplexity,
            expanded=expanded,
        )

    @torch.no_grad()
    def encode(self, z_e: torch.Tensor) -> torch.Tensor:
        """Return token indices for continuous vectors."""
        indices, _ = self._encode_indices(z_e)
        return indices

    @torch.no_grad()
    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        return self.embedding[indices]
