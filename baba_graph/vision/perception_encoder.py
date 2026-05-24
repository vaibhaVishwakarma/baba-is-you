"""Frozen patch encoder + text/physical classification head (V3 Phase 1)."""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from baba_graph.vision.config import VisualEncoderConfig
from baba_graph.vision.encoder import _ResNet18PatchCNN, _TinyPatchCNN

_default_perception_encoder: PerceptionEncoder | None = None

# 0 = physical, 1 = text
LABEL_PHYSICAL = 0
LABEL_TEXT = 1


class PerceptionEncoder(nn.Module):
    """
    Frozen visual backbone + binary head (text vs physical).

    Backbone weights stay frozen in Phase 1; the head can be calibrated on
    simulator-labelled patches (see `calibrate_head`).
    """

    def __init__(
        self,
        backbone: nn.Module,
        embed_dim: int,
        *,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.embed_dim = embed_dim
        self.modality_head = nn.Linear(embed_dim, 2)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    @classmethod
    def create(
        cls,
        backbone_name: Literal["tiny", "resnet18"] = "tiny",
        *,
        embed_dim: int = 128,
        pretrained: bool = False,
        device: str = "cpu",
        freeze_backbone: bool = True,
    ) -> PerceptionEncoder:
        if backbone_name == "tiny":
            core = _TinyPatchCNN(embed_dim=embed_dim)
        elif backbone_name == "resnet18":
            core = _ResNet18PatchCNN(embed_dim=embed_dim, pretrained=pretrained)
        else:
            raise ValueError(backbone_name)
        return cls(core, embed_dim, freeze_backbone=freeze_backbone).to(device)

    def forward(self, patches: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """patches (N,3,H,W) -> embeddings (N,D), logits (N,2)."""
        emb = self.backbone(patches)
        return emb, self.modality_head(emb)

    @torch.no_grad()
    def encode_visual(self, patches: np.ndarray | torch.Tensor) -> np.ndarray:
        self.eval()
        if isinstance(patches, np.ndarray):
            t = torch.from_numpy(patches).float()
        else:
            t = patches.float()
        device = next(self.parameters()).device
        t = t.to(device)
        if t.dim() == 3:
            t = t.unsqueeze(0)
        emb = self.backbone(t)
        return emb.cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def predict_modality(self, patches: np.ndarray | torch.Tensor) -> np.ndarray:
        """Return 0=physical, 1=text per patch."""
        self.eval()
        if isinstance(patches, np.ndarray):
            t = torch.from_numpy(patches).float()
        else:
            t = patches.float()
        device = next(self.parameters()).device
        t = t.to(device)
        if t.dim() == 3:
            t = t.unsqueeze(0)
        _, logits = self.forward(t)
        return logits.argmax(dim=-1).cpu().numpy().astype(np.int64)

    def calibrate_head(
        self,
        patches: np.ndarray,
        labels: np.ndarray,
        *,
        steps: int = 800,
        lr: float = 5e-2,
    ) -> float:
        """
        Train only the modality head on simulator-labelled patches.

        Returns final accuracy on the provided batch.
        """
        self.train()
        for p in self.backbone.parameters():
            p.requires_grad = False

        device = next(self.parameters()).device
        x = torch.from_numpy(patches).float().to(device)
        y = torch.from_numpy(labels).long().to(device)
        opt = torch.optim.Adam(self.modality_head.parameters(), lr=lr)

        for _ in range(steps):
            opt.zero_grad()
            with torch.no_grad():
                emb = self.backbone(x)
            logits = self.modality_head(emb)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()

        self.eval()
        with torch.no_grad():
            pred = self.predict_modality(patches)
        return float((pred == labels).mean())


def get_perception_encoder(
    config: VisualEncoderConfig | None = None,
    *,
    force_new: bool = False,
) -> PerceptionEncoder:
    global _default_perception_encoder
    if force_new or _default_perception_encoder is None:
        cfg = config or VisualEncoderConfig()
        _default_perception_encoder = PerceptionEncoder.create(
            cfg.backbone,  # type: ignore[arg-type]
            embed_dim=cfg.embed_dim,
            pretrained=cfg.pretrained,
            device=cfg.device,
        )
    return _default_perception_encoder
