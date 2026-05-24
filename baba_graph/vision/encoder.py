"""Frozen visual encoders for object patches (open-vocabulary)."""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch
import torch.nn as nn

from baba_graph.vision.config import VisualEncoderConfig

_default_encoder: FrozenVisualEncoder | None = None


class FrozenVisualEncoder(nn.Module):
    """
    Maps object patches (3, H, W) -> continuous embedding vectors.

    Weights are frozen after construction (Phase 1 does not train this).
    """

    def __init__(self, backbone: nn.Module, embed_dim: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.embed_dim = embed_dim
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    @classmethod
    def create(
        cls,
        backbone_name: Literal["tiny", "resnet18"] = "tiny",
        *,
        embed_dim: int = 128,
        pretrained: bool = False,
        device: str = "cpu",
    ) -> FrozenVisualEncoder:
        if backbone_name == "tiny":
            core = _TinyPatchCNN(embed_dim=embed_dim)
        elif backbone_name == "resnet18":
            core = _ResNet18PatchCNN(embed_dim=embed_dim, pretrained=pretrained)
        else:
            raise ValueError(f"Unknown backbone: {backbone_name}")
        model = cls(core, embed_dim)
        return model.to(device)

    @torch.no_grad()
    def encode(self, patches: np.ndarray | torch.Tensor) -> np.ndarray:
        """
        Encode patches to (N, embed_dim).

        patches: (N, 3, H, W) float32 in [0, 1]
        """
        if isinstance(patches, np.ndarray):
            t = torch.from_numpy(patches).float()
        else:
            t = patches.float()

        device = next(self.parameters()).device
        t = t.to(device)
        if t.dim() == 3:
            t = t.unsqueeze(0)

        out = self.backbone(t)
        return out.cpu().numpy().astype(np.float32)

    def encode_single(self, patch: np.ndarray) -> np.ndarray:
        return self.encode(patch[np.newaxis, ...])[0]


class _TinyPatchCNN(nn.Module):
    """Lightweight CNN (no download required)."""

    def __init__(self, embed_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _ResNet18PatchCNN(nn.Module):
    """Truncated ResNet-18 for patch embeddings."""

    def __init__(self, embed_dim: int = 128, pretrained: bool = False) -> None:
        super().__init__()
        from torchvision import models

        weights = None
        if pretrained:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
        resnet = models.resnet18(weights=weights)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.proj = nn.Linear(512, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x)
        h = h.view(h.size(0), -1)
        return self.proj(h)


def get_visual_encoder(
    config: VisualEncoderConfig | None = None,
    *,
    force_new: bool = False,
) -> FrozenVisualEncoder:
    """Shared frozen encoder instance."""
    global _default_encoder
    if force_new or _default_encoder is None:
        cfg = config or VisualEncoderConfig()
        _default_encoder = FrozenVisualEncoder.create(
            cfg.backbone,  # type: ignore[arg-type]
            embed_dim=cfg.embed_dim,
            pretrained=cfg.pretrained,
            device=cfg.device,
        )
    return _default_encoder
