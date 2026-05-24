"""Visual encoder configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VisualEncoderConfig:
    patch_size: int = 32
    embed_dim: int = 128
    backbone: str = "tiny"  # "tiny" | "resnet18"
    pretrained: bool = False
    device: str = "cpu"

    @property
    def property_dim(self) -> int:
        from baba_graph.vocab import OBJECT_PROPERTY_TYPES

        return len(OBJECT_PROPERTY_TYPES)

    def node_feature_dim(self) -> int:
        """visual embedding + normalized (x,y) + property multi-hot."""
        return self.embed_dim + 2 + self.property_dim
