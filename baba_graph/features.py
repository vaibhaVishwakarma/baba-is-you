"""Assemble node feature matrix: visual embedding + position + properties."""

from __future__ import annotations

import numpy as np
import pyBaba

from baba_graph.properties import property_multi_hot
from baba_graph.types import ObjectNode
from baba_graph.vision.config import VisualEncoderConfig
from baba_graph.vision.cache import get_visual_embedding_cache
from baba_graph.vision.encoder import FrozenVisualEncoder, get_visual_encoder
from baba_graph.vision.patches import PatchRenderer
from baba_graph.vision.perception_encoder import PerceptionEncoder
from baba_graph.vocab import OBJECT_PROPERTY_TYPES


def node_feature_dim(config: VisualEncoderConfig | None = None) -> int:
    cfg = config or VisualEncoderConfig()
    return cfg.node_feature_dim()


def build_node_features(
    game: pyBaba.Game,
    nodes: list[ObjectNode],
    *,
    encoder: FrozenVisualEncoder | None = None,
    patch_renderer: PatchRenderer | None = None,
    config: VisualEncoderConfig | None = None,
    rules: tuple[str, ...] | None = None,
) -> tuple[np.ndarray, VisualEncoderConfig]:
    """
    Build x with shape (num_nodes, feature_dim).

    Layout: [visual_embed (D) | pos_x, pos_y | property multi-hot (P)]
    """
    cfg = config or VisualEncoderConfig()
    encoder = encoder or get_visual_encoder(cfg)
    renderer = patch_renderer or PatchRenderer(patch_size=cfg.patch_size)

    width = game.GetMap().GetWidth()
    height = game.GetMap().GetHeight()
    denom_x = max(width - 1, 1)
    denom_y = max(height - 1, 1)

    if not nodes:
        return np.zeros((0, cfg.node_feature_dim()), dtype=np.float32), cfg

    patches = np.stack(
        [renderer.render(n.type_name, is_text=n.is_text) for n in nodes],
        axis=0,
    )
    if isinstance(encoder, PerceptionEncoder):
        visual = get_visual_embedding_cache().encode_visual(encoder, patches)
    else:
        visual = encoder.encode(patches)

    dim = cfg.node_feature_dim()
    x = np.zeros((len(nodes), dim), dtype=np.float32)
    offset_pos = cfg.embed_dim
    offset_props = offset_pos + 2

    for i, node in enumerate(nodes):
        x[i, : cfg.embed_dim] = visual[i]
        x[i, offset_pos] = node.x / denom_x
        x[i, offset_pos + 1] = node.y / denom_y
        if rules is not None:
            from baba_graph.properties import property_multi_hot_from_rules

            x[i, offset_props:] = property_multi_hot_from_rules(rules, node.object_type)
        else:
            x[i, offset_props:] = property_multi_hot(game, node.object_type)

    return x, cfg


def build_node_features_from_rules(
    nodes: list[ObjectNode],
    *,
    width: int,
    height: int,
    active_rules: tuple[str, ...],
    encoder: FrozenVisualEncoder | None = None,
    patch_renderer: PatchRenderer | None = None,
    config: VisualEncoderConfig | None = None,
) -> tuple[np.ndarray, VisualEncoderConfig]:
    """Tensor-only path: properties derived from rule strings."""
    cfg = config or VisualEncoderConfig()
    encoder = encoder or get_visual_encoder(cfg)
    renderer = patch_renderer or PatchRenderer(patch_size=cfg.patch_size)

    if not nodes:
        return np.zeros((0, cfg.node_feature_dim()), dtype=np.float32), cfg

    patches = np.stack(
        [renderer.render(n.type_name, is_text=n.is_text) for n in nodes],
        axis=0,
    )
    if isinstance(encoder, PerceptionEncoder):
        visual = get_visual_embedding_cache().encode_visual(encoder, patches)
    else:
        visual = encoder.encode(patches)

    denom_x = max(width - 1, 1)
    denom_y = max(height - 1, 1)
    dim = cfg.node_feature_dim()
    x = np.zeros((len(nodes), dim), dtype=np.float32)
    offset_pos = cfg.embed_dim
    offset_props = offset_pos + 2

    from baba_graph.properties import property_multi_hot_from_rules

    for i, node in enumerate(nodes):
        x[i, : cfg.embed_dim] = visual[i]
        x[i, offset_pos] = node.x / denom_x
        x[i, offset_pos + 1] = node.y / denom_y
        x[i, offset_props:] = property_multi_hot_from_rules(active_rules, node.object_type)

    return x, cfg
