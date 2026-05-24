"""Build text/physical NodeGroups from enumerated nodes."""

from __future__ import annotations

import numpy as np
import pyBaba

from baba_graph.perception.edges import build_physical_edges, build_text_rule_edges
from baba_graph.perception.types import NodeGroup
from baba_graph.properties import (
    property_multi_hot,
    property_multi_hot_from_rules,
    text_physics_properties,
)
from baba_graph.types import ObjectNode
from baba_graph.vision.config import VisualEncoderConfig
from baba_graph.vision.patches import PatchRenderer
from baba_graph.vision.cache import VisualEmbeddingCache, get_visual_embedding_cache
from baba_graph.vision.perception_encoder import LABEL_PHYSICAL, LABEL_TEXT, PerceptionEncoder


def split_nodes(nodes: list[ObjectNode]) -> tuple[list[ObjectNode], list[ObjectNode]]:
    text = [n for n in nodes if n.is_text]
    physical_icons = [n for n in nodes if not n.is_text]
    return text, physical_icons


def _assign_fresh_ids(nodes: list[ObjectNode]) -> list[ObjectNode]:
    out: list[ObjectNode] = []
    for i, n in enumerate(nodes):
        out.append(
            ObjectNode(
                node_id=i,
                x=n.x,
                y=n.y,
                object_type=n.object_type,
                type_name=n.type_name,
                is_text=n.is_text,
                is_icon=n.is_icon,
                in_rule=n.in_rule,
            )
        )
    return out


def text_feature_dim(embed_dim: int) -> int:
    return embed_dim + 2


def physical_feature_dim(embed_dim: int) -> int:
    from baba_graph.vocab import OBJECT_PROPERTY_TYPES

    return embed_dim + 2 + len(OBJECT_PROPERTY_TYPES)


def build_node_group(
    nodes: list[ObjectNode],
    visual: np.ndarray,
    *,
    modality: str,
    width: int,
    height: int,
    embed_dim: int,
    properties: list[list[float]] | None = None,
) -> NodeGroup:
    nodes = _assign_fresh_ids(nodes)
    denom_x = max(width - 1, 1)
    denom_y = max(height - 1, 1)

    if modality == "text":
        dim = text_feature_dim(embed_dim)
        edge_index = build_text_rule_edges(nodes)
    elif modality == "physical":
        dim = physical_feature_dim(embed_dim)
        edge_index = build_physical_edges(nodes)
    else:
        raise ValueError(modality)

    x = np.zeros((len(nodes), dim), dtype=np.float32)
    for i, node in enumerate(nodes):
        x[i, :embed_dim] = visual[i]
        x[i, embed_dim] = node.x / denom_x
        x[i, embed_dim + 1] = node.y / denom_y
        if modality == "physical" and properties is not None:
            x[i, embed_dim + 2 :] = properties[i]

    return NodeGroup(
        modality=modality,
        nodes=nodes,
        x=x,
        edge_index=edge_index,
        visual=visual,
        feature_dim=dim,
        visual_dim=embed_dim,
    )


def _physics_properties_for_node(
    game: pyBaba.Game,
    node: ObjectNode,
    *,
    rules: tuple[str, ...] | None,
) -> list[float]:
    if node.is_text:
        return text_physics_properties(game, node.object_type, active_rules=rules)
    if rules is not None:
        return property_multi_hot_from_rules(rules, node.object_type)
    return property_multi_hot(game, node.object_type)


def build_perception_groups(
    game: pyBaba.Game,
    nodes: list[ObjectNode],
    encoder: PerceptionEncoder,
    renderer: PatchRenderer,
    config: VisualEncoderConfig,
    *,
    rules: tuple[str, ...] | None = None,
    use_classifier_split: bool = False,
    visual_cache: VisualEmbeddingCache | None = None,
) -> tuple[NodeGroup, NodeGroup, float, np.ndarray]:
    """
    Encode nodes into text (rule graph) and physical (collision graph) groups.

    Text blocks lead a double life:
      - snap.text: directed rule-reading edges, visual + position
      - snap.physical: includes the same text blocks with PUSH + spatial edges
    """
    width = game.GetMap().GetWidth()
    height = game.GetMap().GetHeight()

    if not nodes:
        empty = np.zeros((0, config.embed_dim), dtype=np.float32)
        return (
            build_node_group([], empty, modality="text", width=width, height=height, embed_dim=config.embed_dim),
            build_node_group([], empty, modality="physical", width=width, height=height, embed_dim=config.embed_dim),
            1.0,
            np.zeros(0, dtype=np.int64),
        )

    patches = np.stack([renderer.render(n.type_name, is_text=n.is_text) for n in nodes], axis=0)
    cache = visual_cache if visual_cache is not None else get_visual_embedding_cache()
    visual = cache.encode_visual(encoder, patches)
    gt_labels = np.asarray(
        [LABEL_TEXT if n.is_text else LABEL_PHYSICAL for n in nodes], dtype=np.int64
    )
    preds = encoder.predict_modality(patches)
    accuracy = float((preds == gt_labels).mean())

    if use_classifier_split:
        text_nodes = [n for n, p in zip(nodes, preds) if p == LABEL_TEXT]
        icon_nodes = [n for n, p in zip(nodes, preds) if p == LABEL_PHYSICAL]
        text_visual = visual[preds == LABEL_TEXT]
        icon_visual = visual[preds == LABEL_PHYSICAL]
    else:
        text_nodes, icon_nodes = split_nodes(nodes)
        text_idx = [i for i, n in enumerate(nodes) if n.is_text]
        icon_idx = [i for i, n in enumerate(nodes) if not n.is_text]
        text_visual = visual[text_idx] if text_idx else np.zeros((0, config.embed_dim), np.float32)
        icon_visual = visual[icon_idx] if icon_idx else np.zeros((0, config.embed_dim), np.float32)

    # Rule graph: text only
    text_group = build_node_group(
        text_nodes, text_visual, modality="text", width=width, height=height, embed_dim=config.embed_dim
    )

    # Collision graph: icons + text (text mirrored for push / spatial interaction)
    collision_nodes = icon_nodes + text_nodes
    collision_visual = (
        np.vstack([icon_visual, text_visual])
        if len(icon_nodes) and len(text_nodes)
        else (icon_visual if len(icon_nodes) else text_visual)
    )
    if not collision_nodes:
        collision_visual = np.zeros((0, config.embed_dim), np.float32)

    collision_props = [
        _physics_properties_for_node(game, n, rules=rules) for n in collision_nodes
    ]

    phys_group = build_node_group(
        collision_nodes,
        collision_visual,
        modality="physical",
        width=width,
        height=height,
        embed_dim=config.embed_dim,
        properties=collision_props,
    )
    return text_group, phys_group, accuracy, preds
