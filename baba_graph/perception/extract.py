"""V3 Phase 1 extraction API."""

from __future__ import annotations

import numpy as np
import pyBaba

from baba_graph.nodes import enumerate_nodes
from baba_graph.perception.build import build_perception_groups
from baba_graph.perception.types import PerceptionSnapshot
from baba_graph.types import ObjectNode
from baba_graph.vision.config import VisualEncoderConfig
from baba_graph.vision.patches import PatchRenderer
from baba_graph.vision.perception_encoder import get_perception_encoder
from baba_graph.vocab import IGNORED_NODE_TYPES, type_name
from baba_world.constants import CHANNEL_IS_RULE, TENSOR_CHANNEL_TYPES, TENSOR_DIM
from baba_world.rules import extract_active_rules


def extract_perception_from_game(
    game: pyBaba.Game,
    *,
    config: VisualEncoderConfig | None = None,
    map_name: str = "",
    calibrate_classifier: bool = True,
    use_classifier_split: bool = False,
) -> PerceptionSnapshot:
    """
    V3 canonical Phase 1 output: text rule graph + collision graph.

    - snap.text: text-only, directed rule edges
    - snap.physical: icons + text (text mirrored with PUSH for spatial interaction)

    Splitting uses simulator IsTextType; modality head is calibrated for diagnostics.
    """
    cfg = config or VisualEncoderConfig()
    encoder = get_perception_encoder(cfg)
    renderer = PatchRenderer(patch_size=cfg.patch_size)
    nodes = enumerate_nodes(game)

    if calibrate_classifier and nodes:
        patches = np.stack([renderer.render(n.type_name, is_text=n.is_text) for n in nodes], axis=0)
        labels = np.asarray([1 if n.is_text else 0 for n in nodes], dtype=np.int64)
        encoder.calibrate_head(patches, labels)

    text_g, phys_g, acc, preds = build_perception_groups(
        game,
        nodes,
        encoder,
        renderer,
        cfg,
        use_classifier_split=use_classifier_split,
    )

    return PerceptionSnapshot(
        text=text_g,
        physical=phys_g,
        grid_width=game.GetMap().GetWidth(),
        grid_height=game.GetMap().GetHeight(),
        active_rules=extract_active_rules(game),
        play_state=game.GetPlayState(),
        map_name=map_name,
        classifier_accuracy=acc,
        classifier_preds=preds,
    )


def _decode_types_from_tensor_cell(tensor: np.ndarray, x: int, y: int) -> list[pyBaba.ObjectType]:
    types: list[pyBaba.ObjectType] = []
    for ch, obj_type in TENSOR_CHANNEL_TYPES.items():
        if float(tensor[ch, y, x]) > 0.5:
            types.append(obj_type)
    if float(tensor[0, y, x]) > 0.5 and pyBaba.ObjectType.BABA not in types:
        if pyBaba.ObjectType.ICON_TILE not in types:
            types.append(pyBaba.ObjectType.ICON_TILE)
    return types


def extract_perception_from_tensor(
    tensor: np.ndarray,
    *,
    config: VisualEncoderConfig | None = None,
    map_name: str = "",
    active_rules: tuple[str, ...] = (),
) -> PerceptionSnapshot:
    """Secondary path from grid tensor (lossy for uncoded icons)."""
    if tensor.ndim != 3 or tensor.shape[0] != TENSOR_DIM:
        raise ValueError(f"Expected ({TENSOR_DIM}, H, W), got {tensor.shape}")

    _, height, width = tensor.shape
    rule_cells: set[tuple[int, int]] = set()
    for y in range(height):
        for x in range(width):
            if float(tensor[CHANNEL_IS_RULE, y, x]) > 0.5:
                rule_cells.add((x, y))

    nodes: list[ObjectNode] = []
    nid = 0
    for y in range(height):
        for x in range(width):
            for obj_type in _decode_types_from_tensor_cell(tensor, x, y):
                if obj_type in IGNORED_NODE_TYPES:
                    continue
                nodes.append(
                    ObjectNode(
                        node_id=nid,
                        x=x,
                        y=y,
                        object_type=obj_type,
                        type_name=type_name(obj_type),
                        is_text=bool(pyBaba.IsTextType(obj_type)),
                        is_icon=not bool(pyBaba.IsTextType(obj_type)),
                        in_rule=(x, y) in rule_cells,
                    )
                )
                nid += 1

    class _FakeMap:
        def GetWidth(self):
            return width

        def GetHeight(self):
            return height

    class _FakeGame:
        def GetMap(self):
            return _FakeMap()

    fake = _FakeGame()
    cfg = config or VisualEncoderConfig()
    encoder = get_perception_encoder(cfg)
    renderer = PatchRenderer(patch_size=cfg.patch_size)

    text_g, phys_g, acc, preds = build_perception_groups(
        fake,  # type: ignore[arg-type]
        nodes,
        encoder,
        renderer,
        cfg,
        rules=active_rules,
    )

    return PerceptionSnapshot(
        text=text_g,
        physical=phys_g,
        grid_width=width,
        grid_height=height,
        active_rules=active_rules,
        map_name=map_name,
        classifier_accuracy=acc,
        classifier_preds=preds,
    )
