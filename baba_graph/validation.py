"""Phase 1 V3 validation: perception split + classifier."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pyBaba

from baba_graph.perception.edges import is_directed_rule_edge
from baba_graph.perception.extract import extract_perception_from_game
from baba_graph.properties import _property_index
from baba_graph.vision.config import VisualEncoderConfig
from baba_graph.vision.perception_encoder import get_perception_encoder
from baba_graph.vocab import IGNORED_NODE_TYPES
from baba_world.env import BabaTransitionEnv
from baba_world.paths import map_path


@dataclass
class GraphCheckResult:
    name: str
    passed: bool
    message: str = ""


@dataclass
class GraphValidationReport:
    map_name: str
    results: list[GraphCheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results)

    def summary(self) -> str:
        lines = [f"V3 perception validation: {self.map_name}", "-" * 40]
        for r in self.results:
            lines.append(f"  [{'PASS' if r.passed else 'FAIL'}] {r.name}: {r.message}")
        lines.append("-" * 40)
        lines.append(f"Overall: {'PASS' if self.passed else 'FAIL'}")
        return "\n".join(lines)


def _entity_counts(game: pyBaba.Game) -> tuple[int, int, int]:
    """Returns (text_count, icon_count, collision_count=text+icons)."""
    text = icons = 0
    m = game.GetMap()
    for y in range(m.GetHeight()):
        for x in range(m.GetWidth()):
            for t in m.At(x, y).GetTypes():
                if t in IGNORED_NODE_TYPES:
                    continue
                if pyBaba.IsTextType(t):
                    text += 1
                else:
                    icons += 1
    return text, icons, text + icons


def run_graph_validations(
    map_name: str = "baba_is_you",
    *,
    min_classifier_accuracy: float = 0.85,
) -> GraphValidationReport:
    cfg = VisualEncoderConfig()
    get_perception_encoder(cfg, force_new=True)

    game = pyBaba.Game(str(map_path(map_name)))
    game.Reset()
    snap1 = extract_perception_from_game(game, config=cfg, calibrate_classifier=True)
    snap2 = extract_perception_from_game(game, config=cfg, calibrate_classifier=False)

    report = GraphValidationReport(map_name=map_name)

    if np.allclose(snap1.text.visual, snap2.text.visual) and np.allclose(
        snap1.physical.visual, snap2.physical.visual
    ):
        report.results.append(GraphCheckResult("determinism", True, "visual features stable"))
    else:
        report.results.append(GraphCheckResult("determinism", False, "visual mismatch"))

    exp_text, exp_icons, exp_collision = _entity_counts(game)
    if snap1.num_text == exp_text and snap1.num_physical == exp_collision:
        report.results.append(
            GraphCheckResult(
                "perception_split",
                True,
                f"text={exp_text} collision={exp_collision} (icons={exp_icons}+text)",
            )
        )
    else:
        report.results.append(
            GraphCheckResult(
                "perception_split",
                False,
                f"text={snap1.num_text} collision={snap1.num_physical}, "
                f"expected text={exp_text} collision={exp_collision}",
            )
        )

    mirrored = snap1.text_nodes_in_physical()
    if len(mirrored) == snap1.num_text:
        report.results.append(
            GraphCheckResult(
                "text_double_life",
                True,
                f"{len(mirrored)} text blocks in collision graph",
            )
        )
    else:
        report.results.append(
            GraphCheckResult(
                "text_double_life",
                False,
                f"mirrored {len(mirrored)} != {snap1.num_text} text nodes",
            )
        )

    push_i = _property_index(pyBaba.ObjectType.PUSH)
    stop_i = _property_index(pyBaba.ObjectType.STOP)
    prop_off = snap1.physical.visual_dim + 2
    pushable_text = 0
    stopped_text = 0
    for i, n in enumerate(snap1.physical.nodes):
        if not n.is_text:
            continue
        if snap1.physical.x[i, prop_off + stop_i] > 0.5:
            stopped_text += 1
        elif snap1.physical.x[i, prop_off + push_i] > 0.5:
            pushable_text += 1
        else:
            pushable_text = -1
            break
    if pushable_text >= 0:
        report.results.append(
            GraphCheckResult(
                "text_default_push",
                True,
                f"pushable={pushable_text} stop={stopped_text} (STOP overrides PUSH)",
            )
        )
    else:
        report.results.append(
            GraphCheckResult("text_default_push", False, "text without PUSH or STOP")
        )

    if is_directed_rule_edge(snap1.text.edge_index, snap1.text.nodes):
        report.results.append(
            GraphCheckResult(
                "text_edges_directed",
                True,
                f"{snap1.text.num_edges} forward-only (L-R/T-B) edges",
            )
        )
    else:
        report.results.append(
            GraphCheckResult("text_edges_directed", False, "backward rule edges detected")
        )

    if snap1.classifier_accuracy >= min_classifier_accuracy:
        report.results.append(
            GraphCheckResult(
                "text_physical_classifier",
                True,
                f"accuracy={snap1.classifier_accuracy:.1%}",
            )
        )
    else:
        report.results.append(
            GraphCheckResult(
                "text_physical_classifier",
                False,
                f"accuracy={snap1.classifier_accuracy:.1%} < {min_classifier_accuracy:.0%}",
            )
        )

    if snap1.text.visual.shape[1] == cfg.embed_dim and snap1.physical.visual.shape[1] == cfg.embed_dim:
        report.results.append(
            GraphCheckResult(
                "visual_dim",
                True,
                f"continuous dim={cfg.embed_dim} per node",
            )
        )
    else:
        report.results.append(GraphCheckResult("visual_dim", False, "dim mismatch"))

    if snap1.physical.num_edges > 0 or snap1.physical.num_nodes <= 1:
        report.results.append(
            GraphCheckResult("physical_edges", True, f"{snap1.physical.num_edges} spatial edges")
        )
    else:
        report.results.append(GraphCheckResult("physical_edges", False, "no physical edges"))

    env = BabaTransitionEnv(map_name)
    env.reset()
    for _ in range(5):
        env.step(0)
    snap_step = extract_perception_from_game(env._game, config=cfg, calibrate_classifier=False)
    if snap_step.num_text + snap_step.num_physical > 0:
        report.results.append(GraphCheckResult("step_rollout", True, "5 steps OK"))
    else:
        report.results.append(GraphCheckResult("step_rollout", False, "empty graph"))

    return report
