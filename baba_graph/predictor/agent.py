"""One-step lookahead agent for GUI play (pre-MCTS Phase 6)."""

from __future__ import annotations

import pyBaba

from baba_graph import extract_perception_from_game
from baba_graph.perception.properties_view import (
    node_indices_with_property,
    win_positions,
    you_positions,
)
from baba_graph.perception.types import PerceptionSnapshot
from baba_graph.predictor.model import BabaTransitionModel
from baba_graph.predictor.movement import apply_adjacency
from baba_graph.world_model.model import snapshot_to_tensors
from baba_world.actions import decode_action
from baba_world.paths import map_path


def _filter_you_cells(positions: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Drop padding ghost at (0, 0) when real YOU cells exist elsewhere."""
    real = [p for p in positions if p != (0, 0)]
    return real if real else list(positions)


def _min_manhattan(src: tuple[int, int], targets: list[tuple[int, int]]) -> float:
    if not targets:
        return 0.0
    return float(min(abs(src[0] - t[0]) + abs(src[1] - t[1]) for t in targets))


def _predicted_you_positions(
    snap: PerceptionSnapshot,
    movement_classes: list[int],
) -> list[tuple[int, int]]:
    """Apply movement head predictions to every YOU node (any noun)."""
    group = snap.physical
    you_idx = node_indices_with_property(group, pyBaba.ObjectType.YOU)
    pred: list[tuple[int, int]] = []
    for i in you_idx:
        n = group.nodes[i]
        pred.append(apply_adjacency(n.x, n.y, movement_classes[i]))
    return pred


def score_predicted_state(
    model: BabaTransitionModel,
    snap: PerceptionSnapshot,
    snap_tensors: dict,
    action: int,
    *,
    device: str = "cpu",
) -> float:
    """
    Score action by predicted YOU → WIN distance.

    Uses property embeddings from the physical graph (Phase 1/3), not hardcoded
    FLAG nouns — works when ROCK IS WIN, SKULL IS WIN, etc.
    """
    import torch

    model.eval()
    with torch.no_grad():
        out = model(snap_tensors, action)

    win_cells = _filter_you_cells(win_positions(snap))
    pred_you = _filter_you_cells(
        _predicted_you_positions(
            snap, out.movement_logits.argmax(dim=-1).cpu().tolist()
        )
    )
    if not pred_you:
        return -1e6
    if not win_cells:
        # No WIN in rules yet (common on puzzle maps) — nudge toward text for rule edits.
        text_cells = [(n.x, n.y) for n in snap.text.nodes]
        if text_cells:
            return -sum(_min_manhattan(p, text_cells) for p in pred_you) / len(pred_you)
        return -1e3

    total = sum(_min_manhattan(p, win_cells) for p in pred_you)
    return -total / len(pred_you)


def _you_moves_in_simulator(
    map_key: str,
    history: list[int],
    action: int,
) -> bool:
    """True if this action changes YOU cells in the real simulator (not just the model)."""
    game = pyBaba.Game(str(map_path(map_key)))
    game.Reset()
    for a in history:
        game.MovePlayer(decode_action(a))
    before = _filter_you_cells(
        you_positions(extract_perception_from_game(game, calibrate_classifier=False))
    )
    game.MovePlayer(decode_action(action))
    after = _filter_you_cells(
        you_positions(extract_perception_from_game(game, calibrate_classifier=False))
    )
    return before != after


def choose_action_lookahead(
    game: pyBaba.Game,
    model: BabaTransitionModel,
    *,
    map_key: str = "",
    history: list[int] | None = None,
    codebook_size: int | None = None,
    device: str = "cpu",
) -> int:
    """Pick action 0–3 via one-step model lookahead on (state, action) pairs."""
    cb = codebook_size or model.world_config.codebook_size

    snap = extract_perception_from_game(game, calibrate_classifier=False)
    base_tensors = snapshot_to_tensors(snap, device=device, codebook_size=cb)

    scored: list[tuple[float, int]] = []
    for action in range(4):
        score = score_predicted_state(
            model, snap, base_tensors, action, device=device
        )
        scored.append((score, action))

    best_score = max(s for s, _ in scored)
    candidates = [a for s, a in scored if s >= best_score - 1e-5]
    if len(candidates) == 1:
        return candidates[0]

    # Model ties (often STAY-heavy on unseen maps): prefer a move that actually shifts YOU.
    if map_key and history is not None:
        for action in candidates:
            if _you_moves_in_simulator(map_key, history, action):
                return action
        for action in range(4):
            if action not in candidates and _you_moves_in_simulator(map_key, history, action):
                return action

    return candidates[0]


def replay_and_choose(
    map_key: str,
    action_history: list[int],
    model: BabaTransitionModel,
    *,
    device: str = "cpu",
) -> int:
    """Replay `action_history` on a fresh game, then run lookahead."""
    from baba_world.actions import decode_action
    from baba_world.paths import map_path

    game = pyBaba.Game(str(map_path(map_key)))
    game.Reset()
    for a in action_history:
        game.MovePlayer(decode_action(a))
    return choose_action_lookahead(
        game,
        model,
        map_key=map_key,
        history=action_history,
        codebook_size=model.world_config.codebook_size,
        device=device,
    )
