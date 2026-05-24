#!/usr/bin/env python3
"""Train Phase 3 dual-graph world model (representation warmup)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from baba_graph.world_model import (
    DualGraphWorldModel,
    WorldModelConfig,
    collect_dynamics_transitions,
    train_representation_warmup,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train dual-graph world model (Phase 3)")
    parser.add_argument("--map", default="baba_is_you")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    print(f"Collecting biased transitions: {args.map}", flush=True)
    transitions = collect_dynamics_transitions(
        args.map,
        episodes=args.episodes,
        max_steps=args.max_steps,
    )
    rule_changes = sum(1 for t in transitions if t.rules_before != t.rules_after)
    print(
        f"Collected {len(transitions)} transitions, {rule_changes} rule mutations",
        flush=True,
    )

    model = DualGraphWorldModel(WorldModelConfig())
    history = train_representation_warmup(
        model,
        transitions,
        epochs=args.epochs,
        device=args.device,
    )
    if history:
        print(f"Final loss: {history[-1].loss:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
