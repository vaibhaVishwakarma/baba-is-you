#!/usr/bin/env python3
"""Train Phase 4 discrete transition predictor."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from baba_graph.device import resolve_device
from baba_graph.predictor import BabaTransitionModel, PredictorConfig, train_predictor
from baba_graph.predictor.data import collect_tokenized_transitions
from baba_graph.vq.train import load_quantizer
from baba_graph.world_model import WorldModelConfig


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Phase 4 transition predictor")
    parser.add_argument("--map", default="baba_is_you")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--codebook-size", type=int, default=512)
    parser.add_argument("--vq-checkpoint", default="", help="Optional VQ codebook .pt")
    parser.add_argument(
        "--device",
        default="auto",
        help="auto | cpu | cuda (requires PyTorch built with CUDA)",
    )
    args = parser.parse_args()
    device = resolve_device(args.device)
    print(f"Using device: {device}", flush=True)

    quantizer = None
    if args.vq_checkpoint:
        quantizer = load_quantizer(args.vq_checkpoint, device=device)

    print(f"Collecting transitions: {args.map}", flush=True)
    transitions = collect_tokenized_transitions(
        args.map,
        episodes=args.episodes,
        max_steps=args.max_steps,
    )
    rule_changes = sum(1 for t in transitions if t.rules_before != t.rules_after)
    print(
        f"{len(transitions)} transitions, {rule_changes} rule mutations",
        flush=True,
    )

    wc = WorldModelConfig(codebook_size=args.codebook_size)
    pc = PredictorConfig(codebook_size=args.codebook_size)
    model = BabaTransitionModel(wc, pc)

    history = train_predictor(
        model,
        transitions,
        quantizer,
        epochs=args.epochs,
        device=device,
    )
    if history:
        last = history[-1]
        print(
            f"Epoch {last.epoch}: loss={last.loss:.4f} "
            f"id_acc={last.identity_accuracy:.2%} "
            f"mov_acc={last.movement_accuracy:.2%} pairs={last.num_pairs}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
