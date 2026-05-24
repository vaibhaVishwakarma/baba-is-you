#!/usr/bin/env python3
"""Train Phase 2 VQ codebook on perception visual vectors."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from baba_graph.device import resolve_device
from baba_graph.vq.config import VQConfig
from baba_graph.vq.train import collect_and_train, save_quantizer


def main() -> int:
    parser = argparse.ArgumentParser(description="Train VQ bottleneck (Phase 2)")
    parser.add_argument("--maps", nargs="*", default=["baba_is_you"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--num-codes", type=int, default=512)
    parser.add_argument("--max-codes", type=int, default=2048)
    parser.add_argument(
        "--output",
        default="data/vq/codebook.pt",
        help="Checkpoint path",
    )
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda")
    parser.add_argument(
        "--policy",
        choices=("biased", "random"),
        default="biased",
        help="Rollout policy for data collection",
    )
    args = parser.parse_args()
    device = resolve_device(args.device)
    print(f"Using device: {device}", flush=True)

    vq_cfg = VQConfig(num_codes=args.num_codes, max_codes=args.max_codes)

    print(f"Collecting visual vectors from maps: {args.maps}", flush=True)
    quantizer, data, history = collect_and_train(
        maps=args.maps,
        vq_config=vq_cfg,
        epochs=args.epochs,
        batch_size=args.batch_size,
        episodes_per_map=args.episodes,
        max_steps=args.max_steps,
        policy=args.policy,
        device=device,
    )
    print(f"Collected {len(data)} visual vectors (dim={data.vectors.shape[1]})", flush=True)

    if history:
        last = history[-1]
        print(
            f"Epoch {last.epoch}: loss={last.loss:.4f} "
            f"perplexity={last.perplexity:.1f} codes={last.num_codes} "
            f"expansions={last.expansions}"
        )

    path = save_quantizer(quantizer, args.output)
    print(f"Saved codebook to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
