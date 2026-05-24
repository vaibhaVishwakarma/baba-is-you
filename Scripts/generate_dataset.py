#!/usr/bin/env python3
"""Generate offline transition dataset after validation passes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from baba_world.dataset import episodes_to_dataset, save_dataset
from baba_world.generator import generate_episodes
from baba_world.validation import run_all_validations


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate transition dataset (.npz)")
    parser.add_argument("--map", default="baba_is_you")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        default="data/transitions/baba_is_you.npz",
        help="Output path (writes .npz and .meta.json)",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip pre-training validation (not recommended)",
    )
    args = parser.parse_args()

    if not args.skip_validation:
        report = run_all_validations(args.map)
        print(report.summary())
        if not report.passed:
            print("Validation failed; fix issues before generating dataset.", file=sys.stderr)
            return 1
        print()

    print(
        f"Generating {args.episodes} episodes "
        f"(max_steps={args.max_steps}, seed={args.seed})..."
    )
    episodes = generate_episodes(
        args.map,
        args.episodes,
        max_steps=args.max_steps,
        seed=args.seed,
    )
    dataset = episodes_to_dataset(episodes)
    path = save_dataset(dataset, args.output)
    print(f"Saved {dataset.num_transitions} transitions to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
