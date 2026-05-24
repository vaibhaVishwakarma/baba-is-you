#!/usr/bin/env python3
"""Run pre-training validation suite."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from baba_world.paths import SUPPORTED_MAPS
from baba_world.validation import run_all_validations


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Baba transition pipeline")
    parser.add_argument(
        "--map",
        default="baba_is_you",
        help=f"Map name (default: baba_is_you). Supported: {', '.join(SUPPORTED_MAPS)}",
    )
    args = parser.parse_args()

    report = run_all_validations(args.map)
    print(report.summary())
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
