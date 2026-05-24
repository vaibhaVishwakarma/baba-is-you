#!/usr/bin/env python3
"""Run Phase 1 graph extraction validation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from baba_graph.validation import run_graph_validations


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate object-centric graph extraction")
    parser.add_argument("--map", default="baba_is_you")
    args = parser.parse_args()

    report = run_graph_validations(args.map)
    print(report.summary())
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
