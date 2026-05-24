#!/usr/bin/env python3
"""Print V3 perception split statistics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pyBaba

from baba_graph import extract_perception_from_game
from baba_world.paths import map_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", default="baba_is_you")
    args = parser.parse_args()

    game = pyBaba.Game(str(map_path(args.map)))
    game.Reset()
    snap = extract_perception_from_game(game)

    print(f"Map: {args.map}")
    print(f"Grid: {snap.grid_width}x{snap.grid_height}")
    print(f"Text (rule graph): {snap.num_text}  (visual {snap.text.visual.shape})")
    print(f"Physical (collision): {snap.num_physical} = {snap.num_icon_nodes} icons + {snap.num_text} text")
    print(f"  text mirrored in physical: {len(snap.text_nodes_in_physical())}")
    print(f"Classifier accuracy (text vs physical): {snap.classifier_accuracy:.1%}")
    print(f"Text edges: {snap.text.num_edges}  Physical edges: {snap.physical.num_edges}")
    print(f"Rules: {snap.active_rules}")
    print("\nText nodes (id, type, pos):")
    for n in snap.text.nodes[:12]:
        print(f"  {n.node_id:3d}  {n.type_name:12s}  ({n.x},{n.y})")
    print("\nPhysical nodes (id, type, pos):")
    for n in snap.physical.nodes[:12]:
        print(f"  {n.node_id:3d}  {n.type_name:12s}  ({n.x},{n.y})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
