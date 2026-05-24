#!/usr/bin/env python3
"""Phase 5: sandbox exploration + mixed-batch adaptation for new VQ tokens."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from baba_graph.exploration.online import (
    OnlineExplorationConfig,
    TransitionReplayBuffer,
    on_new_vq_tokens,
    seed_replay_from_rollout,
)
from baba_graph.predictor import BabaTransitionModel, PredictorConfig
from baba_graph.vq.train import load_quantizer
from baba_graph.world_model import WorldModelConfig


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 5 online adaptation")
    parser.add_argument("--map", default="baba_is_you")
    parser.add_argument("--new-tokens", type=int, nargs="+", default=[500, 501])
    parser.add_argument("--vq-checkpoint", default="")
    parser.add_argument("--adapt-steps", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    wc = WorldModelConfig()
    model = BabaTransitionModel(wc, PredictorConfig(codebook_size=wc.codebook_size))
    quantizer = load_quantizer(args.vq_checkpoint) if args.vq_checkpoint else None

    cfg = OnlineExplorationConfig(adapt_steps=args.adapt_steps)
    buffer = TransitionReplayBuffer(cfg.replay_buffer_capacity)
    print(f"Seeding replay buffer from {args.map}...", flush=True)
    n = seed_replay_from_rollout(args.map, buffer, episodes=15, max_steps=80)
    print(f"Replay buffer: {n} transitions", flush=True)

    result = on_new_vq_tokens(
        model,
        quantizer,
        args.map,
        buffer,
        args.new_tokens,
        config=cfg,
        device=args.device,
    )
    print(
        f"Adaptation: new_tokens={result.new_token_ids} "
        f"sandbox={result.sandbox_transitions} "
        f"mix=({result.new_samples} new + {result.replay_samples} replay) "
        f"loss={result.mean_loss:.4f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
