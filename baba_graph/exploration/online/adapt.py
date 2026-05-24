"""Few-shot mixed-batch adaptation after VQ token expansion."""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch

from baba_graph.exploration.online.buffer import TransitionReplayBuffer
from baba_graph.exploration.online.config import OnlineExplorationConfig
from baba_graph.exploration.online.sandbox import collect_sandbox_transitions
from baba_graph.exploration.online.trigger import TokenExpansionTracker
from baba_graph.predictor.model import BabaTransitionModel
from baba_graph.predictor.train import transition_loss
from baba_graph.vq.quantizer import DynamicVectorQuantizer


@dataclass
class AdaptationResult:
    new_token_ids: list[int]
    sandbox_transitions: int
    replay_samples: int
    new_samples: int
    steps: int
    mean_loss: float


def build_mixed_batch(
    new_transitions: list,
    replay_buffer: TransitionReplayBuffer,
    *,
    batch_size: int,
    new_fraction: float,
    rng: random.Random,
) -> list:
    n_new = max(1, int(batch_size * new_fraction)) if new_transitions else 0
    n_old = batch_size - n_new
    batch: list = []
    if new_transitions and n_new > 0:
        batch.extend(rng.choices(new_transitions, k=min(n_new, len(new_transitions))))
    if n_old > 0 and len(replay_buffer) > 0:
        batch.extend(replay_buffer.sample(n_old, rng=rng))
    return batch


def few_shot_adapt(
    model: BabaTransitionModel,
    quantizer: DynamicVectorQuantizer | None,
    new_token_ids: list[int],
    replay_buffer: TransitionReplayBuffer,
    new_transitions: list,
    *,
    config: OnlineExplorationConfig | None = None,
    device: str = "cpu",
) -> AdaptationResult:
    """
    Mixed-batch gradient steps: 20% new object, 80% replay (configurable).

    Prevents Phase 4 predictor from forgetting Walls/Rocks when binding new tokens.
    """
    cfg = config or OnlineExplorationConfig()
    rng = random.Random(0)
    model.to(device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.adapt_lr)

    batch_size = max(cfg.min_replay_for_adapt, 16)
    total_loss = 0.0
    steps_done = 0

    for _ in range(cfg.adapt_steps):
        batch = build_mixed_batch(
            new_transitions,
            replay_buffer,
            batch_size=batch_size,
            new_fraction=cfg.new_data_fraction,
            rng=rng,
        )
        if not batch:
            break
        step_loss = 0.0
        for tr in batch:
            opt.zero_grad()
            loss, _ = transition_loss(model, tr, quantizer, device=device)
            if loss.requires_grad:
                loss.backward()
                opt.step()
            step_loss += float(loss.item())
        total_loss += step_loss / len(batch)
        steps_done += 1

    model.eval()
    n_new_used = max(1, int(batch_size * cfg.new_data_fraction)) if new_transitions else 0
    n_replay = batch_size - n_new_used
    return AdaptationResult(
        new_token_ids=new_token_ids,
        sandbox_transitions=len(new_transitions),
        replay_samples=min(n_replay, len(replay_buffer)),
        new_samples=min(n_new_used, len(new_transitions)),
        steps=steps_done,
        mean_loss=total_loss / max(steps_done, 1),
    )


def run_online_adaptation_loop(
    model: BabaTransitionModel,
    quantizer: DynamicVectorQuantizer | None,
    map_name: str,
    replay_buffer: TransitionReplayBuffer,
    tracker: TokenExpansionTracker,
    *,
    config: OnlineExplorationConfig | None = None,
    device: str = "cpu",
) -> AdaptationResult | None:
    """
    Full Phase 5 loop: detect new tokens → sandbox → mixed-batch adapt.

    Call after VQ expansion or when `tracker.observe_tokens` returns novel IDs.
    """
    cfg = config or OnlineExplorationConfig()
    new_ids = sorted(tracker.known_tokens)[-1:] if tracker.known_tokens else []
    if not new_ids:
        return None

    target = set(new_ids[-max(1, len(new_ids)) :])
    sandbox_data = collect_sandbox_transitions(
        map_name,
        target,
        config=cfg,
        quantizer=quantizer,
    )
    if not sandbox_data:
        return None

    return few_shot_adapt(
        model,
        quantizer,
        list(target),
        replay_buffer,
        sandbox_data,
        config=cfg,
        device=device,
    )


def on_new_vq_tokens(
    model: BabaTransitionModel,
    quantizer: DynamicVectorQuantizer,
    map_name: str,
    replay_buffer: TransitionReplayBuffer,
    new_token_ids: list[int],
    *,
    config: OnlineExplorationConfig | None = None,
    device: str = "cpu",
) -> AdaptationResult:
    """
    Entry point when VQ-VAE adds codebook entries — freezes main loop, runs sandbox.
    """
    cfg = config or OnlineExplorationConfig()
    sandbox_data = collect_sandbox_transitions(
        map_name,
        set(new_token_ids),
        config=cfg,
        quantizer=quantizer,
    )
    return few_shot_adapt(
        model,
        quantizer,
        new_token_ids,
        replay_buffer,
        sandbox_data,
        config=cfg,
        device=device,
    )
