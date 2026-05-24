"""Phase 4 training: identity CE + movement CE on Hungarian-aligned pairs."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from baba_graph.device import action_tensor, is_cuda_device, module_device
from baba_graph.predictor.align import align_nodes_hungarian
from baba_graph.predictor.model import BabaTransitionModel
from baba_graph.predictor.movement import movement_labels_for_pairs
from baba_graph.vq.quantizer import DynamicVectorQuantizer
from baba_graph.world_model.data import DynamicsTransition
from baba_graph.world_model.model import snapshot_to_tensors
from baba_graph.world_model.tokens import snapshot_token_ids


@dataclass
class PredictorTrainStats:
    epoch: int
    loss: float
    identity_accuracy: float
    movement_accuracy: float
    num_pairs: int


def _autocast_ctx(device: str, use_amp: bool):
    if use_amp and is_cuda_device(device):
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _gather_token_ids(
    tokens: np.ndarray | torch.Tensor,
    indices: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Index token ids on ``device``; safe when ``indices`` is CUDA."""
    if isinstance(tokens, torch.Tensor):
        return tokens.to(device=device, dtype=torch.long)[indices]
    return torch.as_tensor(tokens, device=device, dtype=torch.long)[indices]


def transition_loss(
    model: BabaTransitionModel,
    transition: DynamicsTransition,
    quantizer: DynamicVectorQuantizer | None = None,
    *,
    device: str = "cpu",
    use_amp: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Dual loss on Hungarian-matched (t → t+1) node pairs."""
    dev = module_device(model)
    device_str = str(dev)
    cb = model.world_config.codebook_size
    s = snapshot_to_tensors(
        transition.state,
        device=device_str,
        codebook_size=cb,
        quantizer=quantizer,
    )
    _, next_phys_tok = snapshot_token_ids(
        transition.next_state,
        codebook_size=cb,
        quantizer=quantizer,
    )

    prev_tok = s["physical_token_ids"].detach().cpu().numpy()
    pairs = align_nodes_hungarian(
        transition.state.physical.nodes,
        prev_tok,
        transition.next_state.physical.nodes,
        next_phys_tok,
    )
    metrics = {
        "identity_accuracy": 0.0,
        "movement_accuracy": 0.0,
        "num_pairs": float(len(pairs)),
    }
    if not pairs:
        z = torch.tensor(0.0, device=dev, requires_grad=True)
        return z, metrics

    with _autocast_ctx(device_str, use_amp):
        out = model(s, action_tensor(transition.action, model))
        prev_idx = torch.tensor([p[0] for p in pairs], device=dev, dtype=torch.long)
        next_idx = torch.tensor([p[1] for p in pairs], device=dev, dtype=torch.long)

        id_logits = out.identity_logits[prev_idx]
        id_targets = _gather_token_ids(next_phys_tok, next_idx, dev)
        loss_id = F.cross_entropy(
            id_logits,
            id_targets,
            label_smoothing=model.predictor_config.label_smoothing,
        )
        id_pred = id_logits.argmax(dim=-1)
        metrics["identity_accuracy"] = float((id_pred == id_targets).float().mean().item())

        mov_labels, mov_valid = movement_labels_for_pairs(
            transition.state.physical.nodes,
            transition.next_state.physical.nodes,
            pairs,
        )
        mov_logits = out.movement_logits[prev_idx]
        if mov_valid.any():
            valid_t = torch.from_numpy(mov_valid).to(dev)
            mov_targets = torch.from_numpy(mov_labels).long().to(dev)
            loss_mov = F.cross_entropy(mov_logits[valid_t], mov_targets[valid_t])
            mov_pred = mov_logits[valid_t].argmax(dim=-1)
            metrics["movement_accuracy"] = float(
                (mov_pred == mov_targets[valid_t]).float().mean().item()
            )
        else:
            loss_mov = torch.zeros((), device=dev)

    w = model.predictor_config.movement_loss_weight
    total = loss_id + w * loss_mov
    metrics["loss_id"] = float(loss_id.item())
    metrics["loss_mov"] = float(loss_mov.item())
    return total, metrics


transition_token_loss = transition_loss


def train_predictor(
    model: BabaTransitionModel,
    transitions: list[DynamicsTransition],
    quantizer: DynamicVectorQuantizer | None = None,
    *,
    epochs: int = 10,
    lr: float = 1e-3,
    device: str = "cpu",
    use_amp: bool = False,
    grad_accum_steps: int = 1,
) -> list[PredictorTrainStats]:
    model.to(device)
    train_dev = module_device(model)
    train_dev_str = str(train_dev)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler("cuda") if use_amp and train_dev.type == "cuda" else None
    accum = max(1, grad_accum_steps)
    history: list[PredictorTrainStats] = []

    for epoch in range(epochs):
        total_loss = 0.0
        id_acc = 0.0
        mov_acc = 0.0
        total_pairs = 0
        n_batch = 0
        opt.zero_grad(set_to_none=True)
        accum_count = 0

        for tr in transitions:
            loss, m = transition_loss(
                model, tr, quantizer, device=train_dev_str, use_amp=use_amp
            )
            n_pairs = int(m["num_pairs"])
            if n_pairs > 0 and loss.requires_grad:
                scaled = loss / accum
                if scaler is not None:
                    scaler.scale(scaled).backward()
                else:
                    scaled.backward()
                accum_count += 1
                if accum_count >= accum:
                    if scaler is not None:
                        scaler.step(opt)
                        scaler.update()
                    else:
                        opt.step()
                    opt.zero_grad(set_to_none=True)
                    accum_count = 0

            total_loss += float(loss.item()) if n_pairs else 0.0
            id_acc += m["identity_accuracy"]
            mov_acc += m["movement_accuracy"]
            total_pairs += n_pairs
            n_batch += 1

        if accum_count > 0:
            if scaler is not None:
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()
            opt.zero_grad(set_to_none=True)

        history.append(
            PredictorTrainStats(
                epoch=epoch + 1,
                loss=total_loss / max(n_batch, 1),
                identity_accuracy=id_acc / max(n_batch, 1),
                movement_accuracy=mov_acc / max(n_batch, 1),
                num_pairs=total_pairs,
            )
        )

    model.eval()
    return history
