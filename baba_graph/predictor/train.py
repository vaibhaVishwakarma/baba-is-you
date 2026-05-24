"""Phase 4 training: identity CE + movement CE on Hungarian-aligned pairs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

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


def transition_loss(
    model: BabaTransitionModel,
    transition: DynamicsTransition,
    quantizer: DynamicVectorQuantizer | None = None,
    *,
    device: str = "cpu",
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Dual loss on Hungarian-matched (t → t+1) node pairs.

    Returns (total_loss, metrics dict).
    """
    cb = model.world_config.codebook_size
    s = snapshot_to_tensors(
        transition.state,
        device=device,
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
        z = torch.tensor(0.0, device=device, requires_grad=True)
        return z, metrics

    out = model(s, transition.action)
    prev_idx = torch.tensor([p[0] for p in pairs], device=device, dtype=torch.long)
    next_idx = torch.tensor([p[1] for p in pairs], device=device, dtype=torch.long)

    id_logits = out.identity_logits[prev_idx]
    id_targets = torch.from_numpy(next_phys_tok[next_idx]).long().to(device)
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
        valid_t = torch.from_numpy(mov_valid).to(device)
        mov_targets = torch.from_numpy(mov_labels).long().to(device)
        loss_mov = F.cross_entropy(mov_logits[valid_t], mov_targets[valid_t])
        mov_pred = mov_logits[valid_t].argmax(dim=-1)
        metrics["movement_accuracy"] = float(
            (mov_pred == mov_targets[valid_t]).float().mean().item()
        )
    else:
        loss_mov = torch.tensor(0.0, device=device)

    w = model.predictor_config.movement_loss_weight
    total = loss_id + w * loss_mov
    metrics["loss_id"] = float(loss_id.item())
    metrics["loss_mov"] = float(loss_mov.item())
    return total, metrics


# Back-compat alias
transition_token_loss = transition_loss


def train_predictor(
    model: BabaTransitionModel,
    transitions: list[DynamicsTransition],
    quantizer: DynamicVectorQuantizer | None = None,
    *,
    epochs: int = 10,
    lr: float = 1e-3,
    device: str = "cpu",
) -> list[PredictorTrainStats]:
    model.to(device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history: list[PredictorTrainStats] = []

    for epoch in range(epochs):
        total_loss = 0.0
        id_acc = 0.0
        mov_acc = 0.0
        total_pairs = 0
        n_batch = 0
        for tr in transitions:
            opt.zero_grad()
            loss, m = transition_loss(model, tr, quantizer, device=device)
            n_pairs = int(m["num_pairs"])
            if n_pairs > 0 and loss.requires_grad:
                loss.backward()
                opt.step()
            total_loss += float(loss.item()) if n_pairs else 0.0
            id_acc += m["identity_accuracy"]
            mov_acc += m["movement_accuracy"]
            total_pairs += n_pairs
            n_batch += 1

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
