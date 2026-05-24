"""Apply VQ bottleneck to perception snapshots."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from baba_graph.perception.types import PerceptionSnapshot
from baba_graph.vq.quantizer import DynamicVectorQuantizer, VQOutput


@dataclass
class TokenizedSnapshot:
    """Phase 2 output: discrete tokens + quantized latents per modality."""

    perception: PerceptionSnapshot
    text_tokens: np.ndarray  # (N_text,) int64
    physical_tokens: np.ndarray  # (N_phys,) int64
    text_z_q: np.ndarray  # (N_text, D)
    physical_z_q: np.ndarray  # (N_phys, D)
    codebook_size: int = 0

    @property
    def num_text(self) -> int:
        return len(self.text_tokens)

    @property
    def num_physical(self) -> int:
        return len(self.physical_tokens)


def _quantize_group(
    visual: np.ndarray,
    quantizer: DynamicVectorQuantizer,
    *,
    train: bool = False,
) -> tuple[np.ndarray, np.ndarray, VQOutput | None]:
    if visual.size == 0:
        empty = np.zeros(0, dtype=np.int64)
        z_empty = np.zeros((0, quantizer.embedding_dim), dtype=np.float32)
        return empty, z_empty, None

    device = quantizer.embedding.device
    z_e = torch.from_numpy(visual).float().to(device)
    if train:
        out = quantizer(z_e)
        return (
            out.indices.cpu().numpy().astype(np.int64),
            out.z_q.cpu().numpy().astype(np.float32),
            out,
        )
    indices = quantizer.encode(z_e).cpu().numpy().astype(np.int64)
    z_q = quantizer.decode_indices(torch.from_numpy(indices).to(device))
    return indices, z_q.cpu().numpy().astype(np.float32), None


def tokenize_perception(
    snap: PerceptionSnapshot,
    quantizer: DynamicVectorQuantizer,
    *,
    train: bool = False,
) -> tuple[TokenizedSnapshot, list[VQOutput]]:
    """Quantize text and physical visual vectors."""
    losses: list[VQOutput] = []

    text_tok, text_zq, out_t = _quantize_group(
        snap.text.visual, quantizer, train=train
    )
    if out_t is not None:
        losses.append(out_t)

    phys_tok, phys_zq, out_p = _quantize_group(
        snap.physical.visual, quantizer, train=train
    )
    if out_p is not None:
        losses.append(out_p)

    return (
        TokenizedSnapshot(
            perception=snap,
            text_tokens=text_tok,
            physical_tokens=phys_tok,
            text_z_q=text_zq,
            physical_z_q=phys_zq,
            codebook_size=quantizer.num_codes,
        ),
        losses,
    )


def vq_loss_from_outputs(outputs: list[VQOutput]) -> dict[str, torch.Tensor]:
    if not outputs:
        zero = torch.tensor(0.0)
        return {
            "loss": zero,
            "commitment": zero,
            "codebook": zero,
            "perplexity": zero,
        }
    commitment = sum(o.commitment_loss for o in outputs) / len(outputs)
    codebook = sum(o.codebook_loss for o in outputs) / len(outputs)
    perplexity = sum(o.perplexity for o in outputs) / len(outputs)
    return {
        "loss": commitment + codebook,
        "commitment": commitment,
        "codebook": codebook,
        "perplexity": perplexity,
    }
