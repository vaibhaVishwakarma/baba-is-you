"""Phase 2: VQ-VAE discrete token bottleneck."""

from baba_graph.vq.config import VQConfig
from baba_graph.vq.quantizer import DynamicVectorQuantizer, VQOutput
from baba_graph.vq.tokenize import TokenizedSnapshot, tokenize_perception, vq_loss_from_outputs
from baba_graph.vq.train import collect_and_train, load_quantizer, save_quantizer

__all__ = [
    "DynamicVectorQuantizer",
    "TokenizedSnapshot",
    "VQConfig",
    "VQOutput",
    "collect_and_train",
    "load_quantizer",
    "save_quantizer",
    "tokenize_perception",
    "vq_loss_from_outputs",
]
