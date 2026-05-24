"""Phase 4: discrete next-token transition prediction."""

from baba_graph.predictor.agent import choose_action_lookahead, replay_and_choose
from baba_graph.predictor.align import align_nodes_by_token, align_nodes_hungarian
from baba_graph.predictor.config import PredictorConfig
from baba_graph.predictor.head import DualHeadOutput, DualTransitionPredictorHead, TransitionPredictorHead
from baba_graph.predictor.model import BabaTransitionModel, PredictorOutput
from baba_graph.predictor.movement import AdjacencyMove, NUM_ADJACENCY_CLASSES
from baba_graph.predictor.train import PredictorTrainStats, train_predictor, transition_loss, transition_token_loss

__all__ = [
    "AdjacencyMove",
    "BabaTransitionModel",
    "DualHeadOutput",
    "DualTransitionPredictorHead",
    "NUM_ADJACENCY_CLASSES",
    "PredictorConfig",
    "PredictorOutput",
    "PredictorTrainStats",
    "TransitionPredictorHead",
    "align_nodes_by_token",
    "align_nodes_hungarian",
    "choose_action_lookahead",
    "replay_and_choose",
    "train_predictor",
    "transition_loss",
    "transition_token_loss",
]
