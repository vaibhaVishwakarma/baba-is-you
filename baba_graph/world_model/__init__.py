"""Phase 3: dual-graph world model with targeted symbolic binding."""

from baba_graph.world_model.batch import PaddedGraphBatch
from baba_graph.world_model.binding import (
    MultiPropertySlotAggregator,
    RuleBindingHead,
    gather_node_rule_context,
    parse_rule_chains,
)
from baba_graph.world_model.config import WorldModelConfig
from baba_graph.world_model.data import DynamicsTransition, collect_dynamics_transitions
from baba_graph.world_model.model import (
    DualGraphWorldModel,
    PhysicalGraphMPNN,
    RuleGraphEncoder,
    WorldModelOutput,
    snapshot_to_tensors,
)
from baba_graph.world_model.tokens import snapshot_token_ids
from baba_graph.world_model.train import train_representation_warmup

__all__ = [
    "DualGraphWorldModel",
    "DynamicsTransition",
    "PaddedGraphBatch",
    "PhysicalGraphMPNN",
    "MultiPropertySlotAggregator",
    "RuleBindingHead",
    "RuleGraphEncoder",
    "WorldModelConfig",
    "WorldModelOutput",
    "collect_dynamics_transitions",
    "gather_node_rule_context",
    "parse_rule_chains",
    "snapshot_to_tensors",
    "snapshot_token_ids",
    "train_representation_warmup",
]
