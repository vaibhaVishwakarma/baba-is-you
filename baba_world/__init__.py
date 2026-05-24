"""World-transition-centric interface for Baba Is You (pre-JEPA data pipeline)."""

from baba_world.actions import ACTION_COUNT, ACTION_TO_DIRECTION, decode_action, encode_action
from baba_world.dataset import TransitionDataset, load_dataset, save_dataset
from baba_world.env import BabaTransitionEnv
from baba_world.generator import generate_episode, generate_episodes
from baba_world.state import StateExtractor, extract_state
from baba_world.types import Episode, Transition, TransitionSample
from baba_world.validation import ValidationReport, run_all_validations

__all__ = [
    "ACTION_COUNT",
    "ACTION_TO_DIRECTION",
    "BabaTransitionEnv",
    "Episode",
    "StateExtractor",
    "Transition",
    "TransitionDataset",
    "TransitionSample",
    "ValidationReport",
    "decode_action",
    "encode_action",
    "extract_state",
    "generate_episode",
    "generate_episodes",
    "load_dataset",
    "run_all_validations",
    "save_dataset",
]
