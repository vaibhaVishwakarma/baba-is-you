"""Tests for world-transition pipeline (requires built pyBaba)."""

from __future__ import annotations

import numpy as np
import pytest

pyBaba = pytest.importorskip("pyBaba")

from baba_world import (
    BabaTransitionEnv,
    generate_episode,
    generate_episodes,
    load_dataset,
    run_all_validations,
    save_dataset,
)
from baba_world.actions import ACTION_COUNT, decode_action, encode_action
from baba_world.constants import TENSOR_DIM
from baba_world.dataset import episodes_to_dataset
from baba_world.generator import episodes_to_samples
from baba_world.paths import map_path
from baba_world.state import extract_state


MAP = "baba_is_you"


def test_action_codec_roundtrip():
    for action in range(ACTION_COUNT):
        assert encode_action(decode_action(action)) == action


def test_state_shape():
    env = BabaTransitionEnv(MAP)
    state = env.reset()
    assert state.shape == (TENSOR_DIM, 9, 11)


def test_step_returns_transition_sample():
    env = BabaTransitionEnv(MAP)
    env.reset(episode_id=3)
    sample = env.step(0)
    assert sample.action == 0
    assert sample.episode_id == 3
    assert sample.step_id == 0
    assert sample.state.shape == sample.next_state.shape


def test_generate_episode_non_empty():
    ep = generate_episode(MAP, max_steps=10, seed=1)
    assert len(ep.transitions) >= 1
    assert ep.transitions[0].state.dtype == np.float32


def test_dataset_roundtrip(tmp_path):
    episodes = generate_episodes(MAP, num_episodes=3, max_steps=15, seed=0)
    dataset = episodes_to_dataset(episodes)
    out = tmp_path / "transitions"
    save_dataset(dataset, out)
    loaded = load_dataset(out)
    assert loaded.num_transitions == dataset.num_transitions
    np.testing.assert_array_equal(loaded.state, dataset.state)
    np.testing.assert_array_equal(loaded.action, dataset.action)


def test_full_validation_baba_is_you():
    report = run_all_validations(MAP)
    assert report.passed, report.summary()


def test_full_validation_simple_map():
    report = run_all_validations("simple_map")
    assert report.passed, report.summary()


def test_map_path_resolves():
    p = map_path("baba_is_you")
    assert p.name == "baba_is_you.txt"
    assert p.is_file()
