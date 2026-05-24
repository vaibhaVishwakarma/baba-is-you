"""Pre-training validation suite."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pyBaba

from baba_world.actions import ACTION_COUNT, decode_action
from baba_world.constants import (
    CHANNEL_HAS_TEXT,
    CHANNEL_IS_RULE,
    TENSOR_CHANNEL_TYPES,
    TENSOR_DIM_MAP,
    UNCODED_OBJECT_TYPES,
)
from baba_world.env import BabaTransitionEnv
from baba_world.generator import generate_episode
from baba_world.paths import map_path
from baba_world.rules import _cell_rule_string, extract_active_rules, rule_count_matches_manager
from baba_world.state import extract_state


@dataclass
class CheckResult:
    name: str
    passed: bool
    message: str = ""


@dataclass
class ValidationReport:
    map_name: str
    results: list[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results)

    def summary(self) -> str:
        lines = [f"Validation report: {self.map_name}", "-" * 40]
        for r in self.results:
            status = "PASS" if r.passed else "FAIL"
            lines.append(f"  [{status}] {r.name}")
            if r.message:
                lines.append(f"         {r.message}")
        lines.append("-" * 40)
        lines.append(f"Overall: {'PASS' if self.passed else 'FAIL'}")
        return "\n".join(lines)


def _replay_episode(map_name: str, actions: list[int]) -> list[np.ndarray]:
    game = pyBaba.Game(str(map_path(map_name)))
    game.Reset()
    states = [extract_state(game)]
    for action in actions:
        game.MovePlayer(decode_action(action))
        states.append(extract_state(game))
    return states


def check_determinism(
    map_name: str,
    *,
    num_rollouts: int = 5,
    steps_per_rollout: int = 100,
    base_seed: int = 42,
) -> CheckResult:
    """Same seed + action sequence must yield identical state trajectories."""
    import random

    for rollout in range(num_rollouts):
        rng = random.Random(base_seed + rollout)
        actions = [rng.randint(0, ACTION_COUNT - 1) for _ in range(steps_per_rollout)]

        traj_a = _replay_episode(map_name, actions)
        traj_b = _replay_episode(map_name, actions)

        for step, (sa, sb) in enumerate(zip(traj_a, traj_b)):
            if not np.array_equal(sa, sb):
                return CheckResult(
                    "determinism",
                    False,
                    f"Mismatch at step {step} (rollout {rollout})",
                )

    return CheckResult(
        "determinism",
        True,
        f"{num_rollouts} rollouts x {steps_per_rollout} steps matched",
    )


def _expected_rule_mask(game: pyBaba.Game) -> np.ndarray:
    """Cells participating in a valid rule triple (matches C++ isRule flags)."""
    m = game.GetMap()
    height = m.GetHeight()
    width = m.GetWidth()
    mask = np.zeros((height, width), dtype=np.float32)
    for y in range(height):
        for x in range(width):
            for dx, dy in ((1, 0), (0, 1)):
                if _cell_rule_string(game, x, y, dx, dy) is not None:
                    mask[y, x] = 1.0
                    mask[y + dy, x + dx] = 1.0
                    mask[y + 2 * dy, x + 2 * dx] = 1.0
    return mask


def check_tensor_semantics(map_name: str) -> CheckResult:
    """Tensor channels must match map cells for encoded object types."""
    game = pyBaba.Game(str(map_path(map_name)))
    game.Reset()
    tensor = extract_state(game)
    height = game.GetMap().GetHeight()
    width = game.GetMap().GetWidth()
    rule_mask = _expected_rule_mask(game)
    errors = 0
    checked = 0

    for y in range(height):
        for x in range(width):
            cell = game.GetMap().At(x, y)
            types = set(cell.GetTypes())

            expected_channels: set[int] = set()
            if types:
                for obj in types:
                    ch = TENSOR_DIM_MAP.get(obj)
                    if ch is None:
                        # Matches C++ std::map::operator[] defaulting to channel 0.
                        ch = 0
                    expected_channels.add(ch)

            for ch, obj_type in TENSOR_CHANNEL_TYPES.items():
                expected = 1.0 if ch in expected_channels else 0.0
                actual = float(tensor[ch, y, x])
                if actual != expected:
                    errors += 1
                checked += 1

            if types:
                has_text = any(pyBaba.IsTextType(t) for t in types)
                expected_text = 1.0 if has_text else 0.0
                if float(tensor[CHANNEL_HAS_TEXT, y, x]) != expected_text:
                    errors += 1

            if float(tensor[CHANNEL_IS_RULE, y, x]) != rule_mask[y, x]:
                errors += 1

    if errors:
        return CheckResult(
            "tensor_semantics",
            False,
            f"{errors} channel mismatches across {checked} object-channel checks",
        )
    uncoded_found: set[str] = set()
    for y in range(height):
        for x in range(width):
            types = set(game.GetMap().At(x, y).GetTypes())
            for t in types & set(UNCODED_OBJECT_TYPES):
                uncoded_found.add(str(t).split(".")[-1])

    msg = f"Verified {checked} object-channel cells on {width}x{height} grid"
    if uncoded_found:
        msg += f"; uncoded on map (omitted from tensor): {sorted(uncoded_found)}"

    return CheckResult("tensor_semantics", True, msg)


def check_rule_updates(map_name: str) -> CheckResult:
    """Rules must match RuleManager; simple_map loses all rules in one step."""
    game = pyBaba.Game(str(map_path(map_name)))
    game.Reset()
    rules = extract_active_rules(game)
    if not rule_count_matches_manager(game):
        return CheckResult(
            "rule_updates",
            False,
            f"Rule count mismatch: manager={game.GetRuleManager().GetNumRules()} "
            f"extracted={len(rules)}",
        )

    if map_name == "simple_map":
        before = extract_active_rules(game)
        game.MovePlayer(pyBaba.Direction.UP)
        after = extract_active_rules(game)
        if before and not after and game.GetPlayState() == pyBaba.PlayState.LOST:
            return CheckResult(
                "rule_updates",
                True,
                "simple_map: rules cleared on break-as-lose transition",
            )
        return CheckResult(
            "rule_updates",
            False,
            "simple_map golden case did not behave as expected",
        )

    if map_name == "baba_is_you":
        expected = {
            "BABA IS YOU",
            "FLAG IS WIN",
            "ROCK IS PUSH",
            "WALL IS STOP",
        }
        if set(rules) != expected:
            return CheckResult(
                "rule_updates",
                False,
                f"Initial rules mismatch: {set(rules)}",
            )
        return CheckResult(
            "rule_updates",
            True,
            f"Initial rules OK ({len(rules)} rules)",
        )

    return CheckResult(
        "rule_updates",
        True,
        f"Extracted {len(rules)} active rules",
    )


def check_transition_replay(map_name: str, *, num_checks: int = 20) -> CheckResult:
    """Env transitions must match independent simulator replay."""
    import random

    rng = random.Random(0)
    env = BabaTransitionEnv(map_name)
    failures = 0

    for _ in range(num_checks):
        env.reset()
        length = rng.randint(1, 40)
        actions: list[int] = []
        recorded_next: list[np.ndarray] = []

        state = env.clone_state()
        for _ in range(length):
            action = rng.randint(0, ACTION_COUNT - 1)
            sample = env.step(action)
            actions.append(action)
            recorded_next.append(sample.next_state.copy())
            if sample.done:
                break

        replayed = _replay_episode(map_name, actions)
        for i, recorded in enumerate(recorded_next):
            if not np.array_equal(recorded, replayed[i + 1]):
                failures += 1
                break

    if failures:
        return CheckResult(
            "transition_replay",
            False,
            f"{failures}/{num_checks} replay mismatches",
        )
    return CheckResult(
        "transition_replay",
        True,
        f"{num_checks} random segments replayed exactly",
    )


def check_env_wrapper(map_name: str) -> CheckResult:
    """BabaTransitionEnv reset/step contract."""
    env = BabaTransitionEnv(map_name)
    s0 = env.reset(episode_id=7)
    if s0.shape[0] != pyBaba.Preprocess.TENSOR_DIM:
        return CheckResult("env_wrapper", False, f"Bad state shape {s0.shape}")

    sample = env.step(0)
    if sample.action != 0 or sample.episode_id != 7 or sample.step_id != 0:
        return CheckResult("env_wrapper", False, "Metadata mismatch on first step")
    if sample.next_state.shape != s0.shape:
        return CheckResult("env_wrapper", False, "next_state shape mismatch")

    return CheckResult("env_wrapper", True, f"shape={tuple(s0.shape)}")


def run_all_validations(map_name: str = "baba_is_you") -> ValidationReport:
    """Run full pre-training validation suite for a map."""
    report = ValidationReport(map_name=map_name)
    report.results.append(check_env_wrapper(map_name))
    report.results.append(check_determinism(map_name))
    report.results.append(check_tensor_semantics(map_name))
    report.results.append(check_rule_updates(map_name))
    report.results.append(check_transition_replay(map_name))
    return report
