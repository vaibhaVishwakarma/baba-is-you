# baba_world — World-transition pipeline

Pre-JEPA data layer: **structured causal transitions** only (no pygame, Gym, DQN, or rewards).

## Quick start

From repository root (requires `pip install -U .` for `pyBaba`):

```bash
pip install numpy pytest
python scripts/validate_transitions.py --map baba_is_you
python scripts/generate_dataset.py --map baba_is_you --episodes 1000 --output data/transitions/baba_is_you.npz
pytest tests/test_baba_world.py -v
```

## Architecture

```
BabaTransitionEnv  →  reset() / step(action: int)
StateExtractor     →  (16, H, W) tensor
generate_episode() →  list[TransitionSample]
save_dataset()     →  .npz + .meta.json
```

## Action encoding (frozen)

| int | direction |
|-----|-----------|
| 0   | UP        |
| 1   | DOWN      |
| 2   | LEFT      |
| 3   | RIGHT     |

## Canonical sample

```python
from baba_world import BabaTransitionEnv

env = BabaTransitionEnv("baba_is_you")
state = env.reset()
sample = env.step(0)  # sample.state, sample.action, sample.next_state, sample.done
```

## Validation

`run_all_validations(map_name)` checks determinism, tensor semantics, rules, and replay consistency.
