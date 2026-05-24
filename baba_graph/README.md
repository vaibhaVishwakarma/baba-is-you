# baba_graph — Phase 1 V3: Perception Split

**Philosophy:** Baba is dynamic ontology mutation — Phase 1 must **separate reading rules (text) from physical objects** before any physics model runs.

## Outputs (`PerceptionSnapshot`)

| Stream | Contents | Edges |
|--------|----------|-------|
| **Text** (`snap.text`) | Visual (128) + position — rule parsing view | **Directed** → and ↓ only (L→R, T→B) |
| **Physical** (`snap.physical`) | Icons **+ text blocks** with PUSH + properties | Overlap + 4-neighbor spatial (collision) |

**Double life:** every text block appears in `snap.text` (for the Rule Graph in Phase 3) and is **mirrored** into `snap.physical` with default **PUSH** so the agent can push words.

No hardcoded object IDs. Unknown types → procedural patch → frozen encoder → novel continuous vector.

## API

```python
from baba_graph import extract_perception_from_game

snap = extract_perception_from_game(game)

snap.text.visual      # (N_text, 128)  — continuous tokens-in-waiting
snap.physical.visual  # (N_phys, 128)
snap.text.x           # visual + normalized (x,y)
snap.physical.x       # visual + pos + properties (YOU, STOP, …)

snap.classifier_accuracy  # text vs physical head vs simulator
```

## Components

1. **PatchRenderer** — sprite GIF or procedural open-vocab patch  
2. **PerceptionEncoder** — frozen CNN/ResNet + **binary text/physical head** (calibrated on simulator labels)  
3. **Split** — ground-truth `IsTextType` for lists; head for diagnostics / pixel-only future path  

## Validation

```bash
python scripts/validate_graphs.py --map baba_is_you
pytest tests/test_baba_graph.py tests/test_baba_perception.py -v
```

## Phase 2 — VQ bottleneck (`baba_graph/vq/`)

Discretizes `text.visual` and `physical.visual` into codebook token IDs.

```python
from baba_graph import tokenize_perception, load_quantizer

q = load_quantizer("data/vq/codebook.pt")
tok, _ = tokenize_perception(snap, q)
tok.text_tokens      # (N_text,) int64
tok.physical_tokens  # (N_phys,) int64
```

Train codebook:

```bash
python scripts/train_vq.py --maps baba_is_you --epochs 30 --output data/vq/codebook.pt
```

EMA codebook, commitment loss, STE, dynamic expansion for novel objects.

## Performance: visual embedding cache

Patches are static per object type. `VisualEmbeddingCache` hashes patch pixels and
reuses 128-d vectors — encoding cost drops to ~0 after the first frame.

## Data collection: biased exploration

`BiasedTextExplorer` targets text blocks and rewards rule/adjacency changes
(used by `train_vq.py` and world-model rollouts by default).

```bash
python scripts/train_vq.py --policy biased
python scripts/train_world_model.py --map baba_is_you
```

## Phase 3 — Dual-graph world model (`baba_graph/world_model/`)

**Targeted symbolic binding** (no global rule soup):

| Module | Role |
|--------|------|
| `RuleGraphEncoder` | Text MPNN → `rule_embeddings` **(codebook_size, H)** keyed by VQ token |
| `RuleBindingHead` | Parses NOUN-IS-PROPERTY chains; writes PUSH into ROCK slot, etc. |
| `PhysicalGraphMPNN` | Each node fetches `rule_embeddings[physical_token_id]` only |
| `PaddedGraphBatch` | Padding + `node_mask` so ghost nodes do not message (Phase 4 batching) |

```python
from baba_graph.world_model import DualGraphWorldModel, snapshot_to_tensors

tensors = snapshot_to_tensors(snap, quantizer=q)  # includes token ids
out = model.from_snapshot_tensors(tensors, action=0)
out.rule_embeddings      # (K, H) token-indexed rule table
out.node_rule_context    # (N_phys, H) per-object rule context
out.physical_h           # (N_phys, H) for Phase 4 token head
```

## Phase 4 — Dual-headed transition prediction (`baba_graph/predictor/`)

| Head | Output | Loss |
|------|--------|------|
| **Identity** | `(N, codebook_size)` | CE on next VQ token |
| **Movement** | `(N, 5)` STAY/UP/DOWN/LEFT/RIGHT | CE on grid adjacency |

Training alignment: **Hungarian algorithm** (scipy) on L2 position + token penalty — 1-to-1, no greedy cross-wire.

```python
out = model(snapshot_tensors, action=0)
out.identity_logits   # Head A
out.movement_logits   # Head B
```

Train: `python scripts/train_predictor.py --map baba_is_you`

GUI play:

```bash
pip install pygame scipy
python scripts/play_gui.py --map baba_is_you --mode human
python scripts/play_gui.py --map baba_is_you --mode agent
```

**Multi-property binding:** `RuleBindingHead` uses `MultiPropertySlotAggregator` (attention / scatter_max / sum) so `BABA IS YOU` + `BABA IS WEAK` compose into one slot — no last-write overwrite.

## Phase 5 — Online exploration (`baba_graph/exploration/online/`)

When VQ adds a new token: sandbox rollouts + **mixed-batch** adaptation (20% new / 80% replay) so Phase 4 does not forget known objects.

```bash
python scripts/run_online_adapt.py --map baba_is_you --new-tokens 500 501
```

GUI agent uses **YOU/WIN property** positions from the physical graph (not hardcoded FLAG).

## Pipeline

```
Phase 1  →  Phase 2 VQ  →  Phase 3  →  Phase 4  →  Phase 5 adapt  →  Phase 6 planner
```

Legacy `extract_graph_from_game()` still works (merges text+physical for old code).
