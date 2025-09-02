# CC Group Joker Engine (Python, uv-native)

Lightweight ML/AI companion to the Rust solver (Triplecargo). It provides:
- Loading JSONL training data exported by Triplecargo
- Encoding board/hands/rules into tensors
- Training a small policy/value neural net (PyTorch)
- Evaluating accuracy and writing CSV metrics
- Self-play generation, RL fine-tuning, and gating

Key sources:
- Project config: [pyproject.toml](pyproject.toml)
- Training: [src/ccjoker/train.py](src/ccjoker/train.py) ([python.main()](src/ccjoker/train.py:390))
- Evaluation: [src/ccjoker/eval.py](src/ccjoker/eval.py) ([python.main()](src/ccjoker/eval.py:284))
- Self-play: [src/ccjoker/selfplay.py](src/ccjoker/selfplay.py) ([python.main()](src/ccjoker/selfplay.py:86))
- RL training: [src/ccjoker/train_rl.py](src/ccjoker/train_rl.py) ([python.main()](src/ccjoker/train_rl.py:13))
- Gating: [src/ccjoker/gate.py](src/ccjoker/gate.py) ([python.main()](src/ccjoker/gate.py:57))

## Prerequisites

- Install uv: https://docs.astral.sh/uv/
- Python 3.12 (workspace-pinned; uv will create .venv)

## Quickstart

```bash
# Create/sync the local virtual environment (.venv) and install deps
uv sync

# Train (CPU example)
uv run ccj-train --data data/raw/train.jsonl --epochs 10 --batch-size 64 --lr 1e-3 --out data/models/model.pt

# Evaluate (writes CSV metrics)
uv run ccj-eval --data data/raw/val.jsonl --model data/models/model.pt --csv-out data/processed/eval_metrics.csv

# Generate self-play and train RL
uv run ccj-selfplay --model data/models/model.pt --games 100 --out data/raw/selfplay_100.jsonl
uv run ccj-train-rl --data data/raw/selfplay_100.jsonl --epochs 5 --batch-size 256 --out data/models/model_rl.pt

# Gate: compare two models head-to-head
uv run ccj-gate --a data/models/model.pt --b data/models/model_rl.pt --games 200
```

### GPU/CPU notes

- This repository pins a CUDA 12.1 PyTorch wheel in [pyproject.toml](pyproject.toml). On GPU machines, this installs torch==2.3.1+cu121 via the configured PyTorch index.
- On CPU-only machines, you can switch to a CPU wheel by editing [pyproject.toml](pyproject.toml) to remove the custom source and changing the torch version accordingly, then:
  - Recreate the lock/env: `uv lock && uv sync`

## CLI reference

The package exposes Typer apps as uv scripts (see [project.scripts](pyproject.toml)). Each command below lists flags, defaults, and behavior.

### ccj-train — Supervised training
Source: [src/ccjoker/train.py](src/ccjoker/train.py) ([python.main()](src/ccjoker/train.py:390))

Trains the policy/value network on JSONL exported by Triplecargo.

Arguments:
- --data PATH (Path, default: data/raw/train.jsonl)
- --val-data PATH (Path, optional; if omitted, uses --val-split on --data)
- --epochs INT (default: 10, min: 1)
- --batch-size INT (default: 64, min: 1)
- --lr FLOAT (default: 1e-3)
- --weight-decay FLOAT (default: 0.0)
- --out PATH (Path, default: data/models/model.pt)
- --val-split FLOAT (default: 0.1, range: 0.01–0.5) used only when --val-data is not provided
- --num-workers INT (default: 0)
- --seed INT (default: 42)
- --device cpu|cuda (default: cpu)
- --value-loss-weight FLOAT (default: 0.5) weight λ in loss_total = loss_policy + λ * loss_value
- --amp / --no-amp (default: on for CUDA, off for CPU) enable/disable mixed precision
- --amp-debug (flag) log GradScaler scale periodically when AMP is enabled

Training details:
- Policy: mixed loss supporting both onehot labels and MCTS distributions. See [python.compute_mixed_policy_loss()](src/ccjoker/train.py:66).
- Value: CrossEntropy over 3 classes (loss=0, draw=1, win=2). See [python.value_loss()](src/ccjoker/train.py:161).
- Total: loss_total = policy + λ * value (λ controlled by --value-loss-weight).
- AMP: CUDA defaults to AMP on; CPU forces AMP off. Startup logs indicate the effective setting. See [python.train_loop()](src/ccjoker/train.py:277).

Example (CUDA, AMP on):
```bash
uv run ccj-train \
  --data data/raw/train_5k.jsonl \
  --val-data data/raw/val_1k.jsonl \
  --epochs 10 \
  --batch-size 512 \
  --lr 1e-3 \
  --out data/models/model_amp.pt \
  --device cuda \
  --amp
```

### ccj-eval — Metrics and CSV export
Source: [src/ccjoker/eval.py](src/ccjoker/eval.py) ([python.main()](src/ccjoker/eval.py:284))

Evaluates a trained checkpoint with per-slice metrics and appends rows to CSV.

Arguments:
- --data PATH (Path, default: data/raw/val.jsonl) single-dataset mode; if off_pv field exists, slices are inferred
- --data-pv PATH (Path, optional) PV dataset when records lack off_pv
- --data-mixed PATH (Path, optional) off‑PV/mixed dataset; requires --data-pv
- --model PATH (Path, default: data/models/model.pt)
- --batch-size INT (default: 128, min: 1)
- --num-workers INT (default: 0)
- --device cpu|cuda (default: cpu)
- --csv-out PATH (Path, default: data/processed/eval_metrics.csv)

Metrics:
- Policy: top‑1/top‑2/top‑3 accuracy on supervised rows; KL(p||q) on MCTS rows.
- Value: 3‑class accuracy.
- Slicing: 'pv', 'off_pv', and 'all' either inferred from off_pv or constructed from the two-dataset mode.
- CSV columns: timestamp, model, data_tag, n_samples, top1, top2, top3, kl, value_acc.

Example:
```bash
# Single file with off_pv; append slice + aggregate rows
uv run ccj-eval --data data/raw/val.jsonl --model data/models/model.pt --csv-out data/processed/eval_metrics.csv

# Two-dataset mode
uv run ccj-eval --data-pv data/raw/val_pv.jsonl --data-mixed data/raw/val_mixed.jsonl --model data/models/model.pt --csv-out data/processed/eval_metrics.csv
```

### ccj-selfplay — Generate trajectories (AlphaZero‑lite)
Source: [src/ccjoker/selfplay.py](src/ccjoker/selfplay.py) ([python.main()](src/ccjoker/selfplay.py:86))

Generates self-play games using MCTS guided by the current network and appends JSONL trajectories.

Arguments:
- --model PATH (Path, required) checkpoint guiding self‑play (.pt from ccj‑train)
- --games INT (default: 1)
- --out PATH (Path, default: data/raw/selfplay.jsonl) output JSONL to append
- --rollouts INT (default: 64, min: 0)
- --temperature FLOAT (default: 1.0) early‑game temperature for turns < --sample-until
- --dirichlet-alpha FLOAT (default: 0.3) root Dirichlet alpha over legal moves
- --dirichlet-eps FLOAT (default: 0.25) root noise epsilon for turns ≥ --sample-until
- --sample-until INT (default: 6, min: 0) sample from pi until this turn (exclusive), then switch to late settings
- --early-dirichlet-eps FLOAT (default: 0.5) root noise epsilon for turns < --sample-until
- --late-temperature FLOAT (default: 0.0) temperature for turns ≥ --sample-until (0.0 → argmax)
- --seed INT (optional)
- --device cpu|cuda (default: cpu)
- --rules STR (default: "none") comma‑separated: elemental,same,plus,same_wall
- --triplecargo-cmd PATH (optional) path to Triplecargo precompute.exe with --eval-state
- --cards PATH (optional) cards.json path used by both engines
- --use-stub / --no-use-stub (default: false) use deterministic Python stub for CI
- --verbose (flag) detailed per‑game/per‑turn logging
- --debug-ipc (flag) log raw JSON IPC to/from Triplecargo
- --workers INT (default: 1; cpu‑only) number of CPU worker processes; each worker spawns its own Triplecargo --eval-state
- --torch-threads INT (default: 1; cpu‑only) torch.set_num_threads per worker to avoid oversubscription
- --keep-shards / --no-keep-shards (default: false) keep per‑worker shard files instead of merging then deleting

Exploration:
- Root prior is mixed with Dirichlet noise: P' = (1 - eps) * P + eps * Dir(alpha), masked to legal moves and renormalized.
- Two-phase schedule: early (turn < --sample-until) uses --temperature and --early-dirichlet-eps; late uses --late-temperature and --dirichlet-eps.

Progress:
- When --verbose is not passed, a progress bar updates once per completed game (after the JSONL line with state_idx=8 is written) with counts:
  - played = total games completed
  - A = A wins
  - B = B wins
  - D = draws

CPU parallelism (cpu device only):
- Set --workers > 1 to parallelize across CPU processes. Each worker:
  - Loads the model on CPU and creates its own Triplecargo --eval-state subprocess
  - Plays its assigned share of games and writes to a shard file: {--out}.w{worker_id}.jsonl
- The parent process aggregates per‑game progress and merges shards into --out, deleting shards by default.
  - Keep shards by passing --keep-shards.
  - Control torch intra‑op threads per worker with --torch-threads (default 1) to avoid oversubscription.

Example:
```bash
uv run ccj-selfplay \
  --model data/models/mix_5k.pt \
  --games 1000 \
  --rollouts 64 \
  --temperature 1.0 \
  --dirichlet-alpha 0.3 \
  --early-dirichlet-eps 0.5 \
  --dirichlet-eps 0.25
```

CPU parallel example (12 workers on a 12‑core CPU):
```bash
uv run ccj-selfplay \
  --model data/models/mix_5k.pt \
  --games 2400 \
  --rollouts 64 \
  --device cpu \
  --workers 12 \
  --torch-threads 1 \
  --out data/raw/selfplay_2400.jsonl
# Shards will be merged into the --out file and then removed by default.
```

First-player alternation (bias mitigation)
- To reduce start-side bias during self-play (which can cause a strong model to overfit to "A starts"), ccj-selfplay now alternates the starting side deterministically:
  - Single-process mode: game g uses first = "A" when g is even, and "B" when g is odd.
  - Multi-worker mode: a global start offset is partitioned across worker shards so alternation is preserved across workers and merged output.
- The implementation uses the same swap helper as gating to flip hands/owners/to_move when necessary. See implementation notes in: [`src/ccjoker/selfplay.py`](src/ccjoker/selfplay.py:445) and the swap helper in [`src/ccjoker/gate.py`](src/ccjoker/gate.py:391).
- Validation:
  - Single-process: run a small self-play job and inspect the first record of each game in the output JSONL to confirm the "to_move" value alternates between "A" and "B".
  - Multi-worker: run with --workers > 1 and inspect the merged output for global alternation across games.
- Optional stronger control:
  - If you want each sampled initial state played both ways (mirror pairs: same hands/board but A-start and B-start), that can be done externally (orchestration script that re-runs or post-processes swapped states) or by adding an in-process `--mirror-initials` flag. The current change performs deterministic alternation only (no automatic mirrored-pair duplication).

### ccj-train-rl — Train from self-play
Source: [src/ccjoker/train_rl.py](src/ccjoker/train_rl.py) ([python.main()](src/ccjoker/train_rl.py:13))

Uses the same training loop as ccj-train but with defaults suited for self-play data.

Arguments:
- --data PATH (Path, default: data/raw/selfplay.jsonl)
- --val-data PATH (Path, optional)
- --epochs INT (default: 5, min: 1)
- --batch-size INT (default: 256, min: 1)
- --lr FLOAT (default: 1e-3)
- --weight-decay FLOAT (default: 0.0)
- --out PATH (Path, default: data/models/model_rl.pt)
- --val-split FLOAT (default: 0.1, range: 0.01–0.5) used only when --val-data is not provided
- --num-workers INT (default: 0)
- --seed INT (default: 42)
- --device cpu|cuda (default: cpu)
- --value-loss-weight FLOAT (default: 0.5)
- --amp / --no-amp (default: on for CUDA, off for CPU)
- --amp-debug (flag)

### ccj-gate — Promotion gating
Source: [src/ccjoker/gate.py](src/ccjoker/gate.py) ([python.main()](src/ccjoker/gate.py:57))

Plays A vs B and prints a JSON summary with W/D/L, score of B, Elo delta, and a boolean 'promote'.

Arguments:
- --a PATH (Path, required) baseline/old model checkpoint
- --b PATH (Path, required) candidate/new model checkpoint
- --games INT (default: 20)
- --device cpu|cuda (default: cpu)
- --rollouts INT (default: 0, min: 0) 0 → greedy argmax; >0 → MCTS rollouts per move
- --temperature FLOAT (default: 0.25) MCTS sampling temperature
- --seed INT (optional; default: 123)
- --rules STR (default: "none")
- --triplecargo-cmd PATH (optional)
- --cards PATH (optional)
- --use-stub / --no-use-stub (default: false)
- --threshold FLOAT (default: 0.55, range: 0.5–1.0) promotion threshold as score vs A
- --workers INT (default: 1; cpu‑only) number of CPU worker processes to parallelize games
- --torch-threads INT (default: 1; cpu‑only) torch.set_num_threads per worker to avoid oversubscription

Notes:
- Greedy mode uses masked argmax over policy logits. See [python.select_move_greedy()](src/ccjoker/gate.py:32).
- Elo mapping uses d = 400 * log10(s/(1-s)). See [python.elo_delta_from_score()](src/ccjoker/gate.py:45).

CPU parallelism (gate):
- Set --workers > 1 with --device cpu to distribute games across worker processes. Each worker:
  - Loads models A and B on CPU and creates its own Triplecargo --eval-state subprocess
  - Plays its assigned share of games and reports results to the parent
- A single progress bar aggregates W/D/L across workers (stdout remains pure JSON at the end).
- Tune --workers and --torch-threads similar to ccj-selfplay recommendations.

## Data schema (Triplecargo JSONL)

Each line is a JSON object with these fields:

```json
{
  "game_id": 0,
  "state_idx": 0,
  "board": [
    { "cell": 0, "card_id": 12, "owner": "A", "element": "F" }
  ],
  "hands": { "A": [10, 28, 47, 79, 91], "B": [19, 39, 65, 80, 94] },
  "to_move": "A",
  "turn": 0,
  "rules": { "elemental": true, "same": true, "plus": false, "same_wall": false },
  "off_pv": false,
  "policy_target": { "19-7": 0.5, "28-3": 0.5 },
  "value_target": 1,
  "value_mode": "winloss",
  "state_hash": "82d83106..."
}
```

Domains and encoding summary:
- off_pv: boolean, optional (default false if absent)
- value_mode=winloss → value_target ∈ {-1,0,+1} from side-to-move perspective
- value_mode=margin → value_target ∈ [-9,+9], A-perspective; mapped to 3‑class by sign
- policy_target supports either:
  - onehot → {card_id:int, cell:int}
  - mcts → dict of "cardId-cell": float with probs ≥0, sum≈1.0 (normalized on load)
- element: "F","I","T","W","E","P","H","L" or null
- Move space: 5 hand slots × 9 cells = 45 moves; idx = slot*9 + cell
- Targets: policy class (0..44) or distribution [45]; value 3‑class (loss=0, draw=1, win=2)

## Project layout

```
cc-joker-engine/
├── pyproject.toml
├── README.md
├── .python-version
├── src/ccjoker/
│   ├── __init__.py
│   ├── utils.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── eval.py
│   ├── env.py           # TripleTriadEnv wrapper (Triplecargo CLI or stub)
│   ├── mcts.py          # AlphaZero-lite MCTS guided by Joker net
│   ├── selfplay.py      # ccj-selfplay CLI
│   ├── train_rl.py      # ccj-train-rl CLI
│   ├── gate.py          # ccj-gate CLI
│   └── checkpoint.py    # model checkpoint loader utility
├── data/
│   ├── raw/
│   ├── processed/
│   └── models/
├── notebooks/
│   └── explore_data.ipynb
└── tests/
    ├── fixtures/sample.jsonl
    ├── fixtures/mixed.jsonl
    ├── test_dataset.py
    ├── test_model.py
    ├── test_training.py
    ├── test_eval.py
    ├── test_selfplay.py
    ├── test_train_rl.py
    └── test_gate.py
```

## Development workflow

- Sync env: `uv sync`
- Run tests: `uv run pytest`
- Lint/format: `uv run ruff check` and `uv run ruff format`

Data handling:
- Do not commit large datasets. Place large JSONL files under data/raw/.
- Small sample/fixture JSONL lives under tests/fixtures/.

## References

- Triplecargo (Rust solver): https://github.com/Popidge/triplecargo
- Typer docs: https://typer.tiangolo.com/
- PyTorch AMP: https://pytorch.org/docs/stable/amp.html