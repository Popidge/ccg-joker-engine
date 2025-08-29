# CC Group Joker Engine (Python, uv-native)

Lightweight ML/AI companion to the Rust solver (Triplecargo). Handles:
- Loading JSONL training data exported by Triplecargo
- Encoding board/hands/rules into tensors
- Training a small policy/value neural net (PyTorch)
- Evaluating accuracy and saving models
- CLI entry points for training and evaluation

Project follows uv-first workflow with src/ layout and CLI scripts.

## Quickstart

- Prereqs: Install uv (https://docs.astral.sh/uv/)

```bash
# Sync local virtual env (.venv) and install deps
uv sync

# Train (CPU baseline)
uv run ccj-train --data data/raw/train.jsonl --epochs 10 --batch-size 64 --lr 1e-3 --out data/models/model.pt

# Evaluate
uv run ccj-eval --data data/raw/val.jsonl --model data/models/model.pt
```

GPU users:
- This project pins CPU torch in pyproject. To use CUDA, you may install a CUDA wheel for torch compatible with your machine, e.g.:
  - uv pip install --upgrade "torch==<version>+cu118" -f https://download.pytorch.org/whl/torch/
- Keep uv.lock updated if you re-pin torch in this environment.

## CLI

The following entry points are provided:

- ccj-train → ccjoker.train:main
- ccj-eval  → ccjoker.eval:main

Usage (common flags):
- ccj-train
  - --data PATH (default: data/raw/train.jsonl)
  - --val-data PATH (optional; if not provided, --val-split is used)
  - --epochs INT (default: 10)
  - --batch-size INT (default: 64)
  - --lr FLOAT (default: 1e-3)
  - --out PATH (default: data/models/model.pt)
  - --device cpu|cuda (default: cpu)
  - --value-loss-weight FLOAT (default: 0.5)
  - --amp / --no-amp (default: AMP on for CUDA, off for CPU)
  - --amp-debug (optional; log GradScaler scale periodically when AMP enabled)
- ccj-eval
  - --data PATH (default: data/raw/val.jsonl) or use --data-pv and --data-mixed
  - --data-pv PATH (optional; when dataset lacks off_pv)
  - --data-mixed PATH (optional; requires --data-pv)
  - --model PATH (default: data/models/model.pt)
  - --batch-size INT (default: 128)
  - --num-workers INT (default: 0)
  - --device cpu|cuda (default: cpu)
  - --csv-out PATH (default: data/processed/eval_metrics.csv)

## Training

Policy loss
- Onehot samples: CrossEntropyLoss on masked policy logits [B,45] vs class indices [B].
- MCTS samples: KL divergence KL(p||q) = ∑ p log(p/q) where p is the target distribution [B,45], q is model softmax(logits) [B,45]. Implemented with numerical stability (normalize p, clamp q with epsilon).
- Mixed batches: compute CE on the onehot subset and KL on the MCTS subset, then combine by weighted average using the number of supervised samples (CE rows with label != -100, plus all KL rows). If the batch has only one type, compute only that loss.

Value loss
- Always CrossEntropyLoss on value logits [B,3] vs targets [B] (classes: loss=0, draw=1, win=2).

Total loss
- loss_total = loss_policy + λ * loss_value with default λ=0.5.
- Override with --value-loss-weight FLOAT when running ccj-train.

### AMP (Automatic Mixed Precision)

- Training uses PyTorch AMP for faster throughput and lower memory on CUDA.
- Default behavior:
  - CUDA: AMP enabled by default. Disable with --no-amp (or --amp=false if supported).
  - CPU: AMP is always disabled (forced). If requested, a warning is logged.
- Implementation details:
  - Forward/loss under torch.cuda.amp.autocast(enabled=amp)
  - Backward/step via torch.cuda.amp.GradScaler when AMP is enabled
  - Evaluation runs in FP32 for stable metrics
- Startup log:
  - [train] device=cuda amp=on
  - [train] device=cpu amp=off (forced)

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

## Evaluation

Metrics
- Policy:
  - top‑k accuracy: top‑1, top‑2, top‑3 computed over onehot samples (with valid labels) and MCTS samples (using argmax of the target distribution).
  - KL divergence: KL(p||q) averaged over MCTS samples only (p = target distribution, q = model softmax over masked logits).
- Value:
  - 3‑class accuracy (loss=0, draw=1, win=2).

Slicing
- If the dataset contains an off_pv boolean:
  - The evaluator reports three slices: PV (off_pv=false), off‑PV (off_pv=true), and ALL.
- If the dataset does not contain off_pv:
  - Provide two paths and the evaluator will treat them as slices:
    - --data-pv PATH  → PV slice
    - --data-mixed PATH → off‑PV/mixed slice
  - The tool also reports an aggregate over both.

CSV output
- Use --csv-out PATH to append metrics. Default: data/processed/eval_metrics.csv
- Columns: timestamp, model, data_tag, n_samples, top1, top2, top3, kl, value_acc
- When slices are present, one row is written per slice plus an aggregate row.

Examples
```bash
# Single file with off_pv in records; write CSV rows
uv run ccj-eval --data data/raw/val.jsonl --model data/models/model.pt --csv-out data/processed/eval_metrics.csv

# Two-dataset slicing when off_pv is absent
uv run ccj-eval --data-pv data/raw/val_pv.jsonl --data-mixed data/raw/val_mixed.jsonl --model data/models/model.pt --csv-out data/processed/eval_metrics.csv
```

## Data schema (Triplecargo JSONL)

Each line is a JSON object with the following fields:

```json
{
  "game_id": 0,
  "state_idx": 0,
  "board": [
    { "cell": 0, "card_id": 12, "owner": "A", "element": "F" },
    ...
  ],
  "hands": {
    "A": [10, 28, 47, 79, 91],
    "B": [19, 39, 65, 80, 94]
  },
  "to_move": "A",
  "turn": 0,
  "rules": {
    "elemental": true,
    "same": true,
    "plus": false,
    "same_wall": false
  },
  "off_pv": false,
  "policy_target": {
    // onehot → {"card_id": 19, "cell": 7}
    // mcts   → {"19-7": 0.5, "28-3": 0.5}
  },
  "value_target": 1,
  "value_mode": "winloss",
  "state_hash": "82d83106..."
}
```

Domains:
- off_pv: boolean, optional (default False if absent)
- value_mode=winloss → value_target ∈ {-1,0,+1}, from side-to-move perspective
- value_mode=margin → value_target ∈ [-9,+9], A-perspective; mapped to 3-class by sign
- policy_target (mixed formats supported per-sample):
  - onehot → single object {card_id:int, cell:int}
  - mcts → dict of "cardId-cell": float with probs ≥0, sum≈1.0 (normalized on load)
- element: "F","I","T","W","E","P","H","L" or null

Encoding summary:
- Board has 9 cells. For each cell we encode:
  - owner onehot [A, B, empty] → 3 dims
  - element onehot [F, I, T, W, E, P, H, L, none] → 9 dims
  - card embedding (padding for empty) with padding_idx=0
- Active hand uses the to_move side; 5 slots padded to fixed length
- Rules encoded as [elemental, same, plus, same_wall] → 4 dims
- Move space: 5 hand slots × 9 cells = 45 moves
  - Index mapping: idx = slot*9 + cell
- Targets:
  - Policy: class (0..44) or distribution [45]
  - Value: 3-class (loss=0, draw=1, win=2)

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
│   └── eval.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── models/
├── notebooks/
│   └── explore_data.ipynb
└── tests/
    ├── fixtures/sample.jsonl
    ├── test_dataset.py
    ├── test_model.py
    └── test_training.py
```

## Development workflow

- Use uv for all tasks:
  - uv sync
  - uv run pytest
  - uv run ruff check

Optional formatting:
- uv run ruff format

## Notes

- CPU torch is the default; see the GPU note above for CUDA wheels
- Do not commit large datasets. Place large JSONL files under data/raw
- Small sample/fixture JSONL lives under tests/fixtures