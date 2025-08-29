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
- ccj-eval
  - --data PATH (default: data/raw/val.jsonl)
  - --model PATH (default: data/models/model.pt)
  - --batch-size INT (default: 128)
  - --device cpu|cuda (default: cpu)

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