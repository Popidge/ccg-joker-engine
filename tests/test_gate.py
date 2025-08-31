from __future__ import annotations

import json
from pathlib import Path

import torch
from typer.testing import CliRunner

from ccjoker.gate import app as gate_app
from ccjoker.dataset import TriplecargoDataset
from ccjoker.model import ModelConfig, PolicyValueNet


def _save_tiny_checkpoint(path: Path, num_cards: int, seed: int) -> None:
    torch.manual_seed(seed)
    cfg = ModelConfig(num_cards=num_cards, card_embed_dim=8, hidden_dim=32, dropout=0.0)
    model = PolicyValueNet(cfg)
    # random init already seeded; save state
    payload = {
        "model_state_dict": model.state_dict(),
        "config": {
            "num_cards": cfg.num_cards,
            "card_embed_dim": cfg.card_embed_dim,
            "hidden_dim": cfg.hidden_dim,
            "dropout": cfg.dropout,
        },
        "meta": {"epoch": 0, "num_cards": num_cards, "move_space": 45},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def test_gate_cli_smoke(tmp_path: Path):
    # Build two tiny checkpoints with different seeds
    ds = TriplecargoDataset("tests/fixtures/sample.jsonl")
    num_cards = ds.max_card_id + 2
    a_path = tmp_path / "a.pt"
    b_path = tmp_path / "b.pt"
    _save_tiny_checkpoint(a_path, num_cards, seed=1)
    _save_tiny_checkpoint(b_path, num_cards, seed=2)

    # Run gate with stub env and greedy selection (rollouts=0) for speed
    runner = CliRunner()
    res = runner.invoke(
        gate_app,
        [
            "--a",
            str(a_path),
            "--b",
            str(b_path),
            "--games",
            "6",
            "--device",
            "cpu",
            "--rollouts",
            "0",
            "--temperature",
            "0.25",
            "--rules",
            "none",
            "--use-stub",
            "--seed",
            "123",
        ],
    )
    assert res.exit_code == 0, f"ccj-gate failed: {res.output}"
    # Output is pure JSON on stdout
    payload = json.loads(res.stdout.strip())
    assert payload["games"] == 6
    results = payload["results"]
    total = int(results["a_wins"]) + int(results["b_wins"]) + int(results["draws"])
    assert total == 6
    # score_b in [0,1]
    s = float(payload["score_b"])
    assert 0.0 <= s <= 1.0
    # elo_delta is finite
    elo = float(payload["elo_delta_b_vs_a"])
    assert elo == elo  # not NaN