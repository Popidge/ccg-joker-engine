from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

import torch
from typer.testing import CliRunner

from ccjoker.selfplay import app as selfplay_app
from ccjoker.dataset import TriplecargoDataset
from ccjoker.model import ModelConfig, PolicyValueNet


def _save_tiny_checkpoint(path: Path, num_cards: int) -> None:
    cfg = ModelConfig(num_cards=num_cards, card_embed_dim=8, hidden_dim=32, dropout=0.0)
    model = PolicyValueNet(cfg)
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


def test_selfplay_smoke(tmp_path: Path):
    # Build a tiny checkpoint compatible with fixture card space
    ds = TriplecargoDataset("tests/fixtures/sample.jsonl")
    num_cards = ds.max_card_id + 2
    ckpt_path = tmp_path / "tiny.pt"
    _save_tiny_checkpoint(ckpt_path, num_cards)

    out_path = tmp_path / "selfplay.jsonl"

    # Run 2 games with stub env for CI speed
    runner = CliRunner()
    result = runner.invoke(
        selfplay_app,
        [
            "--model",
            str(ckpt_path),
            "--games",
            "2",
            "--out",
            str(out_path),
            "--rollouts",
            "4",
            "--temperature",
            "1.0",
            "--device",
            "cpu",
            "--rules",
            "none",
            "--use-stub",
        ],
    )
    assert result.exit_code == 0, f"ccj-selfplay CLI failed: {result.output}"
    assert out_path.exists(), "Expected self-play JSONL output"

    # Expect 2 games × 9 states = 18 lines
    lines = out_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 18

    # Validate schema properties
    for line in lines:
        rec: Dict[str, Any] = json.loads(line)
        # policy_target must be distribution map (keys "cardId-cell")
        pt = rec.get("policy_target", {})
        assert isinstance(pt, dict)
        s = sum(float(v) for v in pt.values())
        # either terminal last line can have empty {} (sum 0) or sum ~ 1.0
        assert s == 0.0 or abs(s - 1.0) < 1e-4

        # value_target ∈ {-1,0,1} (after backfill)
        vt = rec.get("value_target", 0)
        assert int(vt) in (-1, 0, 1)