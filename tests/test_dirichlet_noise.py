from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List

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


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").strip().splitlines()]


def _sum_probs(pt: Dict[str, float]) -> float:
    return float(sum(float(v) for v in pt.values()))


def _is_legal(rec: Dict[str, Any]) -> bool:
    pt = rec.get("policy_target", {})
    if not isinstance(pt, dict) or not pt:
        return True
    to_move = rec["to_move"]
    hand = list(rec["hands"][to_move])
    empty = {c["cell"] for c in rec["board"] if c.get("card_id") is None}
    ok = True
    for k in pt.keys():
        try:
            cid_str, cell_str = k.split("-", 1)
            cid = int(cid_str)
            cell = int(cell_str)
        except Exception:
            return False
        if cid not in hand:
            ok = False
            break
        if cell not in empty:
            ok = False
            break
    return ok


def _dicts_close(a: Dict[str, float], b: Dict[str, float], tol: float = 1e-6) -> bool:
    if set(a.keys()) != set(b.keys()):
        return False
    for k in a.keys():
        if abs(float(a[k]) - float(b[k])) > tol:
            return False
    return True


def _dists_differ(a: Dict[str, float], b: Dict[str, float], tol: float = 1e-6) -> bool:
    # If keys differ or any value changes beyond tol, consider different
    if set(a.keys()) != set(b.keys()):
        return True
    for k in a.keys():
        if abs(float(a[k]) - float(b[k])) > tol:
            return True
    return False


def test_dirichlet_noise_disabled_deterministic(tmp_path: Path):
    ds = TriplecargoDataset("tests/fixtures/sample.jsonl")
    num_cards = ds.max_card_id + 2
    ckpt = tmp_path / "tiny.pt"
    _save_tiny_checkpoint(ckpt, num_cards)

    out1 = tmp_path / "sp1.jsonl"
    out2 = tmp_path / "sp2.jsonl"

    runner = CliRunner()
    common_args = [
        "--model",
        str(ckpt),
        "--games",
        "1",
        "--out",
        str(out1),
        "--rollouts",
        "4",
        "--temperature",
        "1.0",
        "--device",
        "cpu",
        "--rules",
        "none",
        "--use-stub",
        "--seed",
        "123",
        "--dirichlet-eps",
        "0.0",
    ]
    res1 = runner.invoke(selfplay_app, common_args)
    assert res1.exit_code == 0, f"run1 failed: {res1.output}"

    # second run to out2
    args2 = common_args.copy()
    args2[5] = str(out2)  # update path after "--out"
    res2 = runner.invoke(selfplay_app, args2)
    assert res2.exit_code == 0, f"run2 failed: {res2.output}"

    recs1 = _read_jsonl(out1)
    recs2 = _read_jsonl(out2)
    assert len(recs1) == len(recs2) == 9

    for r1, r2 in zip(recs1, recs2):
        pt1 = r1.get("policy_target", {})
        pt2 = r2.get("policy_target", {})
        # identical when disabled and seeded
        assert _dicts_close(pt1, pt2, tol=1e-8)

        s1 = _sum_probs(pt1)
        if pt1:
            assert abs(s1 - 1.0) < 1e-6
            assert _is_legal(r1)


def test_dirichlet_noise_enabled_nondeterministic_and_valid(tmp_path: Path):
    ds = TriplecargoDataset("tests/fixtures/sample.jsonl")
    num_cards = ds.max_card_id + 2
    ckpt = tmp_path / "tiny.pt"
    _save_tiny_checkpoint(ckpt, num_cards)

    out1 = tmp_path / "sp1_noise.jsonl"
    out2 = tmp_path / "sp2_noise.jsonl"

    runner = CliRunner()
    args = [
        "--model",
        str(ckpt),
        "--games",
        "1",
        "--out",
        str(out1),
        "--rollouts",
        "4",
        "--temperature",
        "1.0",
        "--device",
        "cpu",
        "--rules",
        "none",
        "--use-stub",
        "--dirichlet-alpha",
        "0.3",
        "--dirichlet-eps",
        "0.25",
        # no seed -> allow non-determinism
    ]
    res1 = runner.invoke(selfplay_app, args)
    assert res1.exit_code == 0, f"run1 failed: {res1.output}"

    args2 = args.copy()
    args2[5] = str(out2)  # update path after "--out"
    res2 = runner.invoke(selfplay_app, args2)
    assert res2.exit_code == 0, f"run2 failed: {res2.output}"

    recs1 = _read_jsonl(out1)
    recs2 = _read_jsonl(out2)
    assert len(recs1) == len(recs2) == 9

    # Non-deterministic: distributions should differ in at least one state
    differs = False
    for r1, r2 in zip(recs1, recs2):
        pt1 = r1.get("policy_target", {})
        pt2 = r2.get("policy_target", {})
        if _dists_differ(pt1, pt2, tol=1e-6):
            differs = True
            break
    assert differs, "Expected differing distributions with Dirichlet noise enabled"

    # Validate legality and normalization
    for rec in recs1 + recs2:
        pt = rec.get("policy_target", {})
        if pt:
            assert abs(_sum_probs(pt) - 1.0) < 1e-6
            assert _is_legal(rec)