from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from typer.testing import CliRunner

from ccjoker.selfplay import app as selfplay_app
from ccjoker.train_rl import app as train_rl_app
from ccjoker.dataset import TriplecargoDataset, collate_fn
from ccjoker.model import ModelConfig, PolicyValueNet
from ccjoker.train import evaluate


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


def _evaluate_total_loss_on_dataset(model: PolicyValueNet, data_path: Path) -> float:
    ds = TriplecargoDataset(data_path)
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0, collate_fn=collate_fn)
    metrics = evaluate(model, loader, device=torch.device("cpu"), value_loss_weight=0.5)
    return float(metrics["total_loss"])


def _load_model_from_checkpoint(path: Path) -> PolicyValueNet:
    payload = torch.load(path, map_location="cpu")
    cfg = ModelConfig(**payload["config"])
    model = PolicyValueNet(cfg)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model


def test_train_rl_smoke(tmp_path: Path):
    # Build tiny model checkpoint
    ds_seed = TriplecargoDataset("tests/fixtures/sample.jsonl")
    num_cards = ds_seed.max_card_id + 2
    ckpt_path = tmp_path / "tiny.pt"
    _save_tiny_checkpoint(ckpt_path, num_cards)

    # Generate a small self-play dataset
    sp_path = tmp_path / "sp.jsonl"
    runner = CliRunner()
    sp_res = runner.invoke(
        selfplay_app,
        [
            "--model",
            str(ckpt_path),
            "--games",
            "2",
            "--out",
            str(sp_path),
            "--rollouts",
            "4",
            "--device",
            "cpu",
            "--rules",
            "none",
            "--use-stub",
        ],
    )
    assert sp_res.exit_code == 0, f"self-play failed: {sp_res.output}"
    assert sp_path.exists()

    # Evaluate baseline (random tiny) on self-play dataset
    baseline_model = PolicyValueNet(ModelConfig(num_cards=num_cards))
    baseline_model.eval()
    base_loss = _evaluate_total_loss_on_dataset(baseline_model, sp_path)

    # Train RL for 1 epoch on this self-play file
    out_model = tmp_path / "rl.pt"
    tr_res = runner.invoke(
        train_rl_app,
        [
            "--data",
            str(sp_path),
            "--epochs",
            "1",
            "--batch-size",
            "4",
            "--lr",
            "1e-3",
            "--out",
            str(out_model),
            "--device",
            "cpu",
        ],
    )
    assert tr_res.exit_code == 0, f"ccj-train-rl failed: {tr_res.output}"
    assert out_model.exists(), "RL checkpoint was not written"

    # Evaluate trained model on the same dataset and check improvement vs baseline
    trained_model = _load_model_from_checkpoint(out_model)
    trained_loss = _evaluate_total_loss_on_dataset(trained_model, sp_path)
    assert trained_loss <= base_loss + 1e-6, f"trained loss {trained_loss} did not improve vs baseline {base_loss}"