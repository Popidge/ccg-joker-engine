from dataclasses import asdict
from pathlib import Path
from typing import Tuple
import math

import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typer.testing import CliRunner

import ccjoker.train as train_mod
from ccjoker.train import TrainArgs, train_loop, compute_mixed_policy_loss, evaluate
from ccjoker.model import mask_policy_logits, ModelConfig, PolicyValueNet
from ccjoker.dataset import TriplecargoDataset, collate_fn
from ccjoker.utils import MOVE_SPACE


def test_training_smoke(tmp_path: Path = None):
    # Use fixture as both train and val to keep it simple/small
    data_path = Path("tests/fixtures/sample.jsonl")
    assert data_path.exists(), "Fixture JSONL missing"

    out_dir = Path("data/models")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "smoke_model.pt"

    args = TrainArgs(
        data=data_path,
        val_data=data_path,
        epochs=1,
        batch_size=2,
        lr=1e-3,
        weight_decay=0.0,
        out=out_path,
        val_split=0.5,
        num_workers=0,
        seed=123,
        device="cpu",
    )

    # Run a tiny training loop; should not raise
    train_loop(args)

    assert out_path.exists(), "Expected checkpoint file not created"
    payload = torch.load(out_path, map_location="cpu")
    assert "model_state_dict" in payload


def test_policy_loss_onehot_ce_branch():
    torch.manual_seed(123)
    B = 3
    logits = torch.randn(B, MOVE_SPACE)
    move_mask = torch.ones(B, MOVE_SPACE)
    # targets: two valid, one ignored
    targets_onehot = torch.tensor([1, 2, -100], dtype=torch.long)
    targets_mcts = torch.zeros(B, MOVE_SPACE, dtype=torch.float32)
    policy_mask = torch.tensor([False, False, False], dtype=torch.bool)

    loss = compute_mixed_policy_loss(
        logits=logits,
        move_mask=move_mask,
        targets_onehot=targets_onehot,
        targets_mcts=targets_mcts,
        policy_mask=policy_mask,
    )

    masked = mask_policy_logits(logits, move_mask)
    expected = F.cross_entropy(masked, targets_onehot, ignore_index=-100, reduction="mean")
    assert math.isclose(float(loss.item()), float(expected.item()), rel_tol=1e-6, abs_tol=1e-6)


def test_policy_loss_mcts_kl_branch():
    torch.manual_seed(123)
    B = 2
    logits = torch.randn(B, MOVE_SPACE)
    move_mask = torch.ones(B, MOVE_SPACE)
    targets_onehot = torch.full((B,), -100, dtype=torch.long)
    targets_mcts = torch.zeros(B, MOVE_SPACE, dtype=torch.float32)
    # Put probability mass on two indices for row 0 and one index for row 1
    targets_mcts[0, 0] = 0.25
    targets_mcts[0, 10] = 0.75
    targets_mcts[1, 5] = 1.0
    policy_mask = torch.tensor([True, True], dtype=torch.bool)

    loss = compute_mixed_policy_loss(
        logits=logits,
        move_mask=move_mask,
        targets_onehot=targets_onehot,
        targets_mcts=targets_mcts,
        policy_mask=policy_mask,
    )

    # Expected KL mean across rows
    eps = 1e-8
    masked = mask_policy_logits(logits, move_mask)
    q = F.softmax(masked, dim=-1).clamp(min=eps)
    p = targets_mcts / targets_mcts.sum(dim=-1, keepdim=True).clamp(min=eps)
    log_p = torch.where(p > 0, torch.log(p), torch.zeros_like(p))
    log_q = torch.where(p > 0, torch.log(q), torch.zeros_like(q))
    per_row_kl = (p * (log_p - log_q)).sum(dim=-1)
    expected = per_row_kl.mean()
    assert math.isclose(float(loss.item()), float(expected.item()), rel_tol=1e-6, abs_tol=1e-6)


def test_policy_loss_mixed_weighted_average():
    torch.manual_seed(321)
    B = 3
    logits = torch.randn(B, MOVE_SPACE)
    move_mask = torch.ones(B, MOVE_SPACE)

    # Row 0: CE with valid class
    targets_onehot = torch.tensor([7, -100, -100], dtype=torch.long)
    # Row 1-2: MCTS distributions
    targets_mcts = torch.zeros(B, MOVE_SPACE, dtype=torch.float32)
    targets_mcts[1, 0] = 0.4
    targets_mcts[1, 10] = 0.6
    targets_mcts[2, 5] = 1.0

    policy_mask = torch.tensor([False, True, True], dtype=torch.bool)

    mixed = compute_mixed_policy_loss(
        logits=logits,
        move_mask=move_mask,
        targets_onehot=targets_onehot,
        targets_mcts=targets_mcts,
        policy_mask=policy_mask,
    )

    # Manually compute weighted average
    masked = mask_policy_logits(logits, move_mask)
    # CE part (1 valid row)
    ce_logits = masked[0:1]
    ce_targets = targets_onehot[0:1]
    ce_sum = F.cross_entropy(ce_logits, ce_targets, ignore_index=-100, reduction="sum")
    n_ce = int((ce_targets != -100).sum().item())

    # KL part (2 rows)
    eps = 1e-8
    kl_logits = masked[1:]
    p = targets_mcts[1:].to(dtype=kl_logits.dtype)
    p = p / p.sum(dim=-1, keepdim=True).clamp(min=eps)
    q = F.softmax(kl_logits, dim=-1).clamp(min=eps)
    log_p = torch.where(p > 0, torch.log(p), torch.zeros_like(p))
    log_q = torch.where(p > 0, torch.log(q), torch.zeros_like(q))
    per_row_kl = (p * (log_p - log_q)).sum(dim=-1)
    kl_sum = per_row_kl.sum()
    n_kl = per_row_kl.numel()

    expected = (ce_sum + kl_sum) / float(n_ce + n_kl)
    assert math.isclose(float(mixed.item()), float(expected.item()), rel_tol=1e-6, abs_tol=1e-6)


def test_determinism_policy_loss_fixed_seed():
    B = 4
    torch.manual_seed(777)
    logits1 = torch.randn(B, MOVE_SPACE)
    move_mask1 = torch.ones(B, MOVE_SPACE)
    t_onehot1 = torch.tensor([1, 2, -100, -100], dtype=torch.long)
    t_mcts1 = torch.zeros(B, MOVE_SPACE, dtype=torch.float32)
    t_mcts1[2, 0] = 0.3
    t_mcts1[2, 9] = 0.7
    t_mcts1[3, 5] = 1.0
    p_mask1 = torch.tensor([False, False, True, True], dtype=torch.bool)

    loss1 = compute_mixed_policy_loss(logits1, move_mask1, t_onehot1, t_mcts1, p_mask1)

    torch.manual_seed(777)
    logits2 = torch.randn(B, MOVE_SPACE)
    move_mask2 = torch.ones(B, MOVE_SPACE)
    t_onehot2 = t_onehot1.clone()
    t_mcts2 = t_mcts1.clone()
    p_mask2 = p_mask1.clone()

    loss2 = compute_mixed_policy_loss(logits2, move_mask2, t_onehot2, t_mcts2, p_mask2)

    assert math.isclose(float(loss1.item()), float(loss2.item()), rel_tol=0.0, abs_tol=0.0)


def test_cli_value_loss_weight_override(monkeypatch: pytest.MonkeyPatch):
    recorded = {}

    def stub_train_loop(args):
        recorded["vlw"] = args.value_loss_weight

    runner = CliRunner()
    monkeypatch.setattr(train_mod, "train_loop", stub_train_loop)

    # Invoke CLI with override
    result = runner.invoke(
        train_mod.app,
        [
            "--data",
            "tests/fixtures/sample.jsonl",
            "--val-data",
            "tests/fixtures/sample.jsonl",
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--value-loss-weight",
            "0.25",
        ],
    )
    assert result.exit_code == 0, f"CLI failed: {result.output}"
    assert math.isclose(float(recorded["vlw"]), 0.25, rel_tol=0.0, abs_tol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cli_amp_flag_cuda_disable(monkeypatch: pytest.MonkeyPatch):
    import ccjoker.train as train_mod

    recorded = {}

    def stub_train_loop(args):
        recorded["amp"] = args.amp
        recorded["device"] = args.device

    runner = CliRunner()
    monkeypatch.setattr(train_mod, "train_loop", stub_train_loop)

    result = runner.invoke(
        train_mod.app,
        [
            "--data",
            "tests/fixtures/sample.jsonl",
            "--val-data",
            "tests/fixtures/sample.jsonl",
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--device",
            "cuda",
            "--no-amp",
        ],
    )
    assert result.exit_code == 0, f"CLI failed: {result.output}"
    assert recorded["device"] == "cuda"
    assert recorded["amp"] is False


def _build_loader(path: str, batch_size: int = 4) -> Tuple[TriplecargoDataset, DataLoader]:
    ds = TriplecargoDataset(path)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=False,
    )
    return ds, loader


def _evaluate_checkpoint(chk_path: str) -> float:
    payload = torch.load(chk_path, map_location="cpu")
    cfg = ModelConfig(**payload["config"])
    model = PolicyValueNet(cfg)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    _, val_loader = _build_loader("tests/fixtures/sample.jsonl", batch_size=2)
    metrics = evaluate(model, val_loader, torch.device("cpu"), value_loss_weight=0.5)
    return float(metrics["total_loss"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cpu_vs_cuda_amp_smoke_parity(tmp_path: Path):
    # Common data/params
    data_path = Path("tests/fixtures/sample.jsonl")
    assert data_path.exists()

    # Train CPU (FP32)
    cpu_out = tmp_path / "cpu.pt"
    cpu_args = TrainArgs(
        data=data_path,
        val_data=data_path,
        epochs=2,
        batch_size=2,
        lr=1e-3,
        weight_decay=0.0,
        out=cpu_out,
        val_split=0.5,
        num_workers=0,
        seed=123,
        device="cpu",
        value_loss_weight=0.5,
        amp=False,
    )
    train_loop(cpu_args)
    assert cpu_out.exists()

    # Train CUDA with AMP (default on)
    amp_out = tmp_path / "amp.pt"
    cuda_args = TrainArgs(
        data=data_path,
        val_data=data_path,
        epochs=2,
        batch_size=2,
        lr=1e-3,
        weight_decay=0.0,
        out=amp_out,
        val_split=0.5,
        num_workers=0,
        seed=123,
        device="cuda",
        value_loss_weight=0.5,
        amp=None,  # default behavior: on for CUDA
    )
    train_loop(cuda_args)
    assert amp_out.exists()

    # Evaluate untrained baseline for "decrease" check
    ds, val_loader = _build_loader(str(data_path), batch_size=2)
    cfg_baseline = ModelConfig(num_cards=ds.max_card_id + 2)
    baseline_model = PolicyValueNet(cfg_baseline)
    base_metrics = evaluate(baseline_model, val_loader, torch.device("cpu"), value_loss_weight=0.5)
    base_loss = float(base_metrics["total_loss"])

    # Evaluate both checkpoints on CPU
    cpu_loss = _evaluate_checkpoint(str(cpu_out))
    amp_loss = _evaluate_checkpoint(str(amp_out))

    # Both improved vs baseline
    assert cpu_loss <= base_loss, f"CPU-trained loss {cpu_loss} did not improve vs baseline {base_loss}"
    assert amp_loss <= base_loss, f"AMP-trained loss {amp_loss} did not improve vs baseline {base_loss}"

    # Final losses within ~10% relative difference
    denom = max(cpu_loss, amp_loss, 1e-8)
    rel_diff = abs(cpu_loss - amp_loss) / denom
    assert rel_diff <= 0.10, f"CPU vs AMP final losses differ by {rel_diff*100:.1f}% (>10%)"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_determinism_when_amp_off(tmp_path: Path):
    data_path = Path("tests/fixtures/sample.jsonl")
    assert data_path.exists()

    # First run (CUDA, AMP off)
    out1 = tmp_path / "run1.pt"
    args1 = TrainArgs(
        data=data_path,
        val_data=data_path,
        epochs=2,
        batch_size=2,
        lr=1e-3,
        weight_decay=0.0,
        out=out1,
        val_split=0.5,
        num_workers=0,
        seed=777,
        device="cuda",
        value_loss_weight=0.5,
        amp=False,  # force FP32 on CUDA
    )
    train_loop(args1)
    loss1 = _evaluate_checkpoint(str(out1))

    # Second run with identical params/seed
    out2 = tmp_path / "run2.pt"
    args2 = TrainArgs(**{**asdict(args1), "out": out2})
    train_loop(args2)
    loss2 = _evaluate_checkpoint(str(out2))

    assert math.isclose(loss1, loss2, rel_tol=0.0, abs_tol=0.0)