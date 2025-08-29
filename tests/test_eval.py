import math
from pathlib import Path

import torch
from typer.testing import CliRunner

from ccjoker.dataset import TriplecargoDataset, collate_fn
from ccjoker.model import PolicyValueNet, ModelConfig
from ccjoker.utils import MOVE_SPACE
import ccjoker.eval as eval_mod


def make_x_batch_for_B(B: int):
    # Minimal x_batch with required keys
    x_batch = {
        "move_mask": torch.ones(B, MOVE_SPACE, dtype=torch.float32),
        "policy_targets_onehot": torch.full((B,), -100, dtype=torch.long),
        "policy_targets_mcts": torch.zeros(B, MOVE_SPACE, dtype=torch.float32),
        "policy_mask": torch.zeros(B, dtype=torch.bool),
        "off_pv": torch.zeros(B, dtype=torch.bool),
    }
    return x_batch


def test_topk_accuracy_synthetic():
    torch.manual_seed(0)
    B = 3
    x_batch = make_x_batch_for_B(B)

    # Rows:
    # 0: onehot at idx 7
    # 1: onehot at idx 13
    # 2: mcts with mass on idx 5 (argmax = 5)
    x_batch["policy_targets_onehot"][0] = 7
    x_batch["policy_targets_onehot"][1] = 13
    x_batch["policy_mask"][2] = True
    x_batch["policy_targets_mcts"][2, 5] = 1.0

    # Policy logits: make sure top-1 equals targets above
    policy_logits = torch.full((B, MOVE_SPACE), -10.0)
    policy_logits[0, 7] = 9.0
    policy_logits[1, 13] = 8.0
    policy_logits[2, 5] = 7.0

    # Value logits arbitrary (for 3-class), and y_value labels
    value_logits = torch.randn(B, 3)
    y_value = torch.tensor([0, 1, 2], dtype=torch.long)

    stats = eval_mod.compute_batch_metrics(policy_logits, value_logits, x_batch, y_value)
    final_all = eval_mod.finalize_metrics(stats["all"])

    assert final_all["n_samples"] == 3
    # denom = 2 valid onehot + 1 mcts = 3
    assert math.isclose(final_all["top1"], 1.0, rel_tol=0, abs_tol=0)
    assert math.isclose(final_all["top2"], 1.0, rel_tol=0, abs_tol=0)
    assert math.isclose(final_all["top3"], 1.0, rel_tol=0, abs_tol=0)


def test_kl_correct_simple():
    torch.manual_seed(0)
    B = 2
    x_batch = make_x_batch_for_B(B)
    # All rows are mcts
    x_batch["policy_mask"][:] = True

    # Put probability mass only on indices 0 and 1 for both rows
    p = torch.zeros(B, MOVE_SPACE, dtype=torch.float32)
    # Row 0: p = [0.25, 0.75]
    p[0, 0] = 0.25
    p[0, 1] = 0.75
    # Row 1: p = [0.5, 0.5]
    p[1, 0] = 0.5
    p[1, 1] = 0.5
    x_batch["policy_targets_mcts"] = p.clone()

    # Build logits so that:
    # - Row 0: q == p -> KL=0
    # - Row 1: q = [0.6, 0.4] on indices 0,1 (others near zero)
    policy_logits = torch.full((B, MOVE_SPACE), -1e9)
    # Row 0
    policy_logits[0, 0] = math.log(0.25)
    policy_logits[0, 1] = math.log(0.75)
    # Row 1
    policy_logits[1, 0] = math.log(0.6)
    policy_logits[1, 1] = math.log(0.4)

    value_logits = torch.zeros(B, 3)
    y_value = torch.tensor([0, 0], dtype=torch.long)

    stats = eval_mod.compute_batch_metrics(policy_logits, value_logits, x_batch, y_value)
    final_all = eval_mod.finalize_metrics(stats["all"])
    assert stats["all"].kl_count == 2

    # Expected KL for row1 only (row0 contributes 0), mean over 2 rows
    kl_row1 = 0.5 * (math.log(0.5 / 0.6) + math.log(0.5 / 0.4))
    expected_mean = kl_row1 / 2.0
    assert final_all["kl"] is not None
    assert math.isclose(float(final_all["kl"]), float(expected_mean), rel_tol=1e-6, abs_tol=1e-6)


def test_mixed_subset_kl_only_on_mcts():
    torch.manual_seed(1)
    B = 3
    x_batch = make_x_batch_for_B(B)
    # Make row 2 an mcts row
    x_batch["policy_mask"][2] = True
    x_batch["policy_targets_onehot"][0] = 4
    x_batch["policy_targets_onehot"][1] = 8
    x_batch["policy_targets_mcts"][2, 10] = 1.0

    policy_logits = torch.full((B, MOVE_SPACE), -5.0)
    policy_logits[0, 4] = 3.0
    policy_logits[1, 8] = 2.5
    policy_logits[2, 10] = 1.5

    value_logits = torch.randn(B, 3)
    y_value = torch.tensor([0, 1, 2], dtype=torch.long)

    stats = eval_mod.compute_batch_metrics(policy_logits, value_logits, x_batch, y_value)
    # KL count equals number of mcts rows (1)
    assert stats["all"].kl_count == 1

    final_all = eval_mod.finalize_metrics(stats["all"])
    # When only one mcts row, KL should be a finite number (not None)
    assert final_all["kl"] is not None


def test_off_pv_slicing_with_fixture():
    ds = TriplecargoDataset("tests/fixtures/mixed.jsonl")
    batch = [ds[i] for i in range(len(ds))]
    x_b, y_policy_b, y_value_b = collate_fn(batch)

    B = y_value_b.size(0)
    # Create logits that choose:
    # - onehot rows via their class id (for determinism)
    # - mcts row via its argmax
    policy_logits = torch.full((B, MOVE_SPACE), -2.0)
    # Determine gt for onehot and mcts
    for i in range(B):
        if x_b["policy_mask"][i]:
            gt = int(x_b["policy_targets_mcts"][i].argmax().item())
        else:
            gt = int(x_b["policy_targets_onehot"][i].item())
            if gt == -100:
                gt = 0
        policy_logits[i, gt] = 5.0

    value_logits = torch.randn(B, 3)
    stats = eval_mod.compute_batch_metrics(policy_logits, value_logits, x_b, y_value_b)

    final_pv = eval_mod.finalize_metrics(stats["pv"])
    final_off = eval_mod.finalize_metrics(stats["off_pv"])
    final_all = eval_mod.finalize_metrics(stats["all"])

    # From fixture: first two off_pv=false, third off_pv=true
    assert final_pv["n_samples"] == 2
    assert final_off["n_samples"] == 1
    assert final_all["n_samples"] == 3

    # We constructed logits to be correct top-1
    assert math.isclose(final_pv["top1"], 1.0, rel_tol=0, abs_tol=0)
    assert math.isclose(final_off["top1"], 1.0, rel_tol=0, abs_tol=0)
    assert math.isclose(final_all["top1"], 1.0, rel_tol=0, abs_tol=0)


def test_cli_csv_output_two_modes(tmp_path: Path):
    # Prepare a tiny model checkpoint compatible with data
    data_path = Path("tests/fixtures/mixed.jsonl")
    ds = TriplecargoDataset(data_path)
    num_cards = ds.max_card_id + 2
    model = PolicyValueNet(ModelConfig(num_cards=num_cards))

    model_dir = tmp_path / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "dummy_model.pt"
    torch.save({"model_state_dict": model.state_dict()}, model_path)

    out_csv = tmp_path / "eval.csv"

    runner = CliRunner()
    # Single dataset mode with off_pv slicing
    result = runner.invoke(
        eval_mod.app,
        [
            "--data",
            str(data_path),
            "--model",
            str(model_path),
            "--batch-size",
            "2",
            "--csv-out",
            str(out_csv),
            "--num-workers",
            "0",
            "--device",
            "cpu",
        ],
    )
    assert result.exit_code == 0, f"CLI failed: {result.output}"
    assert out_csv.exists(), "CSV output was not created"
    # Basic content check
    content = out_csv.read_text(encoding="utf-8").strip().splitlines()
    assert len(content) >= 2, "CSV should have header + at least one row"
    header = content[0].split(",")
    assert header[:3] == ["timestamp", "model", "data_tag"]

    # Two-dataset mode: provide pv and mixed separately
    out_csv2 = tmp_path / "eval2.csv"
    result2 = runner.invoke(
        eval_mod.app,
        [
            "--data-pv",
            str(data_path),
            "--data-mixed",
            str(data_path),
            "--model",
            str(model_path),
            "--batch-size",
            "2",
            "--csv-out",
            str(out_csv2),
            "--num-workers",
            "0",
            "--device",
            "cpu",
        ],
    )
    assert result2.exit_code == 0, f"CLI two-dataset failed: {result2.output}"
    assert out_csv2.exists()
    lines2 = out_csv2.read_text(encoding="utf-8").strip().splitlines()
    # Expect header + 3 rows (pv, off_pv, aggregate) or at least 2 rows
    assert len(lines2) >= 3


def test_determinism_with_fixed_seed():
    # Build a fixed synthetic batch and ensure metrics repeat with fixed seed
    torch.manual_seed(1234)
    B = 4
    x_batch = make_x_batch_for_B(B)
    x_batch["policy_targets_onehot"] = torch.tensor([0, 1, -100, -100], dtype=torch.long)
    x_batch["policy_mask"] = torch.tensor([False, False, True, True], dtype=torch.bool)
    x_batch["policy_targets_mcts"][2, 5] = 0.3
    x_batch["policy_targets_mcts"][2, 6] = 0.7
    x_batch["policy_targets_mcts"][3, 10] = 1.0

    torch.manual_seed(999)
    policy_logits1 = torch.randn(B, MOVE_SPACE)
    value_logits1 = torch.randn(B, 3)
    y_value = torch.tensor([0, 1, 2, 0], dtype=torch.long)
    stats1 = eval_mod.compute_batch_metrics(policy_logits1, value_logits1, x_batch, y_value)
    fin1 = eval_mod.finalize_metrics(stats1["all"])

    torch.manual_seed(999)
    policy_logits2 = torch.randn(B, MOVE_SPACE)
    value_logits2 = torch.randn(B, 3)
    stats2 = eval_mod.compute_batch_metrics(policy_logits2, value_logits2, x_batch, y_value)
    fin2 = eval_mod.finalize_metrics(stats2["all"])

    # All finalized floats should match exactly with same seed and inputs
    for key in ["top1", "top2", "top3", "kl", "value_acc"]:
        v1 = fin1[key]
        v2 = fin2[key]
        if v1 is None and v2 is None:
            continue
        assert math.isclose(float(v1), float(v2), rel_tol=0.0, abs_tol=0.0)