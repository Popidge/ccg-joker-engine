from pathlib import Path

import torch

from ccjoker.dataset import TriplecargoDataset, collate_fn
from ccjoker.utils import MOVE_SPACE


def test_dataset_item_shapes():
    sample_path = Path("tests/fixtures/sample.jsonl")
    assert sample_path.exists(), "Fixture JSONL not found"
    ds = TriplecargoDataset(sample_path)
    assert len(ds) == 3

    x, y_policy, y_value = ds[0]

    assert x["board_card_ids"].shape == (9,)
    assert x["board_owner"].shape == (9, 3)
    assert x["board_element"].shape == (9, 9)
    assert x["hand_card_ids"].shape == (5,)
    assert x["hand_mask"].shape == (5,)
    assert x["rules"].shape == (4,)
    assert x["move_mask"].shape == (MOVE_SPACE,)

    # policy can be class [1] long or distribution [45] float
    if y_policy.dtype == torch.long:
        assert y_policy.shape == (1,)
        assert (y_policy.item() == -100) or (0 <= int(y_policy.item()) < MOVE_SPACE)
    else:
        assert y_policy.shape == (MOVE_SPACE,)
        assert torch.all(y_policy >= 0)
        s = float(y_policy.sum().item())
        assert s == 0 or abs(s - 1.0) < 1e-4

    assert y_value.shape == (1,)
    assert int(y_value.item()) in (0, 1, 2)


def test_collate_fn_batching():
    ds = TriplecargoDataset("tests/fixtures/sample.jsonl")
    batch = [ds[0], ds[1]]
    x_b, y_policy_b, y_value_b = collate_fn(batch)

    assert x_b["board_card_ids"].shape == (2, 9)
    assert x_b["board_owner"].shape == (2, 9, 3)
    assert x_b["board_element"].shape == (2, 9, 9)
    assert x_b["hand_card_ids"].shape == (2, 5)
    assert x_b["hand_mask"].shape == (2, 5)
    assert x_b["rules"].shape == (2, 4)
    assert x_b["move_mask"].shape == (2, MOVE_SPACE)

    # With fixture (onehot), collated policy should be [B]
    if y_policy_b.dtype == torch.long:
        assert y_policy_b.shape == (2,)
    else:
        assert y_policy_b.shape == (2, MOVE_SPACE)

    assert y_value_b.shape == (2,)
    for v in y_value_b.tolist():
        assert int(v) in (0, 1, 2)
import math

def test_mixed_dataset_off_pv_and_policy_masks():
    ds = TriplecargoDataset("tests/fixtures/mixed.jsonl")
    assert len(ds) == 3

    # Sample 0: onehot, to_move A, hand A = [1,2,3,4,5], target {card_id:1, cell:0} => slot 0, idx 0
    x0, yp0, yv0 = ds[0]
    assert x0["off_pv"].dtype == torch.bool
    assert x0["policy_is_mcts"].dtype == torch.bool
    assert not bool(x0["off_pv"])
    assert not bool(x0["policy_is_mcts"])
    assert yp0.dtype == torch.long and yp0.shape == (1,)
    assert int(yp0.item()) == 0  # slot 0 * 9 + cell 0

    # Sample 1: mcts, to_move B, hand B = [11,12,13,14,15], mcts {"11-0":0.25,"12-1":0.75}
    x1, yp1, yv1 = ds[1]
    assert not bool(x1["off_pv"])
    assert bool(x1["policy_is_mcts"])
    assert yp1.dtype != torch.long  # distribution
    assert yp1.shape == (MOVE_SPACE,)
    s1 = float(yp1.sum().item())
    assert abs(s1 - 1.0) < 1e-6
    # Expected indices: 11 at slot 0, cell 0 -> 0; 12 at slot 1, cell 1 -> 10
    assert math.isclose(float(yp1[0].item()), 0.25, rel_tol=1e-6, abs_tol=1e-6)
    assert math.isclose(float(yp1[10].item()), 0.75, rel_tol=1e-6, abs_tol=1e-6)

    # Sample 2: onehot with off_pv=true, to_move A, hand A = [21,22,23,24,25], {card_id:21, cell:8}
    x2, yp2, yv2 = ds[2]
    assert bool(x2["off_pv"])
    assert not bool(x2["policy_is_mcts"])
    assert yp2.dtype == torch.long and yp2.shape == (1,)
    assert int(yp2.item()) == 8  # slot 0 * 9 + cell 8

    # Collate all three (mixed batch)
    x_b, y_policy_b, y_value_b = collate_fn([ds[0], ds[1], ds[2]])

    # New fields
    assert x_b["off_pv"].dtype == torch.bool and x_b["off_pv"].shape == (3,)
    assert x_b["policy_is_mcts"].dtype == torch.bool and x_b["policy_is_mcts"].shape == (3,)
    assert x_b["policy_mask"].dtype == torch.bool and x_b["policy_mask"].shape == (3,)
    assert x_b["policy_targets_onehot"].dtype == torch.long and x_b["policy_targets_onehot"].shape == (3,)
    assert x_b["policy_targets_mcts"].dtype == torch.float32 and x_b["policy_targets_mcts"].shape == (3, MOVE_SPACE)

    # Masks: [False(onehot), True(mcts), False(onehot)]
    assert x_b["off_pv"].tolist() == [False, False, True]
    assert x_b["policy_is_mcts"].tolist() == [False, True, False]
    assert x_b["policy_mask"].tolist() == [False, True, False]

    # Targets alignment
    assert x_b["policy_targets_onehot"].tolist() == [0, -100, 8]
    row0 = x_b["policy_targets_mcts"][0]
    row1 = x_b["policy_targets_mcts"][1]
    row2 = x_b["policy_targets_mcts"][2]
    # Onehot rows should be zero distributions
    assert torch.all(row0 == 0)
    assert torch.all(row2 == 0)
    # MCTS row should be normalized and put mass at indices 0 and 10
    s_row1 = float(row1.sum().item())
    assert abs(s_row1 - 1.0) < 1e-6
    assert math.isclose(float(row1[0].item()), 0.25, rel_tol=1e-6, abs_tol=1e-6)
    assert math.isclose(float(row1[10].item()), 0.75, rel_tol=1e-6, abs_tol=1e-6)

    # Legacy y_policy should be a distribution batch since mixed
    assert y_policy_b.shape == (3, MOVE_SPACE)
    assert y_policy_b.dtype == torch.float32
    sums = y_policy_b.sum(dim=-1).tolist()
    for s in sums:
        assert abs(float(s) - 1.0) < 1e-5