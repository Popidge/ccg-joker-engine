from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset

from . import utils as U


Record = Dict[str, Any]


def _load_jsonl(path: Union[str, Path]) -> List[Record]:
    data: List[Record] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def _index_board(cells: Iterable[Dict[str, Any]], elemental: bool) -> List[Dict[str, Any]]:
    """
    Normalize board entries into a list indexed by cell 0..8.
    Each normalized entry has: {"card_id": Optional[int], "owner": Optional[str], "element": Optional[str]}
    """
    out: List[Dict[str, Any]] = [{"card_id": None, "owner": None, "element": None} for _ in range(U.NUM_CELLS)]
    for entry in cells:
        c = int(entry.get("cell", -1))
        if 0 <= c < U.NUM_CELLS:
            out[c] = {
                "card_id": entry.get("card_id", None),
                "owner": entry.get("owner", None),
                "element": entry.get("element", None) if elemental else None,
            }
    return out


def _max_in_list(ints: Iterable[Optional[int]]) -> int:
    m = -1
    for v in ints:
        if v is None:
            continue
        if v > m:
            m = int(v)
    return m


class TriplecargoDataset(Dataset):
    """
    JSONL dataset for CC Group Joker Engine.

    Each JSONL record is expected to conform to the provided schema with keys including:
      - game_id, state_idx, board (list of cells with {cell, card_id, owner, element?})
      - hands: {"A": [int], "B": [int]}
      - to_move: "A" | "B"
      - rules: {"elemental": bool, "same": bool, "plus": bool, "same_wall": bool}
      - off_pv: boolean (optional; default False if absent)
      - policy_target:
          onehot: {"card_id": int, "cell": int}
          mcts: {"cardId-cell": float, ...} (distribution over legal moves; normalized on load)
      - value_mode: "winloss" | "margin"
      - value_target: int

    The dataset returns a tuple: (x: Dict[str, Tensor], y_policy: Tensor, y_value: Tensor)
    where:
      - x["board_card_ids"]: LongTensor [9]
      - x["board_owner"]:    FloatTensor [9,3]
      - x["board_element"]:  FloatTensor [9,9]
      - x["hand_card_ids"]:  LongTensor [5]
      - x["hand_mask"]:      FloatTensor [5]
      - x["rules"]:          FloatTensor [4]
      - x["move_mask"]:      FloatTensor [45]
      - x["off_pv"]:         BoolTensor [] that stacks to [B]
      - x["policy_is_mcts"]: BoolTensor [] that stacks to [B] (True if policy is mcts, else False)
      - y_policy: LongTensor [] (class 0..44) or FloatTensor [45] (distribution) depending on policy_target
                  This is kept for backward-compat with existing training/eval.
      - y_value:  LongTensor [] in {0,1,2}

    Collate notes:
      - In addition to the legacy y_policy tensor above, collate_fn also adds to x-batch:
          x["policy_targets_onehot"]: LongTensor [B] (class indices; -100 where sample is mcts)
          x["policy_targets_mcts"]:   FloatTensor [B,45] (distributions; zeros where sample is onehot)
          x["policy_mask"]:           BoolTensor [B] (True if mcts, False if onehot)
      - This enables mixed-policy batches while keeping legacy outputs intact.
    """

    def __init__(self, path: Union[str, Path]) -> None:
        self.path = str(path)
        self.records: List[Record] = _load_jsonl(path)

        # Pre-compute max card id observed to help model sizing
        max_seen = -1
        for r in self.records:
            rules = r.get("rules", {})
            elemental = bool(rules.get("elemental", False))
            board = _index_board(r.get("board", []), elemental=elemental)
            bmax = _max_in_list([cell.get("card_id") for cell in board])
            hA = r.get("hands", {}).get("A", [])
            hB = r.get("hands", {}).get("B", [])
            hmax = _max_in_list(hA + hB)
            max_seen = max(max_seen, bmax, hmax)
        # +1 for 0-based padding, +1 to include the last id -> padding_idx=0, ids shift by +1
        self.max_card_id: int = max(0, max_seen)

    def __len__(self) -> int:
        return len(self.records)

    def _encode_board(self, rec: Record) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rules = rec.get("rules", {})
        elemental = bool(rules.get("elemental", False))
        board = _index_board(rec.get("board", []), elemental=elemental)
        owners: List[torch.Tensor] = []
        elements: List[torch.Tensor] = []
        card_ids: List[int] = []
        for cell in board:
            owners.append(U.owner_onehot(cell.get("owner")))
            elements.append(U.element_onehot(cell.get("element")))
            card_ids.append(U.card_id_to_embed_index(cell.get("card_id")))
        board_owner = torch.stack(owners, dim=0)  # [9,3]
        board_element = torch.stack(elements, dim=0)  # [9,9]
        board_card_ids = torch.tensor(card_ids, dtype=torch.long)  # [9]
        return board_card_ids, board_owner, board_element

    def _encode_hand(self, rec: Record) -> Tuple[torch.Tensor, torch.Tensor, List[Optional[int]]]:
        to_move = rec.get("to_move", "A")
        hand_ids_raw: List[int] = list(rec.get("hands", {}).get(to_move, []))
        # we keep raw ids to map policy targets
        padded_embed_ids, hand_mask = U.pad_hand_ids(hand_ids_raw, pad_to=U.HAND_SLOTS)
        hand_card_ids = torch.tensor(padded_embed_ids, dtype=torch.long)  # [5]
        hand_mask_t = torch.tensor(hand_mask, dtype=torch.float32)  # [5]
        # raw hand with None for pads (align with padded_embed_ids)
        raw_with_pads: List[Optional[int]] = (hand_ids_raw + [None] * U.HAND_SLOTS)[: U.HAND_SLOTS]
        return hand_card_ids, hand_mask_t, raw_with_pads

    def _encode_rules(self, rec: Record) -> torch.Tensor:
        return U.rules_to_tensor(rec.get("rules", {}))

    def _encode_policy(self, rec: Record, hand_raw: List[Optional[int]]) -> torch.Tensor:
        """
        Returns either:
          - LongTensor scalar containing class index 0..44 (shape [1], squeeze later) for onehot
          - FloatTensor [45] distribution for mcts
          - LongTensor scalar value -100 if onehot refers to a card not in hand (ignored index)
        """
        pt = rec.get("policy_target")
        if pt is None:
            # no policy provided; return uniform over valid moves
            move_mask = U.build_move_mask([1 if x is not None else 0 for x in hand_raw])
            probs = move_mask / move_mask.sum().clamp(min=1.0)
            return probs.to(torch.float32)

        # hand without pads for lookup
        hand_no_pads: List[int] = [int(x) for x in hand_raw if x is not None]

        if isinstance(pt, dict) and "card_id" in pt and "cell" in pt:
            idx = U.onehot_policy_to_index(hand_no_pads, pt)
            if idx is None:
                # Use ignore index -100 to allow CE(ignore_index=-100)
                return torch.tensor([-100], dtype=torch.long)
            return torch.tensor([idx], dtype=torch.long)

        if isinstance(pt, dict):
            # mcts-style map: "card-cell": prob
            probs = U.mcts_policy_to_probs(hand_no_pads, pt)
            return probs.to(torch.float32)

        # Fallback uniform if unknown format
        move_mask = U.build_move_mask([1 if x is not None else 0 for x in hand_raw])
        probs = move_mask / move_mask.sum().clamp(min=1.0)
        return probs.to(torch.float32)

    def _encode_value(self, rec: Record) -> torch.Tensor:
        vm = str(rec.get("value_mode", "winloss"))
        vt = int(rec.get("value_target", 0))
        cls = U.value_to_class(vm, vt)
        return torch.tensor([cls], dtype=torch.long)

    def __getitem__(self, idx: int):
        rec = self.records[idx]

        board_card_ids, board_owner, board_element = self._encode_board(rec)
        hand_card_ids, hand_mask, hand_raw = self._encode_hand(rec)
        rules = self._encode_rules(rec)
        move_mask = U.build_move_mask(hand_mask.tolist())

        # Determine policy target type for this record
        pt = rec.get("policy_target")
        is_onehot = isinstance(pt, dict) and ("card_id" in pt) and ("cell" in pt)
        is_mcts = not is_onehot

        # Encode targets
        y_policy = self._encode_policy(rec, hand_raw)
        y_value = self._encode_value(rec)

        # off_pv flag (default False)
        off_pv = bool(rec.get("off_pv", False))

        x = {
            "board_card_ids": board_card_ids,  # [9] long
            "board_owner": board_owner,  # [9,3] float
            "board_element": board_element,  # [9,9] float
            "hand_card_ids": hand_card_ids,  # [5] long
            "hand_mask": hand_mask,  # [5] float
            "rules": rules,  # [4] float
            "move_mask": move_mask,  # [45] float
            "off_pv": torch.tensor(off_pv, dtype=torch.bool),  # [] -> [B] after stack
            "policy_is_mcts": torch.tensor(is_mcts, dtype=torch.bool),  # [] -> [B] after stack
        }
        return x, y_policy, y_value


def collate_fn(batch: List[Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]]):
    """
    Custom collate that stacks the dict features and supports mixed policy targets.

    Legacy y_policy batching rules (kept for backward-compat):
      - If all entries are class indices (LongTensor with shape [1]), cat -> LongTensor [B]
      - Else if all entries are distributions [45], stack -> FloatTensor [B,45]
      - Else (mixed), convert classes to one-hot distributions and stack -> FloatTensor [B,45]

    New outputs added into x_batch (for mixed training later):
      - x["policy_targets_onehot"]: LongTensor [B] (class index for onehot samples; -100 where mcts)
      - x["policy_targets_mcts"]:   FloatTensor [B,45] (distribution for mcts samples; zeros where onehot)
      - x["policy_mask"]:           BoolTensor [B] (True if mcts, False if onehot)
      - x["off_pv"]:                BoolTensor [B] (True if off_pv)
    """
    xs, yps, yvs = zip(*batch)
    # Stack dict fields
    def stack_key(k: str) -> torch.Tensor:
        return torch.stack([x[k] for x in xs], dim=0)

    x_batch = {
        "board_card_ids": stack_key("board_card_ids"),  # [B,9]
        "board_owner": stack_key("board_owner"),  # [B,9,3]
        "board_element": stack_key("board_element"),  # [B,9,9]
        "hand_card_ids": stack_key("hand_card_ids"),  # [B,5]
        "hand_mask": stack_key("hand_mask"),  # [B,5]
        "rules": stack_key("rules"),  # [B,4]
        "move_mask": stack_key("move_mask"),  # [B,45]
        "off_pv": stack_key("off_pv"),  # [B]
        "policy_is_mcts": stack_key("policy_is_mcts"),  # [B]
    }

    # Build mixed-policy targets for future training
    policy_mask = x_batch["policy_is_mcts"].to(torch.bool)  # [B]
    onehot_targets: List[int] = []
    mcts_targets: List[torch.Tensor] = []
    for i, yp in enumerate(yps):
        if policy_mask[i].item():  # mcts sample
            if yp.dtype == torch.long:
                # Shouldn't happen with correct dataset flags; convert to one-hot distribution if needed
                idx_val = int(yp.item())
                dist = torch.zeros(U.MOVE_SPACE, dtype=torch.float32)
                if 0 <= idx_val < U.MOVE_SPACE:
                    dist[idx_val] = 1.0
                mcts_targets.append(dist)
            else:
                # Ensure float32 and normalized
                p = yp.to(torch.float32)
                s = float(p.sum().item())
                if s > 0:
                    p = p / p.sum()
                mcts_targets.append(p)
            onehot_targets.append(-100)  # ignore in CE for mcts rows
        else:  # onehot sample
            if yp.dtype == torch.long:
                onehot_targets.append(int(yp.item()))
            else:
                # Unexpected; take argmax as class
                onehot_targets.append(int(yp.argmax().item()))
            mcts_targets.append(torch.zeros(U.MOVE_SPACE, dtype=torch.float32))

    x_batch["policy_targets_onehot"] = torch.tensor(onehot_targets, dtype=torch.long)         # [B]
    x_batch["policy_targets_mcts"] = torch.stack(mcts_targets, dim=0)                          # [B,45]
    x_batch["policy_mask"] = policy_mask                                                       # [B]

    # Determine legacy policy format for backward-compat
    is_class = all((yp.dtype == torch.long and yp.dim() == 1 and yp.shape[0] == 1) for yp in yps)
    is_dist = all((yp.dtype in (torch.float32, torch.float64) and yp.dim() == 1 and yp.shape[0] == U.MOVE_SPACE) for yp in yps)

    if is_class:
        y_policy = torch.cat(yps, dim=0)  # [B]
    elif is_dist:
        y_policy = torch.stack(yps, dim=0)  # [B,45]
    else:
        # Mixed: convert any class indices to one-hot distributions
        ys: List[torch.Tensor] = []
        for yp in yps:
            if yp.dtype == torch.long:
                idx_val = int(yp.item())
                if idx_val == -100:
                    # uniform over all moves (shape-only fallback)
                    ys.append(torch.full((U.MOVE_SPACE,), 1.0 / U.MOVE_SPACE, dtype=torch.float32))
                else:
                    vec = torch.zeros(U.MOVE_SPACE, dtype=torch.float32)
                    if 0 <= idx_val < U.MOVE_SPACE:
                        vec[idx_val] = 1.0
                    ys.append(vec)
            else:
                p = yp.to(torch.float32)
                s = float(p.sum().item())
                if s > 0:
                    p = p / p.sum()
                ys.append(p)
        y_policy = torch.stack(ys, dim=0)

    y_value = torch.cat(yvs, dim=0)  # [B]
    return x_batch, y_policy, y_value