from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import torch


# Encoding domains
OWNER_INDEX = {"A": 0, "B": 1, None: 2}
OWNER_DIM = 3

ELEMENTS: List[Optional[str]] = ["F", "I", "T", "W", "E", "P", "H", "L", None]
ELEMENT_INDEX = {e: i for i, e in enumerate(ELEMENTS)}
ELEMENT_DIM = len(ELEMENTS)  # 9

# Board/hand sizes
NUM_CELLS = 9
HAND_SLOTS = 5
MOVE_SPACE = HAND_SLOTS * NUM_CELLS  # 45


def owner_onehot(owner: Optional[str]) -> torch.Tensor:
    """
    Map owner symbol {"A","B",None} to one-hot [3].
    """
    idx = OWNER_INDEX.get(owner, OWNER_INDEX[None])
    return torch.nn.functional.one_hot(torch.tensor(idx), num_classes=OWNER_DIM).to(torch.float32)


def element_onehot(element: Optional[str]) -> torch.Tensor:
    """
    Map element symbol to one-hot [9]. None maps to the last slot.
    """
    idx = ELEMENT_INDEX.get(element, ELEMENT_INDEX[None])
    return torch.nn.functional.one_hot(torch.tensor(idx), num_classes=ELEMENT_DIM).to(torch.float32)


def card_id_to_embed_index(card_id: Optional[int]) -> int:
    """
    Map card_id (int or None) to an embedding index with padding_idx=0.
    None or empty cell -> 0; otherwise card_id+1.
    """
    if card_id is None:
        return 0
    return int(card_id) + 1


def pad_hand_ids(hand_ids: Iterable[int], pad_to: int = HAND_SLOTS) -> Tuple[List[int], List[int]]:
    """
    Pad/truncate a list of card_ids to fixed HAND_SLOTS.
    Returns:
      - embed indices list[int] length pad_to (card_id+1, 0 for pad)
      - mask list[int] length pad_to (1 for real, 0 for pad)
    """
    ids = list(hand_ids)
    ids = ids[:pad_to]
    mask = [1] * len(ids)
    while len(ids) < pad_to:
        ids.append(None)  # pad
        mask.append(0)
    embed_ids = [card_id_to_embed_index(x) for x in ids]
    return embed_ids, mask


def slot_for_card(hand_ids: List[int], card_id: int) -> Optional[int]:
    """
    Find the slot index (0..HAND_SLOTS-1) for card_id within the current hand ids list.
    Returns None if not found.
    """
    for i, cid in enumerate(hand_ids):
        if cid == card_id:
            return i
    return None


def move_index(slot_idx: int, cell_idx: int) -> int:
    """
    Map (hand_slot 0..4, cell 0..8) to flat 0..44.
    """
    return slot_idx * NUM_CELLS + cell_idx


def decode_move_index(idx: int) -> Tuple[int, int]:
    """
    Inverse of move_index. Returns (slot_idx, cell_idx).
    """
    return idx // NUM_CELLS, idx % NUM_CELLS


def onehot_policy_to_index(hand: List[int], policy_target: Dict[str, int]) -> Optional[int]:
    """
    Convert onehot policy_target {card_id:int, cell:int} into class index 0..44 given the current hand (raw ids).
    Returns None if the referenced card is not in the current hand.
    """
    card_id = int(policy_target["card_id"])
    cell = int(policy_target["cell"])
    slot = slot_for_card(hand, card_id)
    if slot is None:
        return None
    return move_index(slot, cell)


def mcts_policy_to_probs(hand: List[int], mcts: Dict[str, float]) -> torch.Tensor:
    """
    Convert an MCTS dict like {"19-7": prob, ...} into a [45] probability vector aligned to (slot,cell).
    Any entries whose card_id is not present in the hand are dropped.
    The result is renormalized to sum to 1 if total > 0; otherwise a uniform over valid hand slots × cells.
    """
    probs = torch.zeros(MOVE_SPACE, dtype=torch.float32)
    total = 0.0
    # build mapping card_id -> slot once
    id_to_slot: Dict[int, int] = {cid: i for i, cid in enumerate(hand)}
    for key, p in mcts.items():
        try:
            card_str, cell_str = key.split("-", 1)
            cid = int(card_str)
            cell = int(cell_str)
        except Exception:
            continue
        slot = id_to_slot.get(cid)
        if slot is None or not (0 <= cell < NUM_CELLS):
            continue
        idx = move_index(slot, cell)
        probs[idx] = float(p)
        total += float(p)
    if total > 0:
        probs /= probs.sum()
        return probs

    # If no mass (e.g., all dropped), fall back to uniform over valid slots × cells
    valid_slots = [i for i, cid in enumerate(hand) if cid is not None]
    if not valid_slots:
        # no valid moves; uniform over all 45 to be safe
        probs.fill_(1.0 / MOVE_SPACE)
        return probs
    count = len(valid_slots) * NUM_CELLS
    for s in valid_slots:
        start = s * NUM_CELLS
        probs[start : start + NUM_CELLS] = 1.0 / count
    return probs


def value_to_class(value_mode: str, value_target: int) -> int:
    """
    Map value_target to 3-class label: loss=0, draw=1, win=2.
    - winloss: {-1,0,+1} -> {0,1,2}
    - margin: [-9..+9] -> sign-based class
    """
    if value_mode == "winloss":
        mapping = {-1: 0, 0: 1, 1: 2}
        if value_target not in mapping:
            # clamp
            vt = max(-1, min(1, int(value_target)))
            return mapping[vt]
        return mapping[int(value_target)]
    # margin
    v = int(value_target)
    if v < 0:
        return 0
    if v > 0:
        return 2
    return 1


def rules_to_tensor(rules: Dict[str, bool]) -> torch.Tensor:
    """
    Encode rules into [4] float tensor in the order:
    [elemental, same, plus, same_wall]
    Missing keys default to False.
    """
    elem = bool(rules.get("elemental", False))
    same = bool(rules.get("same", False))
    plus = bool(rules.get("plus", False))
    same_wall = bool(rules.get("same_wall", False))
    return torch.tensor([elem, same, plus, same_wall], dtype=torch.float32)


def build_move_mask(hand_mask: Iterable[int]) -> torch.Tensor:
    """
    Given hand_mask [5] with 1=valid slot, 0=pad, return [45] mask of 1 for valid moves and 0 otherwise.
    """
    hm = torch.tensor(list(hand_mask), dtype=torch.float32)
    mask_slots = hm.view(HAND_SLOTS, 1).repeat(1, NUM_CELLS)  # [5,9]
    return mask_slots.reshape(-1)  # [45]