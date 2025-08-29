from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn

from .utils import HAND_SLOTS, MOVE_SPACE, NUM_CELLS


@dataclass
class ModelConfig:
    num_cards: int  # number of embedding entries (dataset.max_card_id + 2)
    card_embed_dim: int = 32
    hidden_dim: int = 256
    dropout: float = 0.1


class PolicyValueNet(nn.Module):
    """
    Small MLP with shared trunk over simple encodings.

    Inputs (per-sample):
      - board_card_ids: LongTensor [9] (embedding indices; 0 is padding for empty cell)
      - board_owner:    FloatTensor [9,3]
      - board_element:  FloatTensor [9,9]
      - hand_card_ids:  LongTensor [5] (embedding indices; 0 is padding for empty slot)
      - rules:          FloatTensor [4]

    The forward API expects a batch dict with shapes:
      - board_card_ids: [B,9]
      - board_owner:    [B,9,3]
      - board_element:  [B,9,9]
      - hand_card_ids:  [B,5]
      - rules:          [B,4]

    Outputs:
      - policy_logits: [B,45]
      - value_logits:  [B,3]   (win/draw/loss logits in order [loss, draw, win])
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # Embeddings for cards (board cells + hand slots share)
        self.card_embed = nn.Embedding(cfg.num_cards, cfg.card_embed_dim, padding_idx=0)

        # Infer flat input dimension
        board_vec_per_cell = 3 + 9 + cfg.card_embed_dim  # owner(3) + element(9) + card_emb
        board_flat_dim = NUM_CELLS * board_vec_per_cell   # 9 * per-cell
        hand_flat_dim = HAND_SLOTS * cfg.card_embed_dim   # 5 * emb
        rules_dim = 4
        in_dim = board_flat_dim + hand_flat_dim + rules_dim

        self.trunk = nn.Sequential(
            nn.Linear(in_dim, cfg.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=cfg.dropout),
        )
        self.policy_head = nn.Linear(cfg.hidden_dim, MOVE_SPACE)  # 45 logits
        self.value_head = nn.Linear(cfg.hidden_dim, 3)            # 3 logits

        self._init_params()

    def _init_params(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.normal_(self.card_embed.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.card_embed.weight[0].fill_(0.0)  # padding stays zero

    def forward(self, x: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x is a dict of batched tensors with keys:
          - board_card_ids [B,9] long
          - board_owner    [B,9,3] float
          - board_element  [B,9,9] float
          - hand_card_ids  [B,5] long
          - rules          [B,4] float
        """
        b_card_ids: torch.Tensor = x["board_card_ids"]  # [B,9]
        b_owner: torch.Tensor = x["board_owner"]        # [B,9,3]
        b_element: torch.Tensor = x["board_element"]    # [B,9,9]
        h_card_ids: torch.Tensor = x["hand_card_ids"]   # [B,5]
        rules: torch.Tensor = x["rules"]                # [B,4]

        # Embeddings
        b_card_emb = self.card_embed(b_card_ids)        # [B,9,E]
        h_card_emb = self.card_embed(h_card_ids)        # [B,5,E]

        # Concatenate per-cell owner/element/emb
        board_cat = torch.cat([b_owner, b_element, b_card_emb], dim=-1)  # [B,9, 3+9+E]
        board_flat = board_cat.reshape(board_cat.size(0), -1)            # [B, 9*(3+9+E)]
        hand_flat = h_card_emb.reshape(h_card_emb.size(0), -1)           # [B, 5*E]

        feat = torch.cat([board_flat, hand_flat, rules], dim=-1)         # [B, in_dim]
        h = self.trunk(feat)
        policy_logits = self.policy_head(h)                               # [B,45]
        value_logits = self.value_head(h)                                 # [B,3]
        return policy_logits, value_logits


def mask_policy_logits(logits: torch.Tensor, move_mask: torch.Tensor) -> torch.Tensor:
    """
    Apply an additive mask to policy logits to invalidate illegal moves.
    - logits: [B,45]
    - move_mask: [B,45] with 1 for valid moves and 0 for invalid moves
    Returns masked logits with invalid positions set to -inf (or a large negative).
    """
    # Use a large negative where mask==0
    large_neg = torch.finfo(logits.dtype).min / 2
    masked = logits.masked_fill(move_mask <= 0, large_neg)
    return masked