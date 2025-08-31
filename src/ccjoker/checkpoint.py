from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Tuple

import torch

from .model import ModelConfig, PolicyValueNet


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_checkpoint(model_path: Path | str, device: str = "cpu", train_mode: bool = False) -> Tuple[PolicyValueNet, ModelConfig, dict]:
    """
    Load a PolicyValueNet checkpoint saved by ccjoker.train.
    Returns: (model, cfg, meta) where meta includes at least epoch/num_cards/move_space.
    """
    map_location = torch.device(device)
    payload = torch.load(str(model_path), map_location=map_location)
    cfg_dict = payload.get("config") or {}
    cfg = ModelConfig(**cfg_dict)
    model = PolicyValueNet(cfg).to(map_location)
    state = payload.get("model_state_dict") or payload.get("state_dict") or {}
    model.load_state_dict(state, strict=True)
    if train_mode:
        model.train()
    else:
        model.eval()
    meta = payload.get("meta", {"config": asdict(cfg)})
    return model, cfg, meta