from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from rich.console import Console
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import typer

from .dataset import TriplecargoDataset, collate_fn
from .model import ModelConfig, PolicyValueNet, mask_policy_logits
from .utils import MOVE_SPACE


app = typer.Typer(add_completion=False)
console = Console()


@dataclass
class TrainArgs:
    data: Path = Path("data/raw/train.jsonl")
    val_data: Optional[Path] = None
    epochs: int = 10
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 0.0
    out: Path = Path("data/models/model.pt")
    val_split: float = 0.1
    num_workers: int = 0
    seed: int = 42
    device: str = "cpu"


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def policy_loss(
    logits: torch.Tensor,  # [B,45]
    y_policy: torch.Tensor,  # [B] long OR [B,45] float
    move_mask: torch.Tensor,  # [B,45] float {0,1}
) -> torch.Tensor:
    # Mask invalid moves
    masked = mask_policy_logits(logits, move_mask)  # [B,45]
    if y_policy.dtype == torch.long and y_policy.dim() == 1:
        # Class CE with ignore index -100
        return nn.functional.cross_entropy(masked, y_policy, ignore_index=-100)
    # Distribution KL: -∑ p * log q
    log_q = nn.functional.log_softmax(masked, dim=-1)
    p = y_policy
    # Normalize p just in case due to masking fallbacks
    p = p / p.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    return -(p * log_q).sum(dim=-1).mean()


def value_loss(
    logits: torch.Tensor,  # [B,3]
    y_value: torch.Tensor,  # [B] long in {0,1,2}
) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, y_value)


@torch.no_grad()
def evaluate(
    model: PolicyValueNet,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total = 0
    policy_loss_sum = 0.0
    value_loss_sum = 0.0
    policy_correct = 0
    value_correct = 0

    for x, y_policy, y_value in loader:
        # Move to device
        xb = {
            k: v.to(device)
            for k, v in x.items()
        }
        yp = y_policy.to(device)
        yv = y_value.to(device)

        policy_logits, value_logits = model(xb)

        # Losses
        pl = policy_loss(policy_logits, yp, xb["move_mask"])
        vl = value_loss(value_logits, yv)
        bs = yv.size(0)
        policy_loss_sum += float(pl.item()) * bs
        value_loss_sum += float(vl.item()) * bs

        # Policy accuracy:
        # - If class labels: compare argmax vs label (ignore -100)
        # - If distribution: compare argmax vs argmax of target distribution
        masked_logits = mask_policy_logits(policy_logits, xb["move_mask"])
        pred_policy = masked_logits.argmax(dim=-1)  # [B]
        if yp.dtype == torch.long and yp.dim() == 1:
            valid = yp != -100
            if valid.any():
                policy_correct += (pred_policy[valid] == yp[valid]).sum().item()
                total += int(valid.sum().item())
        else:
            # distribution case
            tgt = yp.argmax(dim=-1)
            policy_correct += (pred_policy == tgt).sum().item()
            total += bs

        # Value accuracy
        pred_value = value_logits.argmax(dim=-1)
        value_correct += (pred_value == yv).sum().item()

    denom = max(1, len(loader.dataset))
    denom_policy = max(1, total)
    return {
        "policy_loss": policy_loss_sum / denom,
        "value_loss": value_loss_sum / denom,
        "policy_acc": policy_correct / max(1, denom_policy),
        "value_acc": value_correct / denom,
    }


def make_loaders(args: TrainArgs) -> Tuple[TriplecargoDataset, DataLoader, Optional[DataLoader]]:
    ds = TriplecargoDataset(args.data)
    if args.val_data is not None:
        ds_val = TriplecargoDataset(args.val_data)
        train_loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=False,
        )
        val_loader = DataLoader(
            ds_val,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=False,
        )
        return ds, train_loader, val_loader

    # Split train/val
    n = len(ds)
    val_n = max(1, int(math.ceil(n * args.val_split)))
    train_n = max(1, n - val_n)
    ds_train, ds_val = random_split(ds, [train_n, val_n], generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(
        ds_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=False,
    )
    val_loader = DataLoader(
        ds_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=False,
    )
    return ds, train_loader, val_loader


def train_loop(args: TrainArgs) -> None:
    set_seed(args.seed)
    device = torch.device(args.device)

    ds, train_loader, val_loader = make_loaders(args)

    # Model config sizing from dataset
    # num_cards = max_card_id + 2 (padding 0 and max_id+1 index)
    num_cards = ds.max_card_id + 2
    cfg = ModelConfig(num_cards=num_cards)
    model = PolicyValueNet(cfg).to(device)

    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_acc = -1.0
    args.out.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", dynamic_ncols=True)
        running = {"policy_loss": 0.0, "value_loss": 0.0}
        samples = 0

        for x, y_policy, y_value in pbar:
            xb = {k: v.to(device) for k, v in x.items()}
            yp = y_policy.to(device)
            yv = y_value.to(device)

            opt.zero_grad()

            policy_logits, value_logits = model(xb)
            pl = policy_loss(policy_logits, yp, xb["move_mask"])
            vl = value_loss(value_logits, yv)
            loss = pl + vl

            loss.backward()
            opt.step()

            bs = yv.size(0)
            samples += bs
            running["policy_loss"] += float(pl.item()) * bs
            running["value_loss"] += float(vl.item()) * bs

            pbar.set_postfix({
                "pl": f"{running['policy_loss']/max(1,samples):.4f}",
                "vl": f"{running['value_loss']/max(1,samples):.4f}",
            })

        # Validation
        metrics = evaluate(model, val_loader, device)
        console.log(f"[bold]Val:[/bold] "
                    f"pl={metrics['policy_loss']:.4f} "
                    f"vl={metrics['value_loss']:.4f} "
                    f"pacc={metrics['policy_acc']:.3f} "
                    f"vacc={metrics['value_acc']:.3f}")

        # Checkpoint if improved policy accuracy
        if metrics["policy_acc"] > best_val_acc:
            best_val_acc = metrics["policy_acc"]
            payload = {
                "model_state_dict": model.state_dict(),
                "config": asdict(cfg),
                "meta": {
                    "epoch": epoch,
                    "num_cards": num_cards,
                    "move_space": MOVE_SPACE,
                },
            }
            torch.save(payload, args.out)
            console.log(f"Saved checkpoint to {args.out}")

    # Final save at the end regardless
    payload = {
        "model_state_dict": model.state_dict(),
        "config": asdict(cfg),
        "meta": {
            "epoch": args.epochs,
            "num_cards": num_cards,
            "move_space": MOVE_SPACE,
        },
    }
    torch.save(payload, args.out)
    console.log(f"Final checkpoint saved to {args.out}")


@app.command()
def main(
    data: Path = typer.Option(Path("data/raw/train.jsonl"), "--data", help="Training JSONL file"),
    val_data: Optional[Path] = typer.Option(None, "--val-data", help="Optional validation JSONL file"),
    epochs: int = typer.Option(10, "--epochs", min=1),
    batch_size: int = typer.Option(64, "--batch-size", min=1),
    lr: float = typer.Option(1e-3, "--lr"),
    weight_decay: float = typer.Option(0.0, "--weight-decay"),
    out: Path = typer.Option(Path("data/models/model.pt"), "--out", help="Output model checkpoint path"),
    val_split: float = typer.Option(0.1, "--val-split", min=0.01, max=0.5, help="Train/val split when no val-data"),
    num_workers: int = typer.Option(0, "--num-workers"),
    seed: int = typer.Option(42, "--seed"),
    device: str = typer.Option("cpu", "--device"),
) -> None:
    """
    Train policy/value net on Triplecargo JSONL data.
    """
    args = TrainArgs(
        data=data,
        val_data=val_data,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        out=out,
        val_split=val_split,
        num_workers=num_workers,
        seed=seed,
        device=device,
    )
    train_loop(args)


if __name__ == "__main__":
    main()