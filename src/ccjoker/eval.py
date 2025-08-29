from __future__ import annotations

from pathlib import Path

import torch
from rich.console import Console
from torch.utils.data import DataLoader
import typer

from .dataset import TriplecargoDataset, collate_fn
from .model import ModelConfig, PolicyValueNet
from .train import evaluate  # reuse eval routine from training


app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def main(
    data: Path = typer.Option(Path("data/raw/val.jsonl"), "--data", help="Evaluation JSONL file"),
    model_path: Path = typer.Option(Path("data/models/model.pt"), "--model", help="Model checkpoint path"),
    batch_size: int = typer.Option(128, "--batch-size", min=1),
    num_workers: int = typer.Option(0, "--num-workers"),
    device: str = typer.Option("cpu", "--device"),
) -> None:
    """
    Evaluate a trained policy/value model on a JSONL dataset.
    Reports policy accuracy (top-1), value accuracy, and their losses.
    """
    console.log(f"Loading dataset from {data}")
    ds = TriplecargoDataset(data)

    console.log(f"Preparing DataLoader (batch_size={batch_size})")
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=False,
    )

    device_t = torch.device(device)

    # Build model sized to dataset card vocabulary
    num_cards = ds.max_card_id + 2
    cfg = ModelConfig(num_cards=num_cards)
    model = PolicyValueNet(cfg).to(device_t)

    console.log(f"Loading checkpoint from {model_path}")
    payload = torch.load(model_path, map_location=device_t)
    state_dict = payload.get("model_state_dict", payload)
    model.load_state_dict(state_dict, strict=False)

    metrics = evaluate(model, loader, device_t)
    console.rule("[bold green]Evaluation Results[/bold green]")
    console.print(
        f"policy_loss={metrics['policy_loss']:.4f}  "
        f"value_loss={metrics['value_loss']:.4f}  "
        f"policy_acc={metrics['policy_acc']:.3f}  "
        f"value_acc={metrics['value_acc']:.3f}"
    )


if __name__ == "__main__":
    main()