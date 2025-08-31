from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from .train import TrainArgs, train_loop

app = typer.Typer(add_completion=False)


@app.command()
def main(
    data: Path = typer.Option(Path("data/raw/selfplay.jsonl"), "--data", help="Self-play JSONL file"),
    val_data: Optional[Path] = typer.Option(None, "--val-data", help="Optional validation JSONL file"),
    epochs: int = typer.Option(5, "--epochs", min=1),
    batch_size: int = typer.Option(256, "--batch-size", min=1),
    lr: float = typer.Option(1e-3, "--lr"),
    weight_decay: float = typer.Option(0.0, "--weight-decay"),
    out: Path = typer.Option(Path("data/models/model_rl.pt"), "--out", help="Output model checkpoint path"),
    val_split: float = typer.Option(0.1, "--val-split", min=0.01, max=0.5, help="Train/val split when no val-data"),
    num_workers: int = typer.Option(0, "--num-workers"),
    seed: int = typer.Option(42, "--seed"),
    device: str = typer.Option("cpu", "--device"),
    value_loss_weight: float = typer.Option(0.5, "--value-loss-weight", help="Weight for value loss in total loss"),
    amp: Optional[bool] = typer.Option(
        None,
        "--amp/--no-amp",
        help="Enable AMP (mixed precision). Default: on for CUDA, off for CPU.",
        show_default=False,
    ),
    amp_debug: bool = typer.Option(
        False,
        "--amp-debug",
        help="If set, log GradScaler scale periodically when AMP is enabled.",
    ),
) -> None:
    """
    Train policy/value net on self-play JSONL data (MCTS distributions supported).
    Uses the same training loop and mixed-loss behavior as ccj-train.
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
        value_loss_weight=value_loss_weight,
        amp=amp,
        amp_debug=amp_debug,
    )
    train_loop(args)


if __name__ == "__main__":
    main()