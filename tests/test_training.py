from pathlib import Path

import torch

from ccjoker.train import TrainArgs, train_loop


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