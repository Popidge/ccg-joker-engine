from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from rich.console import Console
from rich.table import Table
from torch.utils.data import DataLoader
import typer

from .dataset import TriplecargoDataset, collate_fn
from .model import ModelConfig, PolicyValueNet, mask_policy_logits


app = typer.Typer(add_completion=False)
console = Console()


@dataclass
class RawMetrics:
    n: int = 0  # number of samples in slice
    top_hits: Dict[int, int] = field(default_factory=lambda: {1: 0, 2: 0, 3: 0})
    top_denom: int = 0  # denominator for top-k (valid onehot + all mcts rows)
    kl_sum: float = 0.0
    kl_count: int = 0
    value_hits: int = 0

    def add(self, other: "RawMetrics") -> None:
        self.n += other.n
        self.top_denom += other.top_denom
        for k in (1, 2, 3):
            self.top_hits[k] += other.top_hits[k]
        self.kl_sum += other.kl_sum
        self.kl_count += other.kl_count
        self.value_hits += other.value_hits


def _compute_batch_slice_metrics(
    policy_logits: torch.Tensor,
    value_logits: torch.Tensor,
    x_batch: Dict[str, torch.Tensor],
    y_value: torch.Tensor,
    row_mask: torch.Tensor,  # [B] bool
) -> RawMetrics:
    """
    Compute metrics for a subset (slice) of a batch selected by row_mask.

    - top-k accuracy (k=1,2,3) over:
        * onehot rows with target != -100
        * mcts rows using argmax of target distribution
    - KL divergence only over mcts rows
    - value 3-class accuracy over all rows in the slice
    """
    stats = RawMetrics()

    if row_mask.dim() != 1:
        row_mask = row_mask.view(-1)
    if row_mask.numel() == 0 or not bool(row_mask.any()):
        return stats

    idxs = row_mask.nonzero(as_tuple=False).flatten()

    pl = policy_logits.index_select(0, idxs)          # [Ns,45]
    vl = value_logits.index_select(0, idxs)           # [Ns,3]
    mv = x_batch["move_mask"].index_select(0, idxs)   # [Ns,45]
    onehot_t = x_batch["policy_targets_onehot"].index_select(0, idxs)  # [Ns]
    mcts_t = x_batch["policy_targets_mcts"].index_select(0, idxs)      # [Ns,45]
    p_mask = x_batch["policy_mask"].index_select(0, idxs)              # [Ns] True if MCTS
    yv = y_value.index_select(0, idxs)                                  # [Ns]

    stats.n += int(idxs.numel())

    # Value accuracy (all rows)
    pred_value = vl.argmax(dim=-1)
    stats.value_hits += int((pred_value == yv).sum().item())

    # Policy: masked logits
    masked = mask_policy_logits(pl, mv)  # [Ns,45]

    # Prepare masks for subsets
    is_mcts = p_mask.to(torch.bool)
    is_onehot = ~is_mcts
    valid_onehot = is_onehot & (onehot_t != -100)

    # Denominator for top-k
    n_onehot_valid = int(valid_onehot.sum().item())
    n_mcts = int(is_mcts.sum().item())
    stats.top_denom += n_onehot_valid + n_mcts

    if n_onehot_valid + n_mcts > 0:
        # Compute top-3 once; derive top-1/top-2 from that
        top3_idx = masked.topk(k=3, dim=-1).indices  # [Ns,3]

        # Onehot rows
        if n_onehot_valid > 0:
            gt_onehot = onehot_t[valid_onehot]  # [N1]
            t3_onehot = top3_idx.index_select(0, valid_onehot.nonzero(as_tuple=False).flatten())  # [N1,3]
            # k=1,2,3
            for k in (1, 2, 3):
                hits_k = (t3_onehot[:, :k] == gt_onehot.view(-1, 1)).any(dim=-1).sum().item()
                stats.top_hits[k] += int(hits_k)

        # MCTS rows (use argmax of target distribution)
        if n_mcts > 0:
            idx_mcts = is_mcts.nonzero(as_tuple=False).flatten()
            t3_mcts = top3_idx.index_select(0, idx_mcts)  # [N2,3]
            p = mcts_t.index_select(0, idx_mcts)          # [N2,45]
            gt_mcts = p.argmax(dim=-1)                    # [N2]
            for k in (1, 2, 3):
                hits_k = (t3_mcts[:, :k] == gt_mcts.view(-1, 1)).any(dim=-1).sum().item()
                stats.top_hits[k] += int(hits_k)

            # KL(p || q) over mcts rows
            with torch.no_grad():
                eps = 1e-8
                m_logits = masked.index_select(0, idx_mcts)        # [N2,45]
                q = torch.softmax(m_logits, dim=-1).clamp(min=eps)  # [N2,45]
                p = p / p.sum(dim=-1, keepdim=True).clamp(min=eps)  # normalize p
                log_p = torch.where(p > 0, torch.log(p), torch.zeros_like(p))
                log_q = torch.where(p > 0, torch.log(q), torch.zeros_like(q))
                per_row_kl = (p * (log_p - log_q)).sum(dim=-1)      # [N2]
                stats.kl_sum += float(per_row_kl.sum().item())
                stats.kl_count += int(per_row_kl.numel())

    return stats


def compute_batch_metrics(
    policy_logits: torch.Tensor,
    value_logits: torch.Tensor,
    x_batch: Dict[str, torch.Tensor],
    y_value: torch.Tensor,
) -> Dict[str, RawMetrics]:
    """
    Public helper for unit tests:
    Compute per-slice metrics for a single batch and return raw accumulators.

    Returns mapping: {"pv": RawMetrics, "off_pv": RawMetrics, "all": RawMetrics}
    """
    B = y_value.size(0)
    off_mask = x_batch["off_pv"].to(torch.bool)  # [B]
    all_mask = torch.ones(B, dtype=torch.bool, device=y_value.device)

    out = {"pv": RawMetrics(), "off_pv": RawMetrics(), "all": RawMetrics()}

    pv_stats = _compute_batch_slice_metrics(policy_logits, value_logits, x_batch, y_value, ~off_mask)
    off_stats = _compute_batch_slice_metrics(policy_logits, value_logits, x_batch, y_value, off_mask)
    all_stats = _compute_batch_slice_metrics(policy_logits, value_logits, x_batch, y_value, all_mask)

    out["pv"] = pv_stats
    out["off_pv"] = off_stats
    out["all"] = all_stats
    return out


@torch.no_grad()
def evaluate_loader_sliced(
    model: PolicyValueNet,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, RawMetrics]:
    """
    Evaluate a loader and accumulate raw metrics per slice:
    - 'pv'     → off_pv == False
    - 'off_pv' → off_pv == True
    - 'all'    → all rows
    """
    model.eval()
    totals = {"pv": RawMetrics(), "off_pv": RawMetrics(), "all": RawMetrics()}

    for x, _y_policy_legacy, y_value in loader:
        xb = {k: v.to(device) for k, v in x.items()}
        yv = y_value.to(device)

        policy_logits, value_logits = model(xb)

        batch_stats = compute_batch_metrics(policy_logits, value_logits, xb, yv)
        for key in ("pv", "off_pv", "all"):
            totals[key].add(batch_stats[key])

    return totals


def finalize_metrics(r: RawMetrics) -> Dict[str, Optional[float]]:
    """
    Convert RawMetrics to finalized rates.
    Returns dict with: n_samples, top1, top2, top3, kl, value_acc
    """
    def ratio(num: int, den: int) -> Optional[float]:
        return (num / den) if den > 0 else None

    return {
        "n_samples": r.n,
        "top1": ratio(r.top_hits[1], r.top_denom),
        "top2": ratio(r.top_hits[2], r.top_denom),
        "top3": ratio(r.top_hits[3], r.top_denom),
        "kl": (r.kl_sum / r.kl_count) if r.kl_count > 0 else None,
        "value_acc": ratio(r.value_hits, r.n),
    }


def _print_results_table(tag_to_final: Dict[str, Dict[str, Optional[float]]]) -> None:
    table = Table(title="Evaluation Results", show_edge=True)
    table.add_column("slice", justify="left")
    table.add_column("n", justify="right")
    table.add_column("top1", justify="right")
    table.add_column("top2", justify="right")
    table.add_column("top3", justify="right")
    table.add_column("kl", justify="right")
    table.add_column("value_acc", justify="right")

    def fmt(x: Optional[float], prec: int = 3) -> str:
        if x is None:
            return "-"
        return f"{x:.{prec}f}"

    for tag in ["pv", "off_pv", "all"]:
        if tag not in tag_to_final:
            continue
        f = tag_to_final[tag]
        table.add_row(
            tag,
            str(f["n_samples"]),
            fmt(f["top1"]),
            fmt(f["top2"]),
            fmt(f["top3"]),
            fmt(f["kl"], prec=6),
            fmt(f["value_acc"]),
        )

    console.print(table)


def _append_csv_rows(
    csv_path: Path,
    model_path: Path,
    data_tag_rows: List[Tuple[str, Dict[str, Optional[float]]]],
) -> None:
    """
    Append rows to CSV with columns:
    timestamp, model, data_tag, n_samples, top1, top2, top3, kl, value_acc
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    header = ["timestamp", "model", "data_tag", "n_samples", "top1", "top2", "top3", "kl", "value_acc"]
    now = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        for data_tag, metrics in data_tag_rows:
            row = [
                now,
                str(model_path),
                data_tag,
                int(metrics["n_samples"] or 0),
                "" if metrics["top1"] is None else f"{metrics['top1']:.6f}",
                "" if metrics["top2"] is None else f"{metrics['top2']:.6f}",
                "" if metrics["top3"] is None else f"{metrics['top3']:.6f}",
                "" if metrics["kl"] is None else f"{metrics['kl']:.6f}",
                "" if metrics["value_acc"] is None else f"{metrics['value_acc']:.6f}",
            ]
            writer.writerow(row)


def _make_loader(path: Path, batch_size: int, num_workers: int) -> Tuple[TriplecargoDataset, DataLoader]:
    ds = TriplecargoDataset(path)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=False,
    )
    return ds, loader


@app.command()
def main(
    data: Path = typer.Option(Path("data/raw/val.jsonl"), "--data", help="Evaluation JSONL file (single-dataset mode)"),
    data_pv: Optional[Path] = typer.Option(None, "--data-pv", help="PV dataset path (when off_pv flag absent)"),
    data_mixed: Optional[Path] = typer.Option(None, "--data-mixed", help="Off-PV/mixed dataset path (when off_pv flag absent)"),
    model_path: Path = typer.Option(Path("data/models/model.pt"), "--model", help="Model checkpoint path"),
    batch_size: int = typer.Option(128, "--batch-size", min=1),
    num_workers: int = typer.Option(0, "--num-workers"),
    device: str = typer.Option("cpu", "--device"),
    csv_out: Path = typer.Option(Path("data/processed/eval_metrics.csv"), "--csv-out", help="CSV output path"),
) -> None:
    """
    Evaluate a trained policy/value model with richer metrics and CSV export.

    Metrics:
    - Policy: top-1/top-2/top-3 accuracy; KL divergence (only for MCTS targets)
    - Value:  3-class accuracy
    Slicing:
    - If data provides off_pv, report PV (off_pv=false), off-PV (off_pv=true), and ALL.
    - Otherwise, you may provide --data-pv and --data-mixed to evaluate and aggregate.
    """
    device_t = torch.device(device)

    # Prepare datasets/loaders and determine model sizing
    loaders: List[Tuple[str, DataLoader, TriplecargoDataset]] = []
    ds_list: List[TriplecargoDataset] = []

    if data_pv is not None or data_mixed is not None:
        if data_pv is None or data_mixed is None:
            raise typer.BadParameter("Both --data-pv and --data-mixed must be provided together.")
        console.log(f"Loading PV dataset from {data_pv}")
        ds_pv, loader_pv = _make_loader(data_pv, batch_size, num_workers)
        console.log(f"Loading mixed/off-PV dataset from {data_mixed}")
        ds_mixed, loader_mixed = _make_loader(data_mixed, batch_size, num_workers)
        loaders.append(("pv_all", loader_pv, ds_pv))
        loaders.append(("off_pv_all", loader_mixed, ds_mixed))
        ds_list.extend([ds_pv, ds_mixed])
    else:
        console.log(f"Loading dataset from {data}")
        ds, loader = _make_loader(data, batch_size, num_workers)
        loaders.append(("single", loader, ds))
        ds_list.append(ds)

    # Build model sized to max card vocabulary across datasets
    num_cards = max(ds.max_card_id for ds in ds_list) + 2
    cfg = ModelConfig(num_cards=num_cards)
    model = PolicyValueNet(cfg).to(device_t)

    console.log(f"Loading checkpoint from {model_path}")
    payload = torch.load(model_path, map_location=device_t)
    state_dict = payload.get("model_state_dict", payload)
    model.load_state_dict(state_dict, strict=False)

    # Evaluate
    tag_to_raw: Dict[str, RawMetrics] = {}
    if len(loaders) == 1 and loaders[0][0] == "single":
        # Single dataset mode: use slicing by off_pv within the loader
        _, loader, _ = loaders[0]
        raw = evaluate_loader_sliced(model, loader, device_t)
        tag_to_raw["pv"] = raw["pv"]
        tag_to_raw["off_pv"] = raw["off_pv"]
        tag_to_raw["all"] = raw["all"]
        base_name = Path(data).name
        data_tags = {
            "pv": f"{base_name}#pv",
            "off_pv": f"{base_name}#off_pv",
            "all": f"{base_name}#all",
        }
    else:
        # Two-dataset mode: treat each loader as a whole slice, then aggregate
        raw_agg = RawMetrics()
        base_name_pv = Path(data_pv).name if data_pv else "pv"
        base_name_off = Path(data_mixed).name if data_mixed else "off_pv"

        for ltag, loader, _ds in loaders:
            raw_all = evaluate_loader_sliced(model, loader, device_t)["all"]
            if ltag == "pv_all":
                tag_to_raw["pv"] = raw_all
            elif ltag == "off_pv_all":
                tag_to_raw["off_pv"] = raw_all
            raw_agg.add(raw_all)

        tag_to_raw["all"] = raw_agg
        data_tags = {
            "pv": f"{base_name_pv}#pv",
            "off_pv": f"{base_name_off}#off_pv",
            "all": "aggregate",
        }

    # Finalize metrics to ratios
    tag_to_final: Dict[str, Dict[str, Optional[float]]] = {
        tag: finalize_metrics(rm) for tag, rm in tag_to_raw.items()
    }

    # Report
    _print_results_table(tag_to_final)

    # CSV rows
    rows_for_csv: List[Tuple[str, Dict[str, Optional[float]]]] = []
    # Write slices that have any samples, plus aggregate
    for tag in ("pv", "off_pv", "all"):
        if tag not in tag_to_final:
            continue
        if tag != "all" and tag_to_final[tag]["n_samples"] == 0:
            continue
        rows_for_csv.append((data_tags[tag], tag_to_final[tag]))

    _append_csv_rows(csv_out, model_path, rows_for_csv)

    # Also print a compact one-line summary for quick scanning
    f_all = tag_to_final["all"]
    summary = f"ALL n={int(f_all['n_samples'] or 0)} "
    if f_all["top1"] is not None:
        summary += f"top1={f_all['top1']:.3f}"
    else:
        summary += "top1=-"
    console.print(summary, justify="left")


if __name__ == "__main__":
    main()