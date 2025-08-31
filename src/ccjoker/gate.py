from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import typer
from rich.console import Console

from .checkpoint import load_checkpoint, set_seed
from .env import TripleTriadEnv
from .mcts import MCTS
from .model import PolicyValueNet, mask_policy_logits
from . import utils as U

app = typer.Typer(add_completion=False)
console = Console()


def _parse_rules(rules_str: str) -> Dict[str, bool]:
    rules_str = (rules_str or "").strip().lower()
    flags = {"elemental": False, "same": False, "plus": False, "same_wall": False}
    if rules_str in ("", "none", "no", "off"):
        return flags
    parts = [p.strip() for p in rules_str.split(",") if p.strip()]
    for p in parts:
        if p in flags:
            flags[p] = True
    return flags


@torch.no_grad()
def select_move_greedy(net: PolicyValueNet, device: torch.device, env: TripleTriadEnv, state: dict) -> int:
    """
    Greedy argmax over masked policy logits.
    """
    x = env.encode(state)
    xb = {k: v.unsqueeze(0).to(device) for k, v in x.items()}
    policy_logits, _ = net(xb)
    masked = mask_policy_logits(policy_logits, xb["move_mask"])
    a = int(masked.argmax(dim=-1)[0].item())
    return a


def elo_delta_from_score(score: float, min_score: float = 1e-6, max_score: float = 1 - 1e-6) -> float:
    """
    Convert match score s in (0,1) to Elo delta via logistic link:
      s = 1 / (1 + 10^(-d/400))  =>  d = 400 * log10(s/(1-s))
    Draws counted as 0.5 in s.
    """
    import math

    s = max(min_score, min(max_score, float(score)))
    return 400.0 * math.log10(s / (1.0 - s))


@app.command()
def main(
    a: Path = typer.Option(..., "--a", help="Baseline/old model checkpoint"),
    b: Path = typer.Option(..., "--b", help="Candidate/new model checkpoint"),
    games: int = typer.Option(20, "--games", min=1),
    device: str = typer.Option("cpu", "--device", help="cpu|cuda"),
    rollouts: int = typer.Option(0, "--rollouts", min=0, help="MCTS rollouts per move; 0=greedy argmax"),
    temperature: float = typer.Option(0.25, "--temperature", help="MCTS sampling temperature"),
    seed: Optional[int] = typer.Option(123, "--seed"),
    rules: str = typer.Option("none", "--rules"),
    triplecargo_cmd: Optional[Path] = typer.Option(
        None, "--triplecargo-cmd", help="Path to Triplecargo precompute.exe (with --eval-state)"
    ),
    cards: Optional[Path] = typer.Option(None, "--cards", help="Path to cards.json for Triplecargo"),
    use_stub: bool = typer.Option(False, "--use-stub/--no-use-stub", help="Use Python stub env"),
    threshold: float = typer.Option(0.55, "--threshold", min=0.5, max=1.0, help="Promotion threshold as score vs A"),
) -> None:
    """
    Head-to-head gating: play matches A vs B; report W/D/L and Elo delta for B relative to A.
    """
    if seed is not None:
        set_seed(seed)

    # Load models
    net_a, cfg_a, meta_a = load_checkpoint(a, device=device, train_mode=False)
    net_b, cfg_b, meta_b = load_checkpoint(b, device=device, train_mode=False)

    # Env (cap card IDs in stub to smallest model capacity to avoid OOB)
    rules_dict = _parse_rules(rules)
    max_card_id = None
    if use_stub:
        try:
            max_card_id = min(int(cfg_a.num_cards) - 2, int(cfg_b.num_cards) - 2)
        except Exception:
            max_card_id = None
    env = TripleTriadEnv(
        rules=rules_dict,
        cards_path=str(cards) if cards else "data/cards.json",
        triplecargo_cmd=str(triplecargo_cmd) if triplecargo_cmd else "C:/Users/popid/Dev/triplecargo/target/release/precompute.exe",
        seed=seed,
        use_stub=use_stub,
        max_card_id=max_card_id,
    )

    device_t = torch.device(device)
    net_a.eval().to(device_t)
    net_b.eval().to(device_t)

    # MCTS (if enabled)
    mcts_a = MCTS(net=net_a, device=device, rollouts=rollouts, c_puct=1.0, temperature=temperature) if rollouts > 0 else None
    mcts_b = MCTS(net=net_b, device=device, rollouts=rollouts, c_puct=1.0, temperature=temperature) if rollouts > 0 else None

    a_wins = 0
    b_wins = 0
    draws = 0

    for g in range(games):
        # Alternate first player: even -> A starts; odd -> B starts
        first = "A" if (g % 2 == 0) else "B"
        # Reset with deterministic per-game seed
        state = env.reset(seed=None if seed is None else (seed + g))
        # Ensure desired starting side
        if state.get("to_move") != first:
            state = _swap_sides(state)

        # Fixed 9 plies max; break earlier if terminal or no legal moves
        outcome = None
        for turn in range(9):
            # If no legal moves, draw
            root_mask = env.legal_moves(state)
            if float(root_mask.sum().item()) <= 0.0:
                outcome = {"winner": None}
                break

            to_move = state.get("to_move", "A")
            # Decide move via greedy or MCTS with the corresponding net
            if rollouts <= 0:
                net = net_a if to_move == "A" else net_b
                a_idx = select_move_greedy(net, device_t, env, state)
            else:
                mcts = mcts_a if to_move == "A" else mcts_b
                result = mcts.run(env, state)
                a_idx = int(result["selected"])
            next_state, done, outcome_dict = env.apply_move(state, a_idx)
            state = next_state
            if done or (turn == 8):
                outcome = outcome_dict if isinstance(outcome_dict, dict) else None
                break

        # Tally result
        winner = (outcome or {}).get("winner") if isinstance(outcome, dict) else None
        if winner == "A":
            a_wins += 1
        elif winner == "B":
            b_wins += 1
        else:
            draws += 1

    total = a_wins + b_wins + draws
    score_b = (b_wins + 0.5 * draws) / max(1, total)
    elo = elo_delta_from_score(score_b)
    promote = (score_b >= threshold)

    result = {
        "games": games,
        "a": str(a),
        "b": str(b),
        "rules": rules_dict,
        "rollouts": rollouts,
        "temperature": temperature,
        "use_stub": use_stub,
        "results": {"a_wins": a_wins, "b_wins": b_wins, "draws": draws},
        "score_b": score_b,
        "elo_delta_b_vs_a": elo,
        "threshold": threshold,
        "promote": promote,
    }

    # Print only JSON to stdout for easy consumption
    print(json.dumps(result, ensure_ascii=False))


def _swap_sides(state: dict) -> dict:
    """
    Swap A and B labels in a canonical state dict to change the side to move.
    """
    import copy

    s = copy.deepcopy(state)
    # Swap hands
    hands = s.get("hands", {})
    hands_a, hands_b = list(hands.get("A", [])), list(hands.get("B", []))
    s["hands"] = {"A": hands_b, "B": hands_a}
    # Swap board owners
    board = s.get("board", [])
    for cell in board:
        owner = cell.get("owner")
        if owner == "A":
            cell["owner"] = "B"
        elif owner == "B":
            cell["owner"] = "A"
    # Swap to_move
    s["to_move"] = "A" if s.get("to_move") == "B" else "B"
    # Rules unchanged; state_hash will be recomputed by engine on apply
    return s


if __name__ == "__main__":
    main()