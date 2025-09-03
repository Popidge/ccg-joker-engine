from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, List

import torch
import typer
from rich.console import Console
from tqdm import tqdm
import os
from multiprocessing import Process, Queue

from .checkpoint import load_checkpoint, set_seed
from .env import TripleTriadEnv
from .mcts import MCTS
from .model import PolicyValueNet, mask_policy_logits

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


def _worker_gate(
    wid: int,
    num_games: int,
    a_path: str,
    b_path: str,
    rollouts: int,
    temperature: float,
    seed: Optional[int],
    rules: str,
    triplecargo_cmd: Optional[str],
    cards: Optional[str],
    use_stub: bool,
    torch_threads: int,
    debug_ipc: bool,
    mirror: bool,
    start_from: int,
    progress_queue: Optional[Queue],
) -> None:
    """
    Worker: plays num_games of A vs B on CPU and reports per-game results via progress_queue.
    mirror: when True, games are paired so each dealt hand is played twice (mirrored),
      with A starting one game and B starting the other.
    start_from is the global game index offset to preserve alternating first player.
    """
    try:
        # Limit per-worker torch threading to avoid oversubscription
        try:
            torch.set_num_threads(int(torch_threads))
        except Exception:
            pass
        try:
            torch.set_num_interop_threads(1)
        except Exception:
            pass
        os.environ.setdefault("OMP_NUM_THREADS", str(int(torch_threads)))
        os.environ.setdefault("MKL_NUM_THREADS", str(int(torch_threads)))

        # Seed per worker for reproducibility
        worker_seed = None if seed is None else int(seed) + 100000 * int(wid)
        if worker_seed is not None:
            set_seed(worker_seed)

        # Load models on CPU
        net_a, cfg_a, _meta_a = load_checkpoint(a_path, device="cpu", train_mode=False)
        net_b, cfg_b, _meta_b = load_checkpoint(b_path, device="cpu", train_mode=False)

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
            cards_path=cards if cards else "data/cards.json",
            triplecargo_cmd=triplecargo_cmd
            if triplecargo_cmd
            else "C:/Users/popid/Dev/triplecargo/target/release/precompute.exe",
            seed=worker_seed,
            use_stub=use_stub,
            max_card_id=max_card_id,
            debug_ipc=debug_ipc,
        )

        # MCTS (if enabled)
        mcts_a = MCTS(net=net_a, device="cpu", rollouts=rollouts, c_puct=1.0, temperature=temperature) if rollouts > 0 else None
        mcts_b = MCTS(net=net_b, device="cpu", rollouts=rollouts, c_puct=1.0, temperature=temperature) if rollouts > 0 else None

        # Play assigned games
        for i in range(num_games):
            # When mirror is enabled, pair games reuse the same deal.
            if mirror:
                pair_offset = (start_from + i) // 2
                reset_seed = None if worker_seed is None else (worker_seed + int(pair_offset))
            else:
                reset_seed = None if worker_seed is None else (worker_seed + int(i))
            # Global alternation of first player
            first = "A" if ((start_from + i) % 2 == 0) else "B"
            # Reset with deterministic per-game seed (possibly shared for mirrored pair)
            state = env.reset(seed=reset_seed)
            # Ensure desired starting side
            if state.get("to_move") != first:
                state = _swap_sides(state)

            outcome = None
            for turn in range(9):
                # If no legal moves, draw
                root_mask = env.legal_moves(state)
                if float(root_mask.sum().item()) <= 0.0:
                    outcome = {"winner": None}
                    break

                to_move = state.get("to_move", "A")
                if rollouts <= 0:
                    net = net_a if to_move == "A" else net_b
                    a_idx = select_move_greedy(net, torch.device("cpu"), env, state)
                else:
                    mcts = mcts_a if to_move == "A" else mcts_b
                    result = mcts.run(env, state)
                    a_idx = int(result["selected"])
                next_state, done, outcome_dict = env.apply_move(state, a_idx)
                state = next_state
                if done or (turn == 8):
                    outcome = outcome_dict if isinstance(outcome_dict, dict) else None
                    break

            winner = (outcome or {}).get("winner") if isinstance(outcome, dict) else None
            if progress_queue is not None:
                try:
                    progress_queue.put(("done", winner), block=False)
                except Exception:
                    pass

    except Exception as e:
        if progress_queue is not None:
            try:
                progress_queue.put(("error", wid, str(e)), block=False)
            except Exception:
                pass


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
    mirror: bool = typer.Option(True, "--mirror/--no-mirror", help="When on, mirror each dealt hand across a pair of games (A starts one, B starts the other)"),
    threshold: float = typer.Option(0.55, "--threshold", min=0.5, max=1.0, help="Promotion threshold as score vs A"),
    workers: int = typer.Option(1, "--workers", min=1, help="CPU workers (processes). Only effective on --device cpu."),
    torch_threads: int = typer.Option(1, "--torch-threads", min=1, help="Torch threads per CPU worker."),
    debug_ipc: bool = typer.Option(False, "--debug-ipc", help="Log raw JSON IPC to/from Triplecargo (stderr)"),
) -> None:
    """
    Head-to-head gating: play matches A vs B; report W/D/L and Elo delta for B relative to A.
    """
    if seed is not None:
        set_seed(seed)

    # CPU parallelism branch (only for CPU and workers > 1)
    if device == "cpu" and int(workers) > 1:
        # Compute per-worker game allocation with near-even split
        W = int(workers)
        counts: List[int] = [games // W + (1 if i < (games % W) else 0) for i in range(W)]
        total_target = sum(counts)

        q: Queue = Queue()
        procs: List[Process] = []
        a_wins = 0
        b_wins = 0
        draws = 0
        pbar = tqdm(total=total_target, desc="Gate (CPU x workers)", dynamic_ncols=True)

        # Launch workers with non-overlapping start indices for alternating first player
        start_from = 0
        for wid, n in enumerate(counts):
            if n <= 0:
                continue
            p = Process(
                target=_worker_gate,
                args=(
                    wid,
                    int(n),
                    str(a),
                    str(b),
                    int(rollouts),
                    float(temperature),
                    seed,
                    str(rules),
                    str(triplecargo_cmd) if triplecargo_cmd else None,
                    str(cards) if cards else None,
                    bool(use_stub),
                    int(torch_threads),
                    bool(debug_ipc),
                    bool(mirror),
                    int(start_from),
                    q,
                ),
            )
            p.daemon = True
            p.start()
            procs.append(p)
            start_from += int(n)

        finished = 0
        while finished < total_target:
            msg = q.get()
            if isinstance(msg, tuple) and msg:
                if msg[0] == "done":
                    _, winner = msg
                    finished += 1
                    if winner == "A":
                        a_wins += 1
                    elif winner == "B":
                        b_wins += 1
                    else:
                        draws += 1
                    pbar.update(1)
                    pbar.set_postfix({"A": a_wins, "B": b_wins, "D": draws})
                elif msg[0] == "error":
                    _wid = msg[1] if len(msg) > 1 else "?"
                    err_text = msg[2] if len(msg) > 2 else ""
                    console.log(f"[gate][worker-error] wid={_wid} {err_text}")

        for p in procs:
            p.join()

        pbar.close()

        total = a_wins + b_wins + draws
        score_b = (b_wins + 0.5 * draws) / max(1, total)
        elo = elo_delta_from_score(score_b)
        promote = (score_b >= threshold)

        result = {
            "games": games,
            "a": str(a),
            "b": str(b),
            "rules": _parse_rules(rules),
            "rollouts": rollouts,
            "temperature": temperature,
            "use_stub": use_stub,
            "results": {"a_wins": a_wins, "b_wins": b_wins, "draws": draws},
            "score_b": score_b,
            "elo_delta_b_vs_a": elo,
            "threshold": threshold,
            "promote": promote,
        }
        print(json.dumps(result, ensure_ascii=False))
        return

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
        debug_ipc=debug_ipc,
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
        # When mirror is enabled, reuse the same deal for paired games
        if mirror:
            base_offset = g // 2
            reset_seed = None if seed is None else (seed + int(base_offset))
        else:
            reset_seed = None if seed is None else (seed + g)
        # Alternate first player: even -> A starts; odd -> B starts
        first = "A" if (g % 2 == 0) else "B"
        # Reset with deterministic per-game seed (possibly shared for mirrored pair)
        state = env.reset(seed=reset_seed)
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