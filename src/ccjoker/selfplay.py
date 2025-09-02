from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import typer
from rich.console import Console
from tqdm import tqdm
import os
from multiprocessing import Process, Queue

from .checkpoint import load_checkpoint, set_seed
from .env import TripleTriadEnv
from .mcts import MCTS
from . import utils as U
from .gate import _swap_sides

app = typer.Typer(add_completion=False)
console = Console()


def _parse_rules(rules_str: str) -> Dict[str, bool]:
    """
    Parse rules string like "none" or "elemental,same,plus,same_wall".
    """
    rules_str = (rules_str or "").strip().lower()
    flags = {"elemental": False, "same": False, "plus": False, "same_wall": False}
    if rules_str in ("", "none", "no", "off"):
        return flags
    parts = [p.strip() for p in rules_str.split(",") if p.strip()]
    for p in parts:
        if p in flags:
            flags[p] = True
    return flags


def _policy_dist_to_map(pi: torch.Tensor, state: Dict[str, object]) -> Dict[str, float]:
    """
    Convert a [45] distribution into {"cardId-cell": prob} for the active hand.
    Only legal moves should have pi>0 due to masking; we still guard using hand length and empty cells.
    """
    to_move: str = str(state["to_move"])
    hand: List[int] = list(state["hands"][to_move])
    empty_cells = {i for i, cell in enumerate(state["board"]) if cell.get("card_id") is None}
    out: Dict[str, float] = {}
    for idx in range(U.MOVE_SPACE):
        p = float(pi[idx].item())
        if p <= 0.0:
            continue
        slot, cell = U.decode_move_index(idx)
        if slot >= len(hand):
            continue
        if cell not in empty_cells:
            continue
        key = f"{int(hand[slot])}-{int(cell)}"
        out[key] = out.get(key, 0.0) + p
    # Normalize for safety
    s = sum(out.values())
    if s > 0:
        for k in list(out.keys()):
            out[k] = out[k] / s
    return out


def _backfill_values_trajectories(traj: List[Dict[str, object]], winner: Optional[str]) -> None:
    """
    Mutate each record to set value_target ∈ {-1,0,1} by side-to-move perspective.
    """
    for rec in traj:
        to_move = rec.get("to_move", "A")
        v = 0
        if winner is None:
            v = 0
        else:
            v = 1 if winner == to_move else -1
        rec["value_target"] = v
        rec["value_mode"] = "winloss"


def _write_jsonl(path: Path, records: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _worker_selfplay(
    wid: int,
    num_games: int,
    model: str,
    out: str,
    rollouts: int,
    temperature: float,
    dirichlet_alpha: float,
    dirichlet_eps: float,
    sample_until: int,
    early_dirichlet_eps: float,
    late_temperature: float,
    seed: Optional[int],
    rules: str,
    triplecargo_cmd: Optional[str],
    cards: Optional[str],
    use_stub: bool,
    torch_threads: int,
    debug_ipc: bool,
    start_from: int,
    progress_queue: Optional[Queue],
) -> None:
    """
    Worker process: runs num_games on CPU, writes to shard {out}.w{wid}.jsonl,
    and reports per-game completion via progress_queue as ("done", 1, winner).

    start_from: global game index offset used to alternate the first player across workers
    so that games distributed across workers preserve a global alternation pattern.
    """
    try:
        # Limit intra-op and inter-op threads per worker to avoid oversubscription
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

        # Seed offset per worker
        worker_seed = None if seed is None else int(seed) + 100000 * int(wid)
        if worker_seed is not None:
            set_seed(worker_seed)

        # Load model on CPU in this worker
        net, cfg, _meta = load_checkpoint(model, device="cpu", train_mode=False)

        # Build env and MCTS
        rules_dict = _parse_rules(rules)
        env = TripleTriadEnv(
            rules=rules_dict,
            cards_path=cards if cards else "data/cards.json",
            triplecargo_cmd=triplecargo_cmd
            if triplecargo_cmd
            else "C:/Users/popid/Dev/triplecargo/target/release/precompute.exe",
            seed=worker_seed,
            use_stub=use_stub,
            max_card_id=(cfg.num_cards - 2) if use_stub else None,
            debug_ipc=debug_ipc,
        )
        mcts = MCTS(
            net=net,
            device="cpu",
            rollouts=rollouts,
            c_puct=1.0,
            temperature=temperature,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_eps=dirichlet_eps,
            verbose=False,
        )

        out_shard = Path(f"{out}.w{wid}.jsonl")
        game_id = 0

        for g in range(num_games):
            state = env.reset(seed=None if worker_seed is None else (worker_seed + g))

            # Alternate first player globally across workers using start_from offset
            first = "A" if ((start_from + g) % 2 == 0) else "B"
            if state.get("to_move") != first:
                state = _swap_sides(state)

            traj: List[Dict[str, object]] = []
            wrote = False

            for turn in range(9):
                root = state

                tau = float(temperature) if turn < int(sample_until) else float(late_temperature)
                eps_override = float(early_dirichlet_eps) if turn < int(sample_until) else float(dirichlet_eps)

                result = mcts.run(
                    env, root, temperature_override=tau, dirichlet_eps_override=eps_override
                )
                pi: torch.Tensor = result["pi"]

                # Record
                pt_map = _policy_dist_to_map(pi, root)
                record = {
                    "game_id": game_id,
                    "state_idx": turn,
                    "board": root["board"],
                    "hands": root["hands"],
                    "to_move": root["to_move"],
                    "turn": turn,
                    "rules": root["rules"],
                    "off_pv": False,
                    "policy_target": pt_map,
                    "value_target": 0,
                    "value_mode": "winloss",
                    "state_hash": root.get("state_hash"),
                }
                traj.append(record)

                # Step
                move_idx = int(result["selected"])
                next_state, _reward, done, info = env.step(move_idx)
                state = next_state

                if done or (turn == 8):
                    outcome = (info or {}).get("outcome") if isinstance(info, dict) else None
                    winner = None
                    if outcome and isinstance(outcome, dict):
                        winner = outcome.get("winner")
                    _backfill_values_trajectories(traj, winner)
                    traj = traj[:9]
                    _write_jsonl(out_shard, traj)
                    if progress_queue is not None:
                        try:
                            progress_queue.put(("done", 1, winner), block=False)
                        except Exception:
                            pass
                    wrote = True
                    break

            if not wrote:
                _backfill_values_trajectories(traj, None)
                traj = traj[:9]
                _write_jsonl(out_shard, traj)
                if progress_queue is not None:
                    try:
                        progress_queue.put(("done", 1, None), block=False)
                    except Exception:
                        pass

            game_id += 1

    except Exception as e:
        # Best-effort error signal
        if progress_queue is not None:
            try:
                progress_queue.put(("error", wid, str(e)), block=False)
            except Exception:
                pass


@app.command()
def main(
    model: Path = typer.Option(..., "--model", help="Checkpoint guiding self-play (.pt from ccj-train)"),
    games: int = typer.Option(1, "--games", min=1),
    out: Path = typer.Option(Path("data/raw/selfplay.jsonl"), "--out", help="Output JSONL to append"),
    rollouts: int = typer.Option(64, "--rollouts", min=0),
    temperature: float = typer.Option(1.0, "--temperature", help="Early-game temperature (turn < sample-until)"),
    dirichlet_alpha: float = typer.Option(0.3, "--dirichlet-alpha"),
    dirichlet_eps: float = typer.Option(0.25, "--dirichlet-eps", help="Late-game/root eps (turn ≥ sample-until)"),
    sample_until: int = typer.Option(6, "--sample-until", min=0, help="Sample from pi until this turn; then switch to late temperature"),
    early_dirichlet_eps: float = typer.Option(0.5, "--early-dirichlet-eps", help="Root Dirichlet eps while turn < sample-until"),
    late_temperature: float = typer.Option(0.0, "--late-temperature", help="Temperature from sample-until onward; 0.0 → argmax"),
    seed: Optional[int] = typer.Option(None, "--seed"),
    device: str = typer.Option("cpu", "--device", help="cpu|cuda"),
    rules: str = typer.Option("none", "--rules", help="Comma-separated: elemental,same,plus,same_wall or 'none'"),
    triplecargo_cmd: Optional[Path] = typer.Option(
        None, "--triplecargo-cmd", help="Path to Triplecargo precompute.exe (with --eval-state)"
    ),
    cards: Optional[Path] = typer.Option(
        None, "--cards", help="Path to cards.json used by Triplecargo (share with engine)"
    ),
    use_stub: bool = typer.Option(False, "--use-stub/--no-use-stub", help="Use Python stub env for CI"),
    verbose: bool = typer.Option(False, "--verbose", help="Enable detailed per-game/per-turn logging"),
    debug_ipc: bool = typer.Option(False, "--debug-ipc", help="Log raw JSON IPC to/from Triplecargo (stderr)"),
    workers: int = typer.Option(1, "--workers", min=1, help="CPU workers (processes). Only effective on --device cpu."),
    torch_threads: int = typer.Option(1, "--torch-threads", min=1, help="Torch threads per CPU worker."),
    keep_shards: bool = typer.Option(False, "--keep-shards/--no-keep-shards", help="Keep per-worker shard files instead of merging and deleting them."),
) -> None:
    """
    Generate self-play games with MCTS guided by a policy/value net. Append trajectories to JSONL.
    """
    # RNG/device
    if seed is not None:
        set_seed(seed)

    # Load model
    console.print(f"[selfplay] loading model {model} on {device}...")
    net, cfg, meta = load_checkpoint(model, device=device, train_mode=False)
    console.print(f"[selfplay] model={model} device={device} rollouts={rollouts} T={temperature}")
    console.print("[selfplay] model loaded")

    # Multiprocessing branch for CPU parallelism
    if device == "cpu" and int(workers) > 1:
        if verbose:
            console.log("[selfplay] verbose mode is silenced in multi-worker CPU mode", highlight=False)
        # Compute per-worker game allocation
        W = int(workers)
        counts = [games // W + (1 if i < (games % W) else 0) for i in range(W)]
        total_target = sum(counts)

        # Aggregate progress across workers
        q: Queue = Queue()
        procs: List[Process] = []
        a_wins = b_wins = draws = played = 0
        pbar = tqdm(total=total_target, desc="Self-play (CPU x workers)", dynamic_ncols=True)

        # Launch workers
        # Launch workers with a global start index so alternation across workers is preserved
        start_from = 0
        for wid, n in enumerate(counts):
            if n <= 0:
                continue
            p = Process(
                target=_worker_selfplay,
                args=(
                    wid,
                    int(n),
                    str(model),
                    str(out),
                    int(rollouts),
                    float(temperature),
                    float(dirichlet_alpha),
                    float(dirichlet_eps),
                    int(sample_until),
                    float(early_dirichlet_eps),
                    float(late_temperature),
                    seed,
                    str(rules),
                    str(triplecargo_cmd) if triplecargo_cmd else None,
                    str(cards) if cards else None,
                    bool(use_stub),
                    int(torch_threads),
                    bool(debug_ipc),
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
                    _, inc, winner = msg
                    finished += int(inc)
                    if winner == "A":
                        a_wins += 1
                    elif winner == "B":
                        b_wins += 1
                    else:
                        draws += 1
                    played += 1
                    pbar.update(1)
                    pbar.set_postfix({"played": played, "A": a_wins, "B": b_wins, "D": draws})
                elif msg[0] == "error":
                    _wid, _err_wid, err_text = msg[0], msg[1], msg[2] if len(msg) > 2 else ""
                    console.log(f"[selfplay][worker-error] wid={_err_wid} {err_text}")
            # else ignore unknown messages

        for p in procs:
            p.join()

        pbar.close()

        # Merge shards into out and optionally delete them
        out_path = Path(str(out))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "a", encoding="utf-8") as fout:
            for wid, _n in enumerate(counts):
                shard = Path(f"{str(out)}.w{wid}.jsonl")
                if shard.exists():
                    with open(shard, "r", encoding="utf-8") as fin:
                        for line in fin:
                            fout.write(line)
                    if not keep_shards:
                        try:
                            shard.unlink()
                        except Exception:
                            pass

        return

    # Build env
    rules_dict = _parse_rules(rules)
    console.print(f"[selfplay] building env (use_stub={use_stub})...")
    env = TripleTriadEnv(
        rules=rules_dict,
        cards_path=str(cards) if cards else "data/cards.json",
        triplecargo_cmd=str(triplecargo_cmd) if triplecargo_cmd else "C:/Users/popid/Dev/triplecargo/target/release/precompute.exe",
        seed=seed,
        use_stub=use_stub,
        max_card_id=(cfg.num_cards - 2) if use_stub else None,
        debug_ipc=debug_ipc,
    )
    if verbose and not use_stub:
        console.log("[ipc] Triplecargo eval-state enabled", highlight=False)
    console.print("[selfplay] env ready")

    if verbose:
        console.log(
            "[cfg] "
            f"games={games} device={device} rollouts={rollouts} "
            f"T_early={temperature} T_late={late_temperature} "
            f"dirichlet_alpha={dirichlet_alpha} eps_early={early_dirichlet_eps} eps_late={dirichlet_eps} "
            f"sample_until={sample_until} "
            f"rules={rules_dict} use_stub={use_stub} "
            f"cards={str(cards) if cards else 'data/cards.json'} "
            f"triplecargo_cmd={str(triplecargo_cmd) if triplecargo_cmd else 'default'} "
            f"num_cards={cfg.num_cards} max_card_id={(cfg.num_cards - 2) if use_stub else 'n/a'}"
        )

    # MCTS
    mcts = MCTS(
        net=net,
        device=device,
        rollouts=rollouts,
        c_puct=1.0,
        temperature=temperature,
        dirichlet_alpha=dirichlet_alpha,
        dirichlet_eps=dirichlet_eps,
        verbose=verbose,
    )
    console.log("MCTS setup")

    # Progress bar and counters (only when not verbose)
    a_wins = 0
    b_wins = 0
    draws = 0
    played = 0
    pbar = None
    if not verbose:
        pbar = tqdm(total=games, desc="Self-play", dynamic_ncols=True)

    # Run games
    game_id = 0
    total_t_mcts = 0.0
    total_t_step = 0.0
    total_t_write = 0.0
    for g in range(games):
        if verbose:
            console.log(f"[game {g}] start (seed={None if seed is None else (seed + g)})")
        state = env.reset(seed=None if seed is None else (seed + g))
        # Alternate first player each game to avoid start-side bias
        first = "A" if (g % 2 == 0) else "B"
        if state.get("to_move") != first:
            state = _swap_sides(state)

        traj: List[Dict[str, object]] = []
        wrote = False
        game_t_mcts = 0.0
        game_t_step = 0.0
        game_t_write = 0.0
        # Fixed 9 plies (Triple Triad): always emit at most 9 states per game
        for turn in range(9):
            root = state

            # Measure legal moves
            root_mask = env.legal_moves(root)
            legal = int(root_mask.sum().item())

            # MCTS timing
            # Per-turn exploration schedule
            tau = float(temperature) if turn < int(sample_until) else float(late_temperature)
            eps_override = float(early_dirichlet_eps) if turn < int(sample_until) else float(dirichlet_eps)

            t0m = time.perf_counter()
            result = mcts.run(env, root, temperature_override=tau, dirichlet_eps_override=eps_override)
            tm = time.perf_counter() - t0m
            game_t_mcts += tm
            total_t_mcts += tm
            pi: torch.Tensor = result["pi"]

            # Build output record (policy_target distribution map)
            pt_map = _policy_dist_to_map(pi, root)

            record = {
                "game_id": game_id,
                "state_idx": turn,
                "board": root["board"],
                "hands": root["hands"],
                "to_move": root["to_move"],
                "turn": turn,
                "rules": root["rules"],
                "off_pv": False,
                "policy_target": pt_map,  # mcts-style distribution
                "value_target": 0,  # temporary, backfilled later
                "value_mode": "winloss",
                "state_hash": root.get("state_hash"),
            }
            traj.append(record)

            # Select move from MCTS (sampling respects per-turn temperature; tau=0 → argmax)
            move_idx = int(result["selected"])

            # Decode (slot, cell) and card_id for logging
            slot_idx, cell_idx = U.decode_move_index(move_idx)
            hand_active: List[int] = list(root["hands"][root["to_move"]])
            card_id = hand_active[slot_idx] if 0 <= slot_idx < len(hand_active) else None

            # Step timing
            t0s = time.perf_counter()
            next_state, _reward, done, info = env.step(move_idx)
            ts = time.perf_counter() - t0s
            game_t_step += ts
            total_t_step += ts
            state = next_state

            if verbose:
                console.log(
                    f"[game {g} turn {turn}] legal={legal} "
                    f"mcts={tm*1000:.2f}ms step={ts*1000:.2f}ms "
                    f"selected_idx={move_idx} slot={slot_idx} cell={cell_idx} card_id={card_id} "
                    f"tau={tau} dir_eps={eps_override}"
                )

            if done or (turn == 8):
                outcome = (info or {}).get("outcome") if isinstance(info, dict) else None
                winner = None
                if outcome and isinstance(outcome, dict):
                    winner = outcome.get("winner")
                _backfill_values_trajectories(traj, winner)
                # Ensure exactly 9 records per game for deterministic tests
                t0w = time.perf_counter()
                traj = traj[:9]
                _write_jsonl(out, traj)
                tw = time.perf_counter() - t0w
                game_t_write += tw
                total_t_write += tw
                wrote = True

                # Progress update per completed game (only when not verbose)
                if not verbose:
                    if winner == "A":
                        a_wins += 1
                    elif winner == "B":
                        b_wins += 1
                    else:
                        draws += 1
                    played += 1
                    if pbar is not None:
                        pbar.update(1)
                        pbar.set_postfix({"played": played, "A": a_wins, "B": b_wins, "D": draws})

                if verbose:
                    console.log(
                        f"[game {g}] end winner={winner} write={tw*1000:.2f}ms "
                        f"totals: mcts={game_t_mcts*1000:.2f}ms step={game_t_step*1000:.2f}ms"
                    )
                break

        # Safety: if for some reason we didn't write (shouldn't happen), flush trajectory now
        if not wrote:
            _backfill_values_trajectories(traj, None)
            t0w = time.perf_counter()
            traj = traj[:9]
            _write_jsonl(out, traj)
            tw = time.perf_counter() - t0w
            game_t_write += tw
            total_t_write += tw

            # Progress update for forced write (treat as draw)
            if not verbose:
                draws += 1
                played += 1
                if pbar is not None:
                    pbar.update(1)
                    pbar.set_postfix({"played": played, "A": a_wins, "B": b_wins, "D": draws})

            if verbose:
                console.log(
                    f"[game {g}] forced write write={tw*1000:.2f}ms "
                    f"totals: mcts={game_t_mcts*1000:.2f}ms step={game_t_step*1000:.2f}ms"
                )

        game_id += 1

    if not verbose and pbar is not None:
        pbar.close()

    if verbose:
        console.print(
            f"[selfplay] all games done: "
            f"mcts_total={total_t_mcts*1000:.2f}ms step_total={total_t_step*1000:.2f}ms "
            f"write_total={total_t_write*1000:.2f}ms"
        )


if __name__ == "__main__":
    main()