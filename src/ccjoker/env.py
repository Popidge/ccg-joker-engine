from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import threading
import queue
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from . import utils as U


Element = Optional[str]
BoardCell = Dict[str, Any]
StateDict = Dict[str, Any]


def _canonical_state_hash(state: StateDict) -> str:
    """
    Stable hex hash for a canonicalized state dict (order-independent).
    Avoid heavy deps; use Python's built-in hash via JSON string and fallback to sha256 if available.
    """
    try:
        import hashlib

        s = json.dumps(state, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(s.encode("utf-8")).hexdigest()
    except Exception:
        # Non-cryptographic fallback (less stable across processes); acceptable for stub tests
        s = json.dumps(state, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hex(abs(hash(s)) & ((1 << 128) - 1))[2:].rjust(32, "0")


def _load_card_ids(cards_path: Optional[str]) -> List[int]:
    """
    Load the canonical card id set from a Triplecargo-style data/cards.json if present.

    Supported formats:
      - {"cards":[{"id":int, ...}, ...]}
      - [ {"id":int, ...}, ... ]  (top-level list of card objects)

    Fallback to range(0,100) if file missing or malformed.
    """
    if cards_path is None:
        return list(range(100))
    try:
        p = Path(cards_path)
        if not p.exists():
            return list(range(100))
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Normalize to a list of card entries whether the JSON is an object with
        # a "cards" key or a top-level list of card objects.
        if isinstance(data, dict):
            cards_list = data.get("cards")
        elif isinstance(data, list):
            cards_list = data
        else:
            cards_list = None

        if isinstance(cards_list, list):
            ids: List[int] = []
            for c in cards_list:
                # Support either dict entries with "id" or literal ints
                if isinstance(c, dict) and "id" in c:
                    try:
                        ids.append(int(c["id"]))
                    except Exception:
                        continue
                elif isinstance(c, int):
                    ids.append(int(c))
            if ids:
                return sorted(set(ids))
        return list(range(100))
    except Exception:
        return list(range(100))


class _TriplecargoClient:
    """
    Persistent subprocess client for Triplecargo `precompute --eval-state`.

    Protocol (per T13):
    - Send exactly one JSON object per line (UTF-8) on stdin.
    - Receive exactly one JSON object per line (UTF-8) on stdout.
      For stepping, include:
        input:  { ...state..., "apply": {"card_id": int, "cell": int } }
        output: { "state": {...}, "done": bool, "outcome": {"mode":"winloss","value":-1|0|1,"winner":"A"|"B"|null}, "state_hash": "..." }
      For evaluation only:
        input:  { ...state... }
        output: { "best_move": {...}, "value": -1|0|1, "margin": int, "pv":[...], "nodes": int, "depth": int, "state_hash": "..." }
    """

    def __init__(self, cmd_path: str, cards_path: Optional[str] = None, debug: bool = False) -> None:
        self.cmd_path = cmd_path
        self.cards_path = cards_path
        self.debug = bool(debug)
        self.proc: Optional[subprocess.Popen[str]] = None
        self.lock = threading.Lock()
        self._stderr_thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self.proc is not None:
            return
        args: List[str] = [self.cmd_path, "--eval-state"]
        if self.cards_path:
            args += ["--cards", self.cards_path]
        # Windows-friendly creation flags to avoid console window popups
        creationflags = 0x08000000 if os.name == "nt" else 0
        self.proc = subprocess.Popen(
            args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # line-buffered
            encoding="utf-8",
            universal_newlines=True,
            creationflags=creationflags,
        )
        if self.proc.stdin is None or self.proc.stdout is None:
            raise RuntimeError("Failed to start Triplecargo eval-state subprocess (no pipes)")

        # Start a background stderr drainer to prevent child from blocking on a full stderr pipe
        if self.proc.stderr is not None:
            def _drain_stderr(pipe, echo: bool):
                try:
                    for line in iter(pipe.readline, ""):
                        if echo:
                            sys.stderr.write(f"[triplecargo][stderr] {line}")
                            sys.stderr.flush()
                except Exception:
                    pass
            self._stderr_thread = threading.Thread(
                target=_drain_stderr, args=(self.proc.stderr, self.debug), daemon=True
            )
            self._stderr_thread.start()
        if self.debug:
            sys.stderr.write(f"[triplecargo] started: {' '.join(args)}\n")
            sys.stderr.flush()

    def close(self) -> None:
        try:
            if self.proc and self.proc.poll() is None:
                self.proc.terminate()
        except Exception:
            pass
        finally:
            self.proc = None

    def request(self, payload: Dict[str, Any], timeout: float = 30.0) -> Dict[str, Any]:
        """
        Thread-safe single request/response.
        Uses a reader thread with timeout to avoid deadlocks if the child does not respond.
        """
        if self.proc is None:
            self.start()
        assert self.proc is not None
        assert self.proc.stdin is not None
        assert self.proc.stdout is not None

        line = json.dumps(payload, ensure_ascii=False)
        with self.lock:
            try:
                if self.debug:
                    sys.stderr.write(f"[triplecargo] -> ({len(line)} bytes) {line}\n")
                    sys.stderr.flush()
                # Write one complete JSON line and flush
                self.proc.stdin.write(line + "\n")
                self.proc.stdin.flush()

                # Read one line with timeout using a helper thread
                q: "queue.Queue[object]" = queue.Queue(maxsize=1)

                def _reader(out_pipe, qobj):
                    try:
                        resp = out_pipe.readline()
                        qobj.put(resp)
                    except Exception as ex:
                        qobj.put(ex)

                t = threading.Thread(target=_reader, args=(self.proc.stdout, q), daemon=True)
                t.start()

                try:
                    obj = q.get(timeout=timeout)
                except queue.Empty:
                    # Timeout: attempt single-shot fallback (spawn per request)
                    if self.debug:
                        sys.stderr.write("[triplecargo] timeout waiting on persistent process; trying single-shot fallback\n")
                        sys.stderr.flush()
                    return self._request_single_shot(payload, timeout)

                if isinstance(obj, Exception):
                    raise obj  # re-raise reader exception

                resp_line = str(obj)
                if self.debug:
                    sys.stderr.write(f"[triplecargo] <- ({len(resp_line)} bytes) {resp_line}\n")
                    sys.stderr.flush()

                if not resp_line:
                    if self.debug:
                        sys.stderr.write("[triplecargo] empty stdout line; trying single-shot fallback\n")
                        sys.stderr.flush()
                    return self._request_single_shot(payload, timeout)

                resp = json.loads(resp_line)
            except Exception as e:
                # Include stderr if possible
                err = ""
                try:
                    if self.proc and self.proc.stderr:
                        err = self.proc.stderr.read()
                except Exception:
                    pass
                raise RuntimeError(f"Triplecargo eval-state request failed: {e}\nStderr:\n{err}") from e
        return resp

    def _request_single_shot(self, payload: Dict[str, Any], timeout: float = 30.0) -> Dict[str, Any]:
        """
        Fallback: spawn a fresh precompute --eval-state process for this single request,
        write one JSON line, then read exactly one JSON line with a timeout.
        """
        args: List[str] = [self.cmd_path, "--eval-state"]
        if self.cards_path:
            args += ["--cards", self.cards_path]
        if self.debug:
            sys.stderr.write(f"[triplecargo][oneshot] starting: {' '.join(args)}\n")
            sys.stderr.flush()

        proc = subprocess.Popen(
            args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            universal_newlines=True,
        )
        if proc.stdin is None or proc.stdout is None:
            raise RuntimeError("Triplecargo oneshot: no pipes")

        line = json.dumps(payload, ensure_ascii=False) + "\n"
        try:
            # Use communicate for simpler timeout handling; it closes stdin automatically
            out, err = proc.communicate(input=line, timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            out2, err2 = proc.communicate()
            raise RuntimeError(
                f"Triplecargo oneshot timeout after {timeout:.1f}s. "
                f"stderr:\n{(err or '') + (err2 or '')}"
            )
        except Exception as e:
            try:
                proc.kill()
            except Exception:
                pass
            raise RuntimeError(f"Triplecargo oneshot failed: {e}")

        if self.debug:
            sys.stderr.write(f"[triplecargo][oneshot] stdout bytes={len(out)} stderr bytes={len(err or '')}\n")
            if out:
                preview = out.splitlines()[0] if out.splitlines() else ""
                sys.stderr.write(f"[triplecargo][oneshot] <- {preview}\n")
            if err:
                sys.stderr.write(f"[triplecargo][oneshot][stderr] {err}\n")
            sys.stderr.flush()

        # Take first non-empty line from stdout as the JSON object
        line_out = ""
        for ln in (out or "").splitlines():
            ln = ln.strip()
            if ln:
                line_out = ln
                break

        if not line_out:
            raise RuntimeError(f"Triplecargo oneshot: empty stdout. stderr:\n{err or ''}")

        try:
            return json.loads(line_out)
        except Exception as e:
            raise RuntimeError(f"Triplecargo oneshot: invalid JSON: {e}; line={line_out}")


@dataclass
class EnvConfig:
    rules: Dict[str, bool]
    cards_path: Optional[str] = None
    triplecargo_cmd: str = "C:/Users/popid/Dev/triplecargo/target/release/precompute.exe"
    seed: Optional[int] = None
    use_stub: bool = False


class TripleTriadEnv:
    """
    Triple Triad environment wrapper driving state transitions via Triplecargo (or a deterministic stub).

    Key methods:
      - reset(seed?) -> state_dict
      - step(move_idx) -> (next_state, reward, done, info)
      - legal_moves(state) -> FloatTensor [45] {0,1}
      - encode(state) -> dict of tensors compatible with PolicyValueNet.forward
      - apply_move(state, move_idx) -> (next_state, done, outcome/winner)  -- used by MCTS tree simulation
    """

    def __init__(
        self,
        rules: Dict[str, bool],
        cards_path: str | None = "data/cards.json",
        triplecargo_cmd: str = "C:/Users/popid/Dev/triplecargo/target/release/precompute.exe",
        seed: Optional[int] = None,
        use_stub: bool = False,
        max_card_id: Optional[int] = None,
        debug_ipc: bool = False,
    ) -> None:
        self.cfg = EnvConfig(
            rules=dict(
                elemental=bool(rules.get("elemental", False)),
                same=bool(rules.get("same", False)),
                plus=bool(rules.get("plus", False)),
                same_wall=bool(rules.get("same_wall", False)),
            ),
            cards_path=cards_path,
            triplecargo_cmd=triplecargo_cmd,
            seed=seed,
            use_stub=use_stub,
        )
        self.rng = random.Random(self.cfg.seed)
        self._client: Optional[_TriplecargoClient] = None if self.cfg.use_stub else _TriplecargoClient(
            self.cfg.triplecargo_cmd, self.cfg.cards_path, debug=bool(debug_ipc)
        )
        self._current_state: Optional[StateDict] = None
        # Load card id domain; in stub mode we may cap by max_card_id to match model capacity
        self._card_ids: List[int] = _load_card_ids(self.cfg.cards_path)
        # In engine mode, exclude id 0 (Triplecargo CardsDb typically does not contain id 0)
        if not self.cfg.use_stub:
            self._card_ids = [cid for cid in self._card_ids if cid is not None and int(cid) != 0]
        # In stub mode, optionally cap by max_card_id
        if self.cfg.use_stub and (max_card_id is not None):
            # Restrict to [0..max_card_id] to avoid embedding OOB when models are tiny in tests
            self._card_ids = [cid for cid in self._card_ids if 0 <= int(cid) <= int(max_card_id)]
            # Ensure contiguous domain if the source file was sparse
            if not self._card_ids:
                self._card_ids = list(range(int(max_card_id) + 1))
        # Ensure we have a reasonable pool
        if len(self._card_ids) < 10:
            if self.cfg.use_stub and (max_card_id is not None):
                self._card_ids = list(range(int(max_card_id) + 1))
            else:
                # Use 1..100 in engine mode to avoid 0; 0..99 in stub mode is acceptable
                self._card_ids = list(range(1, 101)) if not self.cfg.use_stub else list(range(100))

    # ---------- Public API ----------

    def reset(self, seed: Optional[int] = None) -> StateDict:
        """
        Start a new game with deterministic hand/element sampling.
        Returns the initial canonical state dict.
        """
        if seed is not None:
            self.rng.seed(seed)
        elif self.cfg.seed is not None:
            self.rng.seed(self.cfg.seed)

        # Sample unique hands of length 5 for A and B (no overlap)
        if len(self._card_ids) < 10:
            # Fallback safety
            self._card_ids = list(range(100))
        ids = self._card_ids[:]
        self.rng.shuffle(ids)
        handA = sorted(ids[:5])
        handB = sorted(ids[5:10])

        # Elements per cell if elemental enabled
        elemental = self.cfg.rules["elemental"]
        elements: List[Element] = [None] * U.NUM_CELLS
        if elemental:
            # deterministic assignment from allowed set including None
            allowed = [e for e in U.ELEMENTS if e is not None]
            for i in range(U.NUM_CELLS):
                elements[i] = allowed[self.rng.randrange(len(allowed))]

        board: List[BoardCell] = []
        for c in range(U.NUM_CELLS):
            cell: BoardCell = {"cell": c, "card_id": None, "owner": None}
            if elemental:
                cell["element"] = elements[c]
            board.append(cell)

        state: StateDict = {
            "board": board,
            "hands": {"A": handA, "B": handB},
            "to_move": "A",
            "turn": 0,
            "rules": dict(self.cfg.rules),
        }
        # Hash from engine if available (eval-only), else canonical hash
        state["state_hash"] = _canonical_state_hash(self._strip_hash(state))
        self._current_state = state
        return state

    def step(self, move_idx: int) -> Tuple[StateDict, int, bool, Dict[str, Any]]:
        """
        Apply a move from the current state and update internal pointer.
        Returns (next_state, reward, done, info). Reward is 0 until terminal.
        """
        assert self._current_state is not None, "Call reset() before step()."
        next_state, done, outcome = self.apply_move(self._current_state, move_idx)
        reward = 0
        if done:
            # reward from the perspective of the player who just moved (previous state's to_move)
            # outcome winner vs previous to_move
            prev_player = self._current_state["to_move"]
            winner = outcome.get("winner")
            if winner is None:
                reward = 0
            else:
                reward = 1 if winner == prev_player else -1
        self._current_state = next_state
        info = {
            "outcome": outcome if done else None,
        }
        return next_state, reward, done, info

    def legal_moves(self, state: StateDict) -> torch.Tensor:
        """
        Compute a [45]-dim mask with 1 for legal moves given the current state:
          - hand slots index 0..len(hand)-1 are valid slots; the rest are pads
          - target cells must be empty
        """
        to_move: str = state["to_move"]
        hand: List[int] = list(state["hands"][to_move])
        # Identify empty cells
        empty_cells = [i for i, cell in enumerate(state["board"]) if cell.get("card_id") is None]
        mask = torch.zeros(U.MOVE_SPACE, dtype=torch.float32)
        # Valid slots are contiguous [0..len(hand)-1]
        for slot in range(min(len(hand), U.HAND_SLOTS)):
            for cell in empty_cells:
                idx = U.move_index(slot, cell)
                mask[idx] = 1.0
        return mask

    def encode(self, state: StateDict) -> Dict[str, torch.Tensor]:
        """
        Encode the given state into tensors compatible with PolicyValueNet.forward.
        Mirrors dataset encoding conventions (board, hand, rules, move mask).
        """
        rules = state.get("rules", {})
        elemental = bool(rules.get("elemental", False))
        board = state.get("board", [])
        owners: List[torch.Tensor] = []
        elements: List[torch.Tensor] = []
        card_ids: List[int] = []
        for cell in board:
            owners.append(U.owner_onehot(cell.get("owner")))
            # if elemental=false, dataset drops element info to None
            elements.append(U.element_onehot(cell.get("element") if elemental else None))
            card_ids.append(U.card_id_to_embed_index(cell.get("card_id")))
        board_owner = torch.stack(owners, dim=0)  # [9,3]
        board_element = torch.stack(elements, dim=0)  # [9,9]
        board_card_ids = torch.tensor(card_ids, dtype=torch.long)  # [9]

        to_move = state.get("to_move", "A")
        hand_ids_raw: List[int] = list(state.get("hands", {}).get(to_move, []))
        padded_embed_ids, hand_mask = U.pad_hand_ids(hand_ids_raw, pad_to=U.HAND_SLOTS)
        hand_card_ids = torch.tensor(padded_embed_ids, dtype=torch.long)  # [5]
        hand_mask_t = torch.tensor(hand_mask, dtype=torch.float32)  # [5]

        rules_t = U.rules_to_tensor(rules)
        move_mask = U.build_move_mask(hand_mask)

        return {
            "board_card_ids": board_card_ids,
            "board_owner": board_owner,
            "board_element": board_element,
            "hand_card_ids": hand_card_ids,
            "hand_mask": hand_mask_t,
            "rules": rules_t,
            "move_mask": move_mask,
        }

    # ---------- For MCTS (stateless previews) ----------

    def apply_move(self, state: StateDict, move_idx: int) -> Tuple[StateDict, bool, Dict[str, Any]]:
        """
        Pure transition: apply move to the provided state (does not mutate self._current_state).
        Returns (next_state, done, outcome).
        """
        slot, cell = U.decode_move_index(move_idx)
        to_move: str = state["to_move"]
        hand: List[int] = list(state["hands"][to_move])
        if not (0 <= slot < len(hand)):
            # illegal slot; return same state
            return state, False, {}
        # cell must be empty
        if not (0 <= cell < U.NUM_CELLS) or state["board"][cell].get("card_id") is not None:
            return state, False, {}

        card_id = hand[slot]

        if self.cfg.use_stub:
            return self._apply_move_stub(state, card_id, cell)
        else:
            return self._apply_move_cli(state, card_id, cell)

    # ---------- Internal helpers ----------

    def _apply_move_cli(self, state: StateDict, card_id: int, cell: int) -> Tuple[StateDict, bool, Dict[str, Any]]:
        """
        Try stepping via Triplecargo. If the engine only supports eval (no 'state' in response),
        fall back to the deterministic Python stub to advance the state.
        """
        assert self._client is not None
        payload = dict(self._strip_hash(state))
        payload["apply"] = {"card_id": int(card_id), "cell": int(cell)}
        resp = self._client.request(payload)

        next_state: StateDict | None = resp.get("state")
        # If eval-only response (no next state), or missing essential fields, fall back to stub stepping
        if not isinstance(next_state, dict) or not next_state or ("board" not in next_state) or ("hands" not in next_state):
            return self._apply_move_stub(state, card_id, cell)

        # Ensure required fields
        if "rules" not in next_state:
            next_state["rules"] = dict(self.cfg.rules)
        if "state_hash" not in next_state:
            next_state["state_hash"] = resp.get(
                "state_hash",
                _canonical_state_hash(self._strip_hash(next_state)),
            )
        done = bool(resp.get("done", False))
        outcome = resp.get("outcome", {"mode": "winloss", "value": 0, "winner": None})
        return next_state, done, outcome

    def _apply_move_stub(self, state: StateDict, card_id: int, cell: int) -> Tuple[StateDict, bool, Dict[str, Any]]:
        """
        Deterministic Python fallback:
        - Place the card for to_move at the target cell.
        - No Elemental/Same/Plus/Combo; just ownership of placed cell.
        - Hands shrink; to_move flips; turn increments.
        - Terminal when 9 placements have occurred.
        - Outcome by counting owned cells at terminal.
        """
        to_move: str = state["to_move"]
        other: str = "B" if to_move == "A" else "A"
        turn: int = int(state.get("turn", 0))

        # Build next board
        next_board: List[BoardCell] = []
        for i, old in enumerate(state["board"]):
            if i == cell:
                # place this card
                new_cell: BoardCell = {
                    "cell": i,
                    "card_id": int(card_id),
                    "owner": to_move,
                }
                if "element" in old:
                    new_cell["element"] = old.get("element")
                next_board.append(new_cell)
            else:
                # carry forward
                new_cell = {
                    "cell": i,
                    "card_id": old.get("card_id"),
                    "owner": old.get("owner"),
                }
                if "element" in old:
                    new_cell["element"] = old.get("element")
                next_board.append(new_cell)

        # Shrink hand (remove the played card at the slot index)
        # Reconstruct hands preserving order
        hands = {
            "A": list(state["hands"]["A"]),
            "B": list(state["hands"]["B"]),
        }
        # Remove by first occurrence
        if card_id in hands[to_move]:
            hands[to_move].remove(card_id)

        next_state: StateDict = {
            "board": next_board,
            "hands": hands,
            "to_move": other,
            "turn": turn + 1,
            "rules": dict(state.get("rules", self.cfg.rules)),
        }

        # Terminal after 9 placements
        done = (turn + 1) >= 9
        outcome = {"mode": "winloss", "value": 0, "winner": None}
        if done:
            a_cnt = sum(1 for c in next_board if c.get("owner") == "A")
            b_cnt = sum(1 for c in next_board if c.get("owner") == "B")
            if a_cnt > b_cnt:
                outcome["value"] = 1  # from A-perspective win -> sided evaluation depends on consumer
                outcome["winner"] = "A"
            elif b_cnt > a_cnt:
                outcome["value"] = -1
                outcome["winner"] = "B"
            else:
                outcome["value"] = 0
                outcome["winner"] = None

        next_state["state_hash"] = _canonical_state_hash(self._strip_hash(next_state))
        return next_state, done, outcome

    @staticmethod
    def _strip_hash(state: StateDict) -> StateDict:
        """Return a shallow copy of state without state_hash (engine owns it)."""
        s = dict(state)
        if "state_hash" in s:
            s.pop("state_hash")
        return s
