♠️ Triplecargo — Triple Triad Assistant & Solver (FF8)

Deterministic, high‑performance Rust implementation of the Final Fantasy VIII Triple Triad ruleset, designed to support:

- Perfect play solving (Negamax + αβ + TT).
- Full state‑space precomputation for instant queries.
- Analysis experiments (first‑player advantage, card ratings, rule impacts).
- AI/ML integration for lightweight move suggestion and imperfect‑information play.
Triplecargo is both a research‑grade solver and the foundation for a superhuman Triple Triad assistant — combining exact computation with modern AI techniques.

---
✨ Features

- Full ruleset: Basic, Elemental, Same, Plus, Same Wall, Combo cascades.
- Deterministic engine: fixed move ordering, reproducible cascades, stable hashing.
- High‑performance solver: Negamax + αβ pruning + transposition table.
- Precompute driver: bulk solve all reachable states for a given hand/ruleset.
- Persistence: append‑only batch writes, deterministic compaction, optional compression, checkpoint/resume.
- Analysis mode: RAM‑only runs for experiments (no I/O bottleneck).
- Throughput: ~10M nodes/sec, ~1M states/sec on a Ryzen 3600.
- Training data export (JSONL): trajectory or full state‑space, policy_format onehot|mcts, deterministic with fixed seed.

---
📂 Project layout

- Core engine: rules, state, hashing, scoring.
- Solver: negamax + αβ + TT, PV reconstruction.
- Precompute: state enumeration, parallel solve, persistence.
- Persistence: batch writes, compression, checkpointing.
- Binaries:
	- tt-cli: demo CLI.
	- precompute: bulk solver + DB writer.
- Tests: engine correctness, rule edge cases, solver determinism.

---
🧮 Data model

- Cards: loaded from data/cards.json, validated for ranges and uniqueness.
- Board: 3×3 fixed array, optional per‑cell element.
- Hands: 5 cards per player (engine: fixed arrays; export: shrinking lists as cards are played).
- Rules: independent toggles for Elemental, Same, Plus, Same Wall.
- Hashing: XOR‑based Zobrist, incremental updates, 128‑bit keys.

---
📤 Training data export (JSONL)

Overview
- Export solver-labelled states to JSONL for training and analysis.
- Two export modes:
  - --export-mode trajectory (default): emits a single 9-state principal-variation trajectory per sampled game.
  - --export-mode full: emits the entire reachable state space for one sampled hand pair (exhaustive).
- Deterministic by design: same seed+flags → identical output (including elements layout, off-PV behaviour, and policy/value targets).

Core flags
- Output and mode
  - --export PATH                JSONL output file
  - --export-mode trajectory|full
  - --games N                    number of games (trajectory mode only; full mode ignores this)
- Parallelism and scheduling
  - --threads N                  number of worker threads (default: available_parallelism()-1, min 1)
  - --chunk-size N               number of games each worker fetches at once; if omitted, defaults to min(32, max(1, floor(games/threads))); affects scheduling only
- Sampling and determinism
  - --seed U64                   master seed; controls hand sampling, per-ply RNG (off-PV), and MCTS rollouts
  - --hand-strategy random|stratified
  - --rules elemental,same,plus,same_wall (or 'none')
  - --elements none|random       element layout (random is deterministic per game)
- Policy and value targets
  - --policy-format onehot|mcts
  - --mcts-rollouts N            rollouts used when --policy-format mcts (default 100)
  - --value-mode winloss|margin
- Off-PV stepping (trajectory mode only)
  - --off-pv-rate FLOAT [0..1]   per-game probability of off-PV stepping (0 disables)
  - --off-pv-strategy random|weighted|mcts
    - random: uniform over legal non-PV moves; if no alternative, fall back to PV.
    - weighted: softmax over root child negamax Q values; PV mass set to 0, renormalised, then sampled.
    - mcts: reuse root MCTS distribution when policy_format=mcts; otherwise compute root MCTS with --mcts-rollouts and sample after zeroing PV.

- Transposition table (per worker)
  - --tt-bytes N               TT size per worker in MiB (default 32). Capacity is rounded down to the nearest power-of-two entries that fit under the requested budget. Applies to both trajectory and full export modes. Total memory usage ≈ workers × tt-bytes.
Parallel export (worker + writer) — TASK T7
- The export pipeline now supports an explicit worker + writer model (replacing the previous Rayon pool).
- New CLI flag: --threads N (default: available_parallelism()-1). Workers claim game indices via an atomic counter and operate deterministically.
- Worker behavior:
  - Deterministically sample hands/elements from the master seed and game id.
  - For each game produce 9 JSONL lines by running full-depth solver searches (using search_root_with_children() for policy Q values).
  - Send (game_id, Vec<String>) to the single writer thread.
- Writer behavior:
  - Single dedicated writer receives per-game line blocks, buffers out-of-order arrivals, and writes games in strict increasing game_id order to maintain byte-for-byte determinism across different thread counts.
- Determinism guarantees:
  - With identical seed + flags the JSONL output is byte-for-byte identical whether --threads=1 or --threads>1.
  - Tests: --games 10 produces 90 lines; output compared equal across thread counts.
- Progress:
  - The trajectory export includes an indicatif ProgressBar with a background updater that reports states/sec, nodes/sec, ETA, and policy info.

Chunked scheduling — TASK T9
- New CLI flag: --chunk-size N (dynamic default).
- Default behaviour when --chunk-size is omitted:
  - effective_chunk_size = min(32, max(1, floor(games/threads))).
  - This keeps small runs snappy (smaller chunks) while avoiding excessive overhead on large runs (cap at 32).
- Scheduling model:
  - Each worker requests N games at a time (a "chunk") from a deterministic round‑robin dispatcher.
  - Mapping is round‑based: in round r, worker w processes chunk index (r * workers + w).
  - For a chunk index k, the assigned game_id range is [k*N, min((k+1)*N, games)).
  - Workers solve games in their chunk sequentially, then proceed to their next round’s chunk.
- Determinism:
  - Assignment is a pure function of (worker_id, worker_count, chunk_size), not timing.
  - The single writer preserves strict increasing game_id order, so JSONL output is byte‑for‑byte identical for the same seed + flags regardless of --chunk-size and --threads.
  - game_id stability is preserved across different chunk sizes and thread counts.
- Logging:
  - At startup, export logs include the configured chunk size:
    [export] chunk_size=N
- Trade‑offs:
  - Larger chunks reduce scheduling overhead and improve locality for per‑worker TT warming.
  - Smaller chunks can improve tail latency and load balance for heterogeneous workloads.
  - Chunking affects scheduling only; output content and ordering are unchanged.

Per-worker transposition table — TASK T8
- Each worker owns a fixed-size, direct-mapped transposition table (no sharing; no locks).
- Default budget is 32 MiB per worker; configurable via --tt-bytes N (MiB).
- Warming: in trajectory mode, a worker keeps its TT alive across all games it processes to maximise reuse.
- Full mode uses a single TT of the same budget for the entire DFS export.
- Rounding rule: the requested budget is rounded down to the largest power-of-two capacity that fits. Effective bytes are approximately capacity × entry_size.
- Memory formula: total TT memory ≈ workers × tt-bytes.
- Determinism: with identical seed + flags, output is byte-for-byte identical regardless of --tt-bytes (TT size does not influence labels).
- Startup logs (one line per worker) show the chosen capacity, e.g.:
  [worker 0] TT target=32 MiB capacity=8388608 entries ≈31.9 MiB
Determinism specifics
- Trajectory mode:
  - Labels (policy/value) are computed from full-depth perfect play at each ply; labels remain PV-optimal regardless of stepping strategy.
  - Off-PV stepping uses a per-ply RNG derived from (seed, game_id, turn) for deterministic sampling.
- Full mode:
  - Enumeration is deterministic. For optional MCTS policy distributions in full mode, rollout RNG derives from (seed XOR state_zobrist) to preserve traversal-order invariance.
- With identical seeds and flags, the exported JSONL is byte-for-byte identical.

CLI examples
- Trajectory export (onehot), 10 games, no elements:
  - target/release/precompute --export export.jsonl --export-mode trajectory --games 10 --seed 42 --hand-strategy random --rules none --elements none --policy-format onehot --value-mode winloss
- Trajectory export (MCTS policy, 256 rollouts), stratified hands, Elemental+Same, margin values:
  - target/release/precompute --export export_mcts.jsonl --export-mode trajectory --games 10 --seed 42 --hand-strategy stratified --rules elemental,same --elements random --policy-format mcts --mcts-rollouts 256 --value-mode margin
- Trajectory export with off-PV weighted stepping @ 20% games:
  - target/release/precompute --export export_weighted.jsonl --export-mode trajectory --games 100 --seed 123 --hand-strategy random --rules none --elements none --policy-format onehot --off-pv-rate 0.2 --off-pv-strategy weighted
- Trajectory export with off-PV mcts stepping, reusing policy distribution:
  - target/release/precompute --export export_mcts_offpv.jsonl --export-mode trajectory --games 100 --seed 123 --hand-strategy stratified --rules elemental,same --elements random --policy-format mcts --mcts-rollouts 128 --off-pv-rate 0.2 --off-pv-strategy mcts
- Full state‑space export (exhaustive) for one sampled hand pair:
  - target/release/precompute --export export_full.jsonl --export-mode full --seed 42 --hand-strategy random --rules none --elements none --policy-format onehot --value-mode winloss

JSONL schema
Each line contains one state emitted before a move is taken.

{
  "game_id": 12,                // sequential per sampled game (trajectory); 0 in full mode
  "state_idx": 0,               // 0..8 within a trajectory (equals "turn")
  "board": [
    {"cell":0,"card_id":12,"owner":"A","element":"F"}, // element present only when rules.elemental=true; otherwise omitted
    {"cell":1,"card_id":null,"owner":null,"element":null},
    {"cell":2,"card_id":null,"owner":null,"element":"W"}
  ],
  "hands": {
    "A": [34,56,78,90,11],      // shrinking lists (5 → 0 across a trajectory)
    "B": [22,33,44,55,66]
  },
  "to_move": "A",
  "turn": 0,
  "rules": {
    "elemental": true,
    "same": true,
    "plus": false,
    "same_wall": false
  },

  // Policy target semantics:
  // onehot → single best move object:
  //   "policy_target": {"card_id":34,"cell":0}
  // mcts → distribution over legal moves:
  //   "policy_target": {"34-0":0.7,"56-1":0.3}
  "policy_target": {"34-0": 0.7, "56-1": 0.3},

  // Value target semantics controlled by --value-mode:
  // - winloss → {-1,0,+1} from side-to-move perspective (sign of solver value)
  // - margin  → integer margin A_cards − B_cards (A-perspective)
  "value_target": 1,
  "value_mode": "winloss",

  // Off-principal-variation sampling flag; true for all lines of off-PV games, false otherwise
  "off_pv": false,

  // 128-bit Zobrist hash of the state (hex string)
  "state_hash": "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4"
}

Export semantics

State timing
- Each JSONL line represents the state before a move is made.
- At turn = t, the board has t occupied cells; the side in to_move will play their (t+1)-th card next.
- Hands shrink accordingly; at the final turn = 8 there is still one card left (and a policy_target showing where it will be placed).

Policy targets
- --policy-format onehot:
  - Exported as a single best-move object:
    "policy_target": {"card_id": 34, "cell": 0}
- --policy-format mcts:
  - Exported as a {"cardId-cell": prob} map over legal moves:
    "policy_target": {"34-0": 0.7, "56-1": 0.3}
- Terminal states:
  - onehot → policy_target omitted (null).
  - mcts → empty map {}.

Off-PV stepping (trajectory mode)
- Labels remain PV-optimal: policy/value targets reflect perfect play; stepping only affects which move is played next to advance the trajectory.
- Per-game activation: with probability --off-pv-rate a game uses off-PV stepping for all its plies.
- Strategies:
  - random:
    - Uniform over legal non-PV moves. If no alternative exists, fall back to the PV move (or None at terminal).
  - weighted:
    - Use root child negamax values (Q). Compute softmax(Q), set PV move probability to 0, renormalise, sample with a deterministic RNG derived from (seed, game_id, turn).
    - If no alternative exists, fall back to PV.
  - mcts:
    - If --policy-format mcts, reuse the root MCTS distribution computed for policy_target; otherwise compute a fresh root MCTS with --mcts-rollouts (deterministic per-ply seed).
    - Set PV probability to 0, renormalise, sample deterministically.
    - Fallback: if the PV move had all probability mass (sum of non-PV probs == 0), select the highest-probability non-PV move deterministically.

Value targets
- --value-mode winloss:
  - value ∈ {-1, 0, +1}, always from side-to-move perspective.
    - +1 = side-to-move wins under perfect play
    - 0  = draw
    - −1 = side-to-move loses
- --value-mode margin:
  - integer final score difference (A_cards − B_cards), always from A’s perspective, independent of who is to move.
- Values are computed by solving the full remaining depth at each ply, so they match final outcomes.

Elements
- When rules.elemental = true, each board cell includes an "element" field with one of: F, I, T, W, E, P, H, L.
- Otherwise the "element" field is omitted.

Metadata
- game_id: sequential per sampled game (trajectory); 0 in full mode.
- state_idx: 0..8 within a trajectory (equals turn).
- off_pv: true for all lines of off-PV games; false otherwise.
- state_hash: 128‑bit Zobrist hash of the state, hex string.

🕹️ Rules implemented

- Basic capture: placed card flips weaker adjacent opponents.
- Elemental: +1/−1 adjustments based on cell element.
- Same: 2+ equalities trigger flips.
- Same Wall: walls count as 10 for Same.
- Plus: equal sums trigger flips.
- Combo: cascades apply Basic rule only.

---
⚡ Performance & determinism

- Move ordering: Corners > Edges > Center, then cell index, then card id.
- Cascade order: BFS queue, ascending indices.
- TT: fixed‑size array, depth‑preferred replacement, full key verification.
- Progress: MultiProgress bars, states/sec, nodes/sec.
- Determinism: same state → same result, reproducible DBs.

---
🔮 Roadmap

Near‑term

- Query API: CLI/HTTP service for instant best‑move lookups from solved DBs.
- Analysis experiments:
	- First‑player advantage quantification.
	- Card/hand Elo ratings.
	- Elemental variance studies.
	- Rule interaction balance.

Mid‑term

- AI/ML integration:
	- NPZ export option for compact storage.
	- PyTorch Dataset utilities to load JSONL → tensors.
	- Train small policy/value nets for lightweight move suggestion.
	- Reinforced self‑play (AlphaZero‑lite) to refine policies.
- Imperfect‑info play:
	- Expectimax/MCTS with hidden hands.
	- Opponent modelling.

Long‑term

- Superhuman assistant:
	- Instant move suggestions in Open and Closed formats.
	- Configurable playstyles: Optimal, Robust, Exploitative.
	- Human‑like play modes for fun.
- Meta‑analysis:
	- “Best hand” search for each ruleset.
	- Card tier lists and balance insights.
	- Quantitative impact of Elemental RNG.

---
🧪 Testing scope

- Engine tests: rule correctness, determinism, allocation‑free hot paths.
- Solver tests: terminal values, PV determinism.
- Rule edge tests: Same, Plus, Same Wall, Combo cascades.
- Persistence tests: batch flush, compression, checkpoint/resume determinism.

---
📜 License


The code in this repository is provided under an open license (TBD).
Triple Triad belongs to Square Enix (Final Fantasy VIII). This is a clean‑room reimplementation for research and educational purposes.

---
🙏 Acknowledgements

- FF8’s Triple Triad ruleset by Square Enix.
- Chess/Go engine design patterns (negamax, αβ, Zobrist, TT).
- OpenAI GPT‑5 + Kilo Code for collaborative design and implementation.