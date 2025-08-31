from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import math
import torch

from .model import PolicyValueNet, mask_policy_logits
from . import utils as U


@dataclass
class Node:
    """
    A search tree node for AlphaZero-lite MCTS.

    Attributes:
      - state_hash: stable identifier for deduplication
      - prior: torch.FloatTensor[45] prior over moves at this node (masked to legal moves, renormalized)
      - move_mask: torch.FloatTensor[45] {0,1}
      - N: visit counts per action (list[int] length 45)
      - W: total value per action (list[float] length 45)
      - Q: mean value per action (list[float] length 45)
      - children: map action_idx -> child Node
      - is_expanded: whether priors/value were evaluated
      - value: leaf value v in [-1,1] from side-to-move perspective at this node (set on first expand)
    """
    state_hash: str
    prior: Optional[torch.Tensor] = None
    move_mask: Optional[torch.Tensor] = None
    N: List[int] = field(default_factory=lambda: [0 for _ in range(U.MOVE_SPACE)])
    W: List[float] = field(default_factory=lambda: [0.0 for _ in range(U.MOVE_SPACE)])
    Q: List[float] = field(default_factory=lambda: [0.0 for _ in range(U.MOVE_SPACE)])
    children: Dict[int, "Node"] = field(default_factory=dict)
    is_expanded: bool = False
    value: float = 0.0  # v in [-1,1]


def logits_to_priors(logits: torch.Tensor, move_mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Convert masked logits [45] to prior distribution [45] with zero mass on illegal moves.
    """
    assert logits.dim() == 1 and logits.numel() == U.MOVE_SPACE
    assert move_mask.shape == (U.MOVE_SPACE,)
    masked = mask_policy_logits(logits.unsqueeze(0), move_mask.unsqueeze(0))[0]  # [45]
    # Replace -inf with large negative safely handled by softmax
    p = torch.softmax(masked, dim=-1)
    # Zero out illegal explicitly to ensure exact zeros, then renormalize
    p = p * move_mask
    s = p.sum().clamp(min=eps)
    p = p / s
    return p


def value_logits_to_scalar(value_logits: torch.Tensor) -> float:
    """
    Convert value head logits [3] (order [loss, draw, win]) to scalar v in [-1,1].
    v = p_win - p_loss
    """
    assert value_logits.dim() == 1 and value_logits.numel() == 3
    probs = torch.softmax(value_logits, dim=-1)
    p_loss = float(probs[0].item())
    p_draw = float(probs[1].item())  # unused in scalar
    p_win = float(probs[2].item())
    return p_win - p_loss


def temperature_transform(counts: torch.Tensor, tau: float, eps: float = 1e-12) -> torch.Tensor:
    """
    Apply temperature to visit counts.
      - tau > 0: pi_i ∝ N_i^(1/tau)
      - tau == 0: argmax as one-hot
    """
    if tau <= 0.0:
        # One-hot on argmax over counts
        idx = int(counts.argmax().item()) if counts.numel() > 0 else 0
        out = torch.zeros_like(counts, dtype=torch.float32)
        out[idx] = 1.0
        return out
    ex = torch.pow(counts.clamp_min(0.0) + eps, 1.0 / tau)
    s = ex.sum().clamp_min(eps)
    return (ex / s).to(torch.float32)


class MCTS:
    """
    AlphaZero-lite MCTS guided by PolicyValueNet.

    Usage:
      mcts = MCTS(net, device="cpu", rollouts=64)
      result = mcts.run(env, root_state)
      pi = result["pi"]  # FloatTensor [45]
      visit_counts = result["visit_counts"]  # List[int] length 45
      selected = result["selected"]  # int action index sampled (after temperature)
    """

    def __init__(
        self,
        net: PolicyValueNet,
        device: str = "cpu",
        rollouts: int = 64,
        c_puct: float = 1.0,
        temperature: float = 1.0,
        dirichlet_alpha: float = 0.3,
        dirichlet_eps: float = 0.25,
        verbose: bool = False,
    ) -> None:
        self.net = net
        self.device = torch.device(device)
        self.rollouts = int(rollouts)
        self.c_puct = float(c_puct)
        self.temperature = float(temperature)
        self.dirichlet_alpha = float(dirichlet_alpha)
        self.dirichlet_eps = float(dirichlet_eps)
        self.verbose = bool(verbose)

        self.net.eval().to(self.device)

        # Node cache by state_hash for light transposition sharing at the root subtree
        self._cache: Dict[str, Node] = {}

    @torch.no_grad()
    def _evaluate_leaf(self, env, state) -> Tuple[torch.Tensor, float, torch.Tensor]:
        """
        Encode state, run the network, and return (priors [45], value scalar, move_mask [45]).
        """
        x = env.encode(state)
        xb = {k: v.unsqueeze(0).to(self.device) for k, v in x.items()}  # batch of 1
        policy_logits, value_logits = self.net(xb)
        policy_logits = policy_logits[0]  # [45]
        # move_mask from x (already computed from hand)
        move_mask = xb["move_mask"][0].to(torch.float32).cpu()
        priors = logits_to_priors(policy_logits.cpu(), move_mask)
        v = value_logits_to_scalar(value_logits[0].cpu())
        return priors, v, move_mask

    def _select_action(self, node: Node, total_N: int) -> int:
        """
        Select action using PUCT: Q + c_puct * P * sqrt(N_total) / (1 + N(a))
        Returns action index.
        """
        assert node.prior is not None and node.move_mask is not None
        sqrtN = math.sqrt(max(1, total_N))
        best_score = -1e18
        best_a = 0
        for a in range(U.MOVE_SPACE):
            if node.move_mask[a].item() <= 0:
                continue
            P = float(node.prior[a].item())
            N = node.N[a]
            Q = node.Q[a]
            u = self.c_puct * P * (sqrtN / (1 + N))
            score = Q + u
            if score > best_score:
                best_score = score
                best_a = a
        return best_a

    def _expand(self, node: Node, env, state) -> float:
        """
        Expand leaf node: evaluate priors/value and mark node expanded.
        Returns leaf value v from current player's perspective.
        """
        priors, v, move_mask = self._evaluate_leaf(env, state)
        node.prior = priors
        node.move_mask = move_mask
        node.is_expanded = True
        node.value = float(v)
        return node.value

    def _backup(self, path: List[Tuple[Node, int]], leaf_value: float) -> None:
        """
        Backup leaf value along the path. Alternate perspective sign at each ply.
        path is list of (node, action_taken_at_node).
        """
        v = leaf_value
        for depth, (node, a) in enumerate(reversed(path)):
            # On the way back, alternate sign
            node.N[a] += 1
            node.W[a] += v
            node.Q[a] = node.W[a] / max(1, node.N[a])
            v = -v

    def run(
        self,
        env,
        root_state,
        temperature_override: Optional[float] = None,
        dirichlet_eps_override: Optional[float] = None,
    ) -> Dict[str, object]:
        """
        Perform rollouts from the root_state using env transitions (env.apply_move).

        Optional overrides:
          - temperature_override: if provided, used for visit->pi transform at root
          - dirichlet_eps_override: if provided, used for root prior Dirichlet mixing

        Returns:
          - pi: FloatTensor [45] normalized visit distribution (temperature applied)
          - visit_counts: List[int] length 45
          - selected: int sampled action from pi
        """
        # Resolve per-call knobs
        tau = float(temperature_override) if (temperature_override is not None) else self.temperature
        dir_eps = float(dirichlet_eps_override) if (dirichlet_eps_override is not None) else self.dirichlet_eps

        root_hash = root_state.get("state_hash")
        if not root_hash:
            # Fallback: compute a canonical hash if not present
            from .env import _canonical_state_hash, TripleTriadEnv  # local import to avoid cycles
            state_for_hash = dict(root_state)
            state_for_hash.pop("state_hash", None)
            root_hash = _canonical_state_hash(state_for_hash)

        # Initialize or fetch root node
        root = self._cache.get(root_hash)
        if root is None:
            root = Node(state_hash=root_hash)
            self._cache[root_hash] = root

        # Detect terminal at root via legal moves (no empty cells or no hand)
        root_move_mask = env.legal_moves(root_state)
        if (root_move_mask.sum().item() <= 0) or (int(root_state.get("turn", 0)) >= 9):
            # Terminal: no moves. Use zero distribution.
            visit_counts = torch.zeros(U.MOVE_SPACE, dtype=torch.float32)
            pi = temperature_transform(visit_counts, tau)
            selected = int(pi.argmax().item()) if tau <= 0.0 else int(torch.multinomial(pi.clamp_min(0), 1).item())
            return {"pi": pi, "visit_counts": visit_counts.tolist(), "selected": selected}

        # Ensure root has move mask
        root.move_mask = root_move_mask

        # Expand root if needed
        if not root.is_expanded or root.prior is None:
            v0 = self._expand(root, env, root_state)
            # Note: even if expanded, we still run additional simulations to refine visit counts

        # Root Dirichlet noise mixing (apply only at root before rollouts)
        orig_prior = root.prior.clone() if root.prior is not None else None
        if (dir_eps > 0.0) and (root.prior is not None) and (root.move_mask is not None):
            mask = root.move_mask
            legal_idx = (mask > 0).nonzero(as_tuple=False).flatten()
            k = int(legal_idx.numel())
            if k > 0:
                alpha_vec = torch.full((k,), float(self.dirichlet_alpha), dtype=torch.float32)
                noise = torch.distributions.Dirichlet(alpha_vec).sample()
                noisy_full = torch.zeros_like(root.prior)
                noisy_full[legal_idx] = noise
                mixed = (1.0 - float(dir_eps)) * root.prior + float(dir_eps) * noisy_full
                # Ensure mask and renormalize
                mixed = mixed * mask
                s_mix = mixed.sum().clamp_min(1e-12)
                mixed = mixed / s_mix
                root.prior = mixed
                if self.verbose:
                    print(f"[selfplay] root priors mixed with Dirichlet(alpha={self.dirichlet_alpha}, eps={dir_eps})")

        # Rollouts
        for _ in range(self.rollouts):
            node = root
            state = root_state
            path: List[Tuple[Node, int]] = []

            # Selection down to a leaf
            while node.is_expanded and node.move_mask is not None:
                total_N = sum(node.N)
                action = self._select_action(node, total_N)
                path.append((node, action))
                # Transition to child state (stateless apply)
                next_state, done, _ = env.apply_move(state, action)
                # Get/create child node
                child_hash = next_state.get("state_hash")
                if not child_hash:
                    from .env import _canonical_state_hash
                    s = dict(next_state)
                    s.pop("state_hash", None)
                    child_hash = _canonical_state_hash(s)
                child = node.children.get(action)
                if child is None:
                    child = Node(state_hash=child_hash)
                    node.children[action] = child
                state = next_state
                node = child
                # If terminal reached, evaluate terminal value (draw as 0.0 fallback)
                if done or int(state.get("turn", 0)) >= 9:
                    leaf_v = 0.0
                    self._backup(path, leaf_v)
                    break
            else:
                # We are at a leaf (not expanded yet) — expand
                leaf_v = self._expand(node, env, state)
                self._backup(path, leaf_v)

        # Restore original root priors if we mixed noise
        if 'orig_prior' in locals() and orig_prior is not None:
            root.prior = orig_prior

        # Build root visit distribution (masked)
        visit_counts = torch.tensor(root.N, dtype=torch.float32)
        mask = root.move_mask if root.move_mask is not None else torch.ones(U.MOVE_SPACE, dtype=torch.float32)
        visit_counts = visit_counts * mask
        pi = temperature_transform(visit_counts, tau)
        # Reapply mask and renormalize to ensure zero mass on illegal actions
        pi = (pi * mask)
        s = float(pi.sum().item())
        if s > 0:
            pi = pi / pi.sum()
        else:
            # Fallback: uniform over legal moves
            pi = torch.zeros_like(pi)
            legal_idx = (mask > 0).nonzero(as_tuple=False).flatten()
            if legal_idx.numel() > 0:
                pi[legal_idx] = 1.0 / float(legal_idx.numel())

        # Sample action from pi
        selected = int(torch.multinomial(pi.clamp_min(0), 1).item())

        return {
            "pi": pi,  # [45] float, masked and normalized
            "visit_counts": visit_counts.tolist(),
            "selected": selected,
        }