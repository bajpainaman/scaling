"""
Vanilla Counterfactual Regret Minimization (CFR) implementation.

This is the basic CFR algorithm from Zinkevich et al. (2008).
Key properties:
- Converges to Nash equilibrium in two-player zero-sum games
- Average strategy converges at O(1/sqrt(T))
- Full tree traversal each iteration

Reference:
Zinkevich, M., et al. (2008). "Regret Minimization in Games with Incomplete Information"
"""

from typing import Optional, Callable
import numpy as np
from tqdm import tqdm

from .solver import CFRSolver, SolverResult, regret_matching
from ..games.base import Game, GameState


class VanillaCFR(CFRSolver):
    """
    Vanilla CFR implementation.

    Uses reach-weighted strategy averaging as in the original CFR paper.
    Traverses the complete game tree on each iteration.
    
    IMPORTANT: Policies are cached at the start of each iteration to ensure
    all game states in the same infoset use the same policy within an iteration.
    This is critical for correct CFR convergence.
    """

    def __init__(self, game: Game):
        super().__init__(game)
        # Cache for policies computed at the start of each iteration
        # This ensures all game states in the same infoset use the same policy
        self._iteration_policy_cache: dict[str, np.ndarray] = {}

    def solve(
        self,
        iterations: int,
        convergence_threshold: float = 1e-6,
        callback: Optional[Callable[[int, float], None]] = None,
        eval_every: int = 100,
        show_progress: bool = True,
    ) -> SolverResult:
        """
        Run Vanilla CFR for specified iterations.

        Args:
            iterations: Number of CFR iterations
            convergence_threshold: Stop if exploitability below this
            callback: Called with (iteration, exploitability) every eval_every iters
            eval_every: How often to evaluate exploitability
            show_progress: Whether to show progress bar

        Returns:
            SolverResult with converged strategy
        """
        iterator = range(iterations)
        if show_progress:
            iterator = tqdm(iterator, desc="CFR")

        for t in iterator:
            self.iterations_run = t + 1

            # Clear policy cache at start of iteration
            # Policies will be computed once per infoset on first visit
            self._iteration_policy_cache.clear()

            # Single traversal for both players
            initial_state = self.game.initial_state()
            self._cfr(initial_state, np.ones(self.game.num_players + 1, dtype=np.float64))

            # Periodic evaluation
            if (t + 1) % eval_every == 0:
                exploitability = self.get_exploitability()

                if show_progress:
                    iterator.set_postfix({"exploit": f"{exploitability:.6f}"})

                if callback:
                    callback(t + 1, exploitability)

                if exploitability < convergence_threshold:
                    print(f"\nConverged at iteration {t + 1} "
                          f"(exploitability: {exploitability:.6e})")
                    break

        return self.get_result()

    def _cfr(self, state: GameState, reach: np.ndarray) -> np.ndarray:
        """
        CFR traversal that computes utilities, updates regrets, and accumulates
        reach-weighted strategies.

        Args:
            state: Current game state
            reach: Reach probabilities [chance, p0, p1, ...]

        Returns:
            Expected utilities for all players [num_players]
        """
        if self.game.is_terminal(state):
            return np.array(self.game.get_payoffs(state), dtype=np.float64)

        if self.game.is_chance_node(state):
            utility = np.zeros(self.game.num_players, dtype=np.float64)
            for next_state, prob in self.game.chance_outcomes(state):
                new_reach = reach.copy()
                new_reach[0] *= prob  # Index 0 is chance
                utility += prob * self._cfr(next_state, new_reach)
            return utility

        # Decision node
        player = self.game.current_player(state)
        infoset = self.game.get_infoset(state, player)
        key = infoset.to_key()
        actions = self.game.get_actions(state)
        num_actions = len(actions)

        # Get current policy via regret matching
        # CRITICAL: Use cached policy to ensure all game states in the same
        # infoset use the same policy within a single iteration
        if key not in self._iteration_policy_cache:
            self._iteration_policy_cache[key] = regret_matching(
                self._get_regrets(key)[:num_actions]
            )
        policy = self._iteration_policy_cache[key]

        # Accumulate reach-weighted strategy for average
        player_reach = reach[player + 1]  # Player's reach probability
        strategy_sum = self._get_strategy_sum(key)
        for i in range(num_actions):
            strategy_sum[i] += player_reach * policy[i]

        # Compute utility for each action
        utility = np.zeros((num_actions, self.game.num_players), dtype=np.float64)
        for i, action in enumerate(actions):
            next_state = self.game.apply_action(state, action)
            new_reach = reach.copy()
            new_reach[player + 1] *= policy[i]  # Update player's reach
            utility[i] = self._cfr(next_state, new_reach)

        # Expected value under current policy
        value = np.einsum('ap,a->p', utility, policy)

        # Counterfactual reach probability (all players except current, including chance)
        # reach[0] = chance, reach[1] = p0, reach[2] = p1, ...
        cfr_reach = np.prod(reach[:player + 1]) * np.prod(reach[player + 2:])

        # Update regrets for current player
        regrets = self._get_regrets(key)
        for i in range(num_actions):
            regrets[i] += cfr_reach * (utility[i, player] - value[player])

        # Track reach probability for this player
        self.reach_probs[key] = player_reach

        return value

    def iterate_once(self) -> None:
        """
        Run a single CFR iteration (useful for external control).
        """
        self.iterations_run += 1

        # Clear policy cache at start of iteration
        self._iteration_policy_cache.clear()

        # CFR traversal - updates regrets and strategy sums
        initial_state = self.game.initial_state()
        self._cfr(initial_state, np.ones(self.game.num_players + 1, dtype=np.float64))

    def reset(self) -> None:
        """Reset solver state for a fresh solve."""
        super().reset()
        self._iteration_policy_cache.clear()
