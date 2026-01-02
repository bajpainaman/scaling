"""
CFR Solver base classes and data structures.

This module defines the core abstractions for Counterfactual Regret Minimization solvers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Optional, Callable
import numpy as np

from ..games.base import Game, GameState, InfoSet


@dataclass
class CFROutput:
    """
    Output from CFR for a single information set.

    Contains all the information needed for training neural networks
    to predict counterfactual values.
    """
    infoset_key: str                 # Unique identifier for this infoset
    player: int                      # Which player's infoset
    cfv: np.ndarray                  # Counterfactual values per action [num_actions]
    strategy: np.ndarray             # Average strategy (action probs) [num_actions]
    regrets: np.ndarray              # Cumulative regrets [num_actions]
    reach_prob: float                # Probability of reaching this infoset

    def __post_init__(self):
        # Ensure arrays are float64 for numerical precision
        self.cfv = np.array(self.cfv, dtype=np.float64)
        self.strategy = np.array(self.strategy, dtype=np.float64)
        self.regrets = np.array(self.regrets, dtype=np.float64)


@dataclass
class SolverResult:
    """
    Complete result from running CFR solver.

    Contains the converged strategy and all infoset data needed for
    generating training datasets.
    """
    game_name: str
    iterations: int
    exploitability: float
    infoset_data: Dict[str, CFROutput] = field(default_factory=dict)

    def get_strategy(self, infoset_key: str) -> np.ndarray:
        """Get the strategy for an information set."""
        if infoset_key in self.infoset_data:
            return self.infoset_data[infoset_key].strategy
        raise KeyError(f"Unknown infoset: {infoset_key}")

    def num_infosets(self) -> int:
        """Number of information sets visited."""
        return len(self.infoset_data)


def regret_matching(regrets: np.ndarray) -> np.ndarray:
    """
    Convert cumulative regrets to strategy via regret matching.

    CRITICAL: Must handle all-negative regrets by returning uniform distribution.
    This is a common source of bugs in CFR implementations.

    Args:
        regrets: Cumulative regrets for each action [num_actions]

    Returns:
        Strategy (probability distribution over actions) [num_actions]
    """
    positive = np.maximum(regrets, 0.0)
    total = positive.sum()

    if total > 0:
        return positive / total
    else:
        # Uniform fallback when all regrets are non-positive
        return np.ones_like(regrets) / len(regrets)


def regret_matching_batch(regrets: np.ndarray) -> np.ndarray:
    """
    Batched regret matching for numpy arrays.

    Args:
        regrets: Regrets of shape [batch, num_actions]

    Returns:
        Strategies of shape [batch, num_actions]
    """
    positive = np.maximum(regrets, 0.0)
    totals = positive.sum(axis=-1, keepdims=True)

    # Handle zero totals (uniform fallback)
    uniform = np.ones_like(positive) / positive.shape[-1]
    return np.where(totals > 0, positive / totals, uniform)


class CFRSolver(ABC):
    """
    Abstract base class for CFR solvers.

    Subclasses implement specific variants like Vanilla CFR, DCFR, MCCFR.
    """

    def __init__(self, game: Game):
        """
        Initialize solver for a specific game.

        Args:
            game: Game instance to solve
        """
        self.game = game
        self.num_actions = game.num_actions

        # Regret and strategy accumulators
        # Use float64 for numerical precision
        self.regret_sum: Dict[str, np.ndarray] = {}
        self.strategy_sum: Dict[str, np.ndarray] = {}

        # Track reach probabilities
        self.reach_probs: Dict[str, float] = {}

        # Iteration counter
        self.iterations_run = 0

    def _get_regrets(self, key: str) -> np.ndarray:
        """Get or initialize regret accumulator for an infoset."""
        if key not in self.regret_sum:
            self.regret_sum[key] = np.zeros(self.num_actions, dtype=np.float64)
        return self.regret_sum[key]

    def _get_strategy_sum(self, key: str) -> np.ndarray:
        """Get or initialize strategy accumulator for an infoset."""
        if key not in self.strategy_sum:
            self.strategy_sum[key] = np.zeros(self.num_actions, dtype=np.float64)
        return self.strategy_sum[key]

    def get_current_strategy(self, infoset_key: str) -> np.ndarray:
        """
        Get current strategy from regrets via regret matching.

        Args:
            infoset_key: Key identifying the information set

        Returns:
            Current strategy (probability distribution)
        """
        regrets = self._get_regrets(infoset_key)
        return regret_matching(regrets)

    def get_average_strategy(self, infoset_key: str) -> np.ndarray:
        """
        Get time-averaged strategy (converges to Nash).

        Args:
            infoset_key: Key identifying the information set

        Returns:
            Average strategy (probability distribution)
        """
        strategy_sum = self._get_strategy_sum(infoset_key)
        total = strategy_sum.sum()

        if total > 0:
            return strategy_sum / total
        else:
            return np.ones(self.num_actions) / self.num_actions

    @abstractmethod
    def solve(
        self,
        iterations: int,
        convergence_threshold: float = 1e-6,
        callback: Optional[Callable[[int, float], None]] = None,
    ) -> SolverResult:
        """
        Run CFR to find Nash equilibrium.

        Args:
            iterations: Number of CFR iterations
            convergence_threshold: Stop early if exploitability below this
            callback: Optional callback(iteration, exploitability) called periodically

        Returns:
            SolverResult with converged strategy and infoset data
        """
        ...

    def get_exploitability(self) -> float:
        """
        Compute exploitability of current strategy.

        Exploitability = sum of best-response values for each player.
        At Nash equilibrium, this equals 0.

        Returns:
            Exploitability value (lower is better, 0 at Nash)
        """
        total = 0.0
        for player in range(self.game.num_players):
            br_value = self._compute_best_response_value(player)
            total += br_value
        return total

    def _compute_best_response_value(self, player: int) -> float:
        """
        Compute best-response value for a player against current strategy.

        Uses a two-pass approach:
        1. First pass: compute counterfactual values for each action at each infoset
        2. Second pass: compute expected value using BR strategy (argmax at each infoset)

        Args:
            player: Player computing best response

        Returns:
            Expected value of best response
        """
        # First pass: accumulate counterfactual values per infoset per action
        infoset_action_values: dict[str, np.ndarray] = {}
        infoset_reach_sum: dict[str, float] = {}
        
        self._br_accumulate(
            state=self.game.initial_state(),
            player=player,
            reach_prob=1.0,
            infoset_action_values=infoset_action_values,
            infoset_reach_sum=infoset_reach_sum,
        )
        
        # Compute BR strategy: argmax at each infoset
        br_strategy: dict[str, np.ndarray] = {}
        for key, action_values in infoset_action_values.items():
            # Normalize by reach probability
            if infoset_reach_sum.get(key, 0) > 0:
                normalized = action_values / infoset_reach_sum[key]
            else:
                normalized = action_values
            # BR is deterministic: put all probability on best action
            br_strat = np.zeros(len(action_values))
            br_strat[np.argmax(normalized)] = 1.0
            br_strategy[key] = br_strat
        
        # Second pass: compute expected value under BR strategy
        return self._br_eval(
            state=self.game.initial_state(),
            player=player,
            br_strategy=br_strategy,
        )

    def _br_accumulate(
        self,
        state: GameState,
        player: int,
        reach_prob: float,
        infoset_action_values: dict[str, np.ndarray],
        infoset_reach_sum: dict[str, float],
    ) -> float:
        """
        First pass: accumulate expected values for each action at each infoset.
        Returns the expected value at this state under opponent's average strategy.
        """
        if self.game.is_terminal(state):
            return self.game.get_payoffs(state)[player]

        if self.game.is_chance_node(state):
            value = 0.0
            for next_state, prob in self.game.chance_outcomes(state):
                value += prob * self._br_accumulate(
                    next_state, player, reach_prob * prob,
                    infoset_action_values, infoset_reach_sum
                )
            return value

        current = self.game.current_player(state)
        infoset = self.game.get_infoset(state, current)
        key = infoset.to_key()
        actions = self.game.get_actions(state)
        num_actions = len(actions)

        if current == player:
            # For the BR player: compute value of each action
            action_values = np.zeros(num_actions, dtype=np.float64)
            for i, action in enumerate(actions):
                next_state = self.game.apply_action(state, action)
                action_values[i] = self._br_accumulate(
                    next_state, player, reach_prob,
                    infoset_action_values, infoset_reach_sum
                )
            
            # Accumulate reach-weighted action values for this infoset
            if key not in infoset_action_values:
                infoset_action_values[key] = np.zeros(num_actions, dtype=np.float64)
                infoset_reach_sum[key] = 0.0
            
            infoset_action_values[key] += reach_prob * action_values
            infoset_reach_sum[key] += reach_prob
            
            # Return max (for the expected value computation)
            return np.max(action_values)
        else:
            # Opponent plays their average strategy
            strategy = self.get_average_strategy(key)
            value = 0.0
            for i, action in enumerate(actions):
                if i < len(strategy):
                    next_state = self.game.apply_action(state, action)
                    child_value = self._br_accumulate(
                        next_state, player, reach_prob * strategy[i],
                        infoset_action_values, infoset_reach_sum
                    )
                    value += strategy[i] * child_value
            return value
    
    def _br_eval(
        self,
        state: GameState,
        player: int,
        br_strategy: dict[str, np.ndarray],
    ) -> float:
        """
        Second pass: compute expected value under computed BR strategy.
        """
        if self.game.is_terminal(state):
            return self.game.get_payoffs(state)[player]
        
        if self.game.is_chance_node(state):
            value = 0.0
            for next_state, prob in self.game.chance_outcomes(state):
                value += prob * self._br_eval(next_state, player, br_strategy)
            return value
        
        current = self.game.current_player(state)
        infoset = self.game.get_infoset(state, current)
        key = infoset.to_key()
        actions = self.game.get_actions(state)
        
        if current == player:
            # Use computed BR strategy
            strategy = br_strategy.get(key, np.ones(len(actions)) / len(actions))
        else:
            # Opponent uses average strategy
            strategy = self.get_average_strategy(key)
        
        value = 0.0
        for i, action in enumerate(actions):
            if i < len(strategy):
                next_state = self.game.apply_action(state, action)
                value += strategy[i] * self._br_eval(next_state, player, br_strategy)
            return value

    def get_result(self) -> SolverResult:
        """
        Package current solver state into a SolverResult.

        Returns:
            SolverResult with all infoset data
        """
        infoset_data = {}

        for key in self.regret_sum.keys():
            regrets = self._get_regrets(key)
            strategy = self.get_average_strategy(key)
            cfv = regrets.copy()  # CFV approximated by regrets

            # Parse player from key
            player = int(key.split('|')[0][1])

            reach_prob = self.reach_probs.get(key, 0.0)

            infoset_data[key] = CFROutput(
                infoset_key=key,
                player=player,
                cfv=cfv,
                strategy=strategy,
                regrets=regrets,
                reach_prob=reach_prob,
            )

        exploitability = self.get_exploitability()

        return SolverResult(
            game_name=self.game.name,
            iterations=self.iterations_run,
            exploitability=exploitability,
            infoset_data=infoset_data,
        )

    def reset(self) -> None:
        """Reset solver state for a fresh solve."""
        self.regret_sum.clear()
        self.strategy_sum.clear()
        self.reach_probs.clear()
        self.iterations_run = 0
