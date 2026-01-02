"""
Discounted Counterfactual Regret Minimization (DCFR) implementation.

DCFR applies discounting to older iterations, which leads to faster
convergence in practice. The key insight is that early regrets/strategies
are computed with poor estimates, so they should be down-weighted.

Discount parameters:
- alpha (α): Positive regret discount power (default: 1.5)
- beta (β): Negative regret discount power (default: 0.5)
- gamma (γ): Strategy averaging power (default: 2.0)

At iteration t:
- Positive regrets weighted by: t^α / (t^α + 1)
- Negative regrets weighted by: t^β / (t^β + 1)
- Strategy contribution weighted by: (t / (t+1))^γ

Reference:
Brown, N., & Sandholm, T. (2019). "Solving Imperfect-Information Games
via Discounted Regret Minimization"
"""

from typing import Optional, Callable
import numpy as np
from tqdm import tqdm

from .solver import CFRSolver, SolverResult, regret_matching
from ..games.base import Game, GameState


class DiscountedCFR(CFRSolver):
    """
    Discounted CFR implementation.

    Faster convergence than Vanilla CFR by discounting older iterations.
    """

    def __init__(
        self,
        game: Game,
        alpha: float = 1.5,
        beta: float = 0.5,
        gamma: float = 2.0,
    ):
        """
        Initialize DCFR solver.

        Args:
            game: Game to solve
            alpha: Positive regret discount power
            beta: Negative regret discount power
            gamma: Strategy averaging power
        """
        super().__init__(game)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        # Cache for current iteration's strategy (fixed during iteration)
        self._iteration_strategy: dict[str, np.ndarray] = {}

    def _cache_current_strategies(self) -> None:
        """Cache current strategies from regrets (fixed for entire iteration)."""
        self._iteration_strategy.clear()
        for key in self.regret_sum:
            regrets = self.regret_sum[key]
            self._iteration_strategy[key] = regret_matching(regrets)

    def _get_cached_strategy(self, key: str, num_actions: int) -> np.ndarray:
        """Get cached strategy for this iteration, or uniform if not seen yet."""
        if key in self._iteration_strategy:
            return self._iteration_strategy[key][:num_actions]
        return np.ones(num_actions) / num_actions

    def solve(
        self,
        iterations: int,
        convergence_threshold: float = 1e-6,
        callback: Optional[Callable[[int, float], None]] = None,
        eval_every: int = 100,
        show_progress: bool = True,
    ) -> SolverResult:
        """
        Run DCFR for specified iterations.

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
            iterator = tqdm(iterator, desc="DCFR")

        for t in iterator:
            self.iterations_run = t + 1

            # Compute discount factors for this iteration
            pos_discount = self._positive_regret_discount(t + 1)
            neg_discount = self._negative_regret_discount(t + 1)
            strat_discount = self._strategy_discount(t + 1)

            # Apply discounting to existing values BEFORE new iteration
            if t > 0:
                self._apply_discounts(pos_discount, neg_discount, strat_discount)

            # Cache current strategies at start of iteration (CRITICAL!)
            self._cache_current_strategies()

            # Traverse for each player
            for player in range(self.game.num_players):
                initial_state = self.game.initial_state()
                reach_probs = np.ones(self.game.num_players, dtype=np.float64)

                self._cfr_traverse(
                    state=initial_state,
                    player=player,
                    reach_probs=reach_probs,
                    chance_prob=1.0,
                    iteration=t + 1,
                )

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

    def _positive_regret_discount(self, t: int) -> float:
        """
        Discount factor for positive regrets.
        weight = t^α / (t^α + 1)
        """
        t_alpha = t ** self.alpha
        return t_alpha / (t_alpha + 1)

    def _negative_regret_discount(self, t: int) -> float:
        """
        Discount factor for negative regrets.
        weight = t^β / (t^β + 1)
        """
        t_beta = t ** self.beta
        return t_beta / (t_beta + 1)

    def _strategy_discount(self, t: int) -> float:
        """
        Discount factor for strategy averaging.
        weight = (t / (t+1))^γ
        """
        return (t / (t + 1)) ** self.gamma

    def _apply_discounts(
        self,
        pos_discount: float,
        neg_discount: float,
        strat_discount: float,
    ) -> None:
        """
        Apply discounting to all accumulated values.
        """
        for key in self.regret_sum:
            regrets = self.regret_sum[key]
            # Apply different discounts to positive and negative regrets
            self.regret_sum[key] = np.where(
                regrets > 0,
                regrets * pos_discount,
                regrets * neg_discount,
            )

        for key in self.strategy_sum:
            self.strategy_sum[key] *= strat_discount

    def _cfr_traverse(
        self,
        state: GameState,
        player: int,
        reach_probs: np.ndarray,
        chance_prob: float,
        iteration: int,
    ) -> float:
        """
        Recursive DCFR traversal.

        Similar to Vanilla CFR, but new regrets/strategies are added
        with weight 1 (discounting happens at start of next iteration).

        Args:
            state: Current game state
            player: Player we're computing regrets for
            reach_probs: Reach probabilities for each player
            chance_prob: Product of chance probabilities to reach this state
            iteration: Current iteration number

        Returns:
            Expected value for the traversing player
        """
        # Terminal state: return payoff
        if self.game.is_terminal(state):
            return self.game.get_payoffs(state)[player]

        # Chance node: expected value over outcomes
        if self.game.is_chance_node(state):
            value = 0.0
            for next_state, prob in self.game.chance_outcomes(state):
                value += prob * self._cfr_traverse(
                    next_state, player, reach_probs, chance_prob * prob, iteration
                )
            return value

        # Decision node
        current_player = self.game.current_player(state)
        infoset = self.game.get_infoset(state, current_player)
        key = infoset.to_key()
        actions = self.game.get_actions(state)
        num_actions = len(actions)

        # Get CACHED strategy (fixed for this iteration)
        strategy = self._get_cached_strategy(key, num_actions)

        # Compute action values
        action_values = np.zeros(num_actions, dtype=np.float64)

        for i, action in enumerate(actions):
            next_state = self.game.apply_action(state, action)

            new_reach = reach_probs.copy()
            new_reach[current_player] *= strategy[i]

            action_values[i] = self._cfr_traverse(
                next_state, player, new_reach, chance_prob, iteration
            )

        # Node value under current strategy
        node_value = np.dot(strategy, action_values)

        # Update for traversing player only
        if current_player == player:
            # Counterfactual reach includes chance probability and opponent reach
            opp_reach = chance_prob
            for p in range(self.game.num_players):
                if p != player:
                    opp_reach *= reach_probs[p]

            # Add new regrets (will be discounted next iteration)
            regret_update = self._get_regrets(key)
            for i in range(num_actions):
                regret_update[i] += opp_reach * (action_values[i] - node_value)

            # Add to strategy sum (will be discounted next iteration)
            strategy_update = self._get_strategy_sum(key)
            for i in range(num_actions):
                strategy_update[i] += reach_probs[player] * strategy[i]

            # Track reach probability
            self.reach_probs[key] = reach_probs[player]

        return node_value


class LinearCFR(CFRSolver):
    """
    Linear CFR variant.

    Uses simple linear weighting: iteration t contributes with weight t.
    Simpler than DCFR but still faster than Vanilla CFR.
    """

    def __init__(self, game: Game):
        super().__init__(game)
        self._iteration_strategy: dict[str, np.ndarray] = {}

    def _cache_current_strategies(self) -> None:
        """Cache current strategies from regrets."""
        self._iteration_strategy.clear()
        for key in self.regret_sum:
            self._iteration_strategy[key] = regret_matching(self.regret_sum[key])

    def _get_cached_strategy(self, key: str, num_actions: int) -> np.ndarray:
        """Get cached strategy or uniform."""
        if key in self._iteration_strategy:
            return self._iteration_strategy[key][:num_actions]
        return np.ones(num_actions) / num_actions

    def solve(
        self,
        iterations: int,
        convergence_threshold: float = 1e-6,
        callback: Optional[Callable[[int, float], None]] = None,
        eval_every: int = 100,
        show_progress: bool = True,
    ) -> SolverResult:
        """
        Run Linear CFR for specified iterations.
        """
        iterator = range(iterations)
        if show_progress:
            iterator = tqdm(iterator, desc="Linear CFR")

        for t in iterator:
            self.iterations_run = t + 1
            weight = t + 1  # Linear weight

            # Cache current strategies at start of iteration
            self._cache_current_strategies()

            for player in range(self.game.num_players):
                initial_state = self.game.initial_state()
                reach_probs = np.ones(self.game.num_players, dtype=np.float64)

                self._cfr_traverse(
                    state=initial_state,
                    player=player,
                    reach_probs=reach_probs,
                    chance_prob=1.0,
                    weight=weight,
                )

            if (t + 1) % eval_every == 0:
                exploitability = self.get_exploitability()

                if show_progress:
                    iterator.set_postfix({"exploit": f"{exploitability:.6f}"})

                if callback:
                    callback(t + 1, exploitability)

                if exploitability < convergence_threshold:
                    break

        return self.get_result()

    def _cfr_traverse(
        self,
        state: GameState,
        player: int,
        reach_probs: np.ndarray,
        chance_prob: float,
        weight: float,
    ) -> float:
        """Linear CFR traversal with weighted updates."""
        if self.game.is_terminal(state):
            return self.game.get_payoffs(state)[player]

        if self.game.is_chance_node(state):
            value = 0.0
            for next_state, prob in self.game.chance_outcomes(state):
                value += prob * self._cfr_traverse(
                    next_state, player, reach_probs, chance_prob * prob, weight
                )
            return value

        current_player = self.game.current_player(state)
        infoset = self.game.get_infoset(state, current_player)
        key = infoset.to_key()
        actions = self.game.get_actions(state)
        num_actions = len(actions)

        # Get CACHED strategy (fixed for this iteration)
        strategy = self._get_cached_strategy(key, num_actions)

        action_values = np.zeros(num_actions, dtype=np.float64)

        for i, action in enumerate(actions):
            next_state = self.game.apply_action(state, action)
            new_reach = reach_probs.copy()
            new_reach[current_player] *= strategy[i]
            action_values[i] = self._cfr_traverse(
                next_state, player, new_reach, chance_prob, weight
            )

        node_value = np.dot(strategy, action_values)

        if current_player == player:
            # Counterfactual reach includes chance probability
            opp_reach = chance_prob
            for p in range(self.game.num_players):
                if p != player:
                    opp_reach *= reach_probs[p]

            # Weighted regret update
            regret_update = self._get_regrets(key)
            for i in range(num_actions):
                regret_update[i] += weight * opp_reach * (action_values[i] - node_value)

            # Weighted strategy update
            strategy_update = self._get_strategy_sum(key)
            for i in range(num_actions):
                strategy_update[i] += weight * reach_probs[player] * strategy[i]

            self.reach_probs[key] = reach_probs[player]

        return node_value


class CFRPlus(CFRSolver):
    """
    CFR+ implementation (Tammelin et al., 2015).

    Key improvements over Vanilla CFR:
    1. Floor regrets at 0 after each iteration (no negative regrets)
    2. Use alternating per-player traversals
    3. Cache strategy at start of iteration

    This prevents the oscillation problem where negative regrets cause
    policies to flip back and forth, preventing convergence.

    Reference:
    Tammelin, O. (2014). "Solving Large Imperfect Information Games
    Using CFR+"
    """

    def __init__(self, game: Game):
        super().__init__(game)
        self._iteration_strategy: dict[str, np.ndarray] = {}

    def _cache_current_strategies(self) -> None:
        """Cache current strategies from regrets (fixed for entire iteration)."""
        self._iteration_strategy.clear()
        for key in self.regret_sum:
            self._iteration_strategy[key] = regret_matching(self.regret_sum[key])

    def _get_cached_strategy(self, key: str, num_actions: int) -> np.ndarray:
        """Get cached strategy or uniform."""
        if key in self._iteration_strategy:
            return self._iteration_strategy[key][:num_actions]
        return np.ones(num_actions) / num_actions

    def _floor_regrets(self) -> None:
        """Floor all regrets at 0 (the key CFR+ innovation)."""
        for key in self.regret_sum:
            self.regret_sum[key] = np.maximum(self.regret_sum[key], 0.0)

    def solve(
        self,
        iterations: int,
        convergence_threshold: float = 1e-6,
        callback: Optional[Callable[[int, float], None]] = None,
        eval_every: int = 100,
        show_progress: bool = True,
    ) -> SolverResult:
        """
        Run CFR+ for specified iterations.
        """
        iterator = range(iterations)
        if show_progress:
            iterator = tqdm(iterator, desc="CFR+")

        for t in iterator:
            self.iterations_run = t + 1
            weight = t + 1  # Linear weighting like Linear CFR

            # Cache current strategies at start of iteration
            self._cache_current_strategies()

            # Alternating traversals for each player
            for player in range(self.game.num_players):
                initial_state = self.game.initial_state()
                reach_probs = np.ones(self.game.num_players, dtype=np.float64)

                self._cfr_traverse(
                    state=initial_state,
                    player=player,
                    reach_probs=reach_probs,
                    chance_prob=1.0,
                    weight=weight,
                )

            # CFR+ key innovation: floor regrets at 0 after each iteration
            self._floor_regrets()

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

    def _cfr_traverse(
        self,
        state: GameState,
        player: int,
        reach_probs: np.ndarray,
        chance_prob: float,
        weight: float,
    ) -> float:
        """CFR+ traversal with weighted updates."""
        if self.game.is_terminal(state):
            return self.game.get_payoffs(state)[player]

        if self.game.is_chance_node(state):
            value = 0.0
            for next_state, prob in self.game.chance_outcomes(state):
                # Propagate chance probability
                value += prob * self._cfr_traverse(
                    next_state, player, reach_probs, chance_prob * prob, weight
                )
            return value

        current_player = self.game.current_player(state)
        infoset = self.game.get_infoset(state, current_player)
        key = infoset.to_key()
        actions = self.game.get_actions(state)
        num_actions = len(actions)

        # Get CACHED strategy (fixed for this iteration)
        strategy = self._get_cached_strategy(key, num_actions)

        action_values = np.zeros(num_actions, dtype=np.float64)

        for i, action in enumerate(actions):
            next_state = self.game.apply_action(state, action)
            new_reach = reach_probs.copy()
            new_reach[current_player] *= strategy[i]
            action_values[i] = self._cfr_traverse(
                next_state, player, new_reach, chance_prob, weight
            )

        node_value = np.dot(strategy, action_values)

        if current_player == player:
            # Counterfactual reach includes chance probability
            opp_reach = chance_prob
            for p in range(self.game.num_players):
                if p != player:
                    opp_reach *= reach_probs[p]

            # Weighted regret update (regrets will be floored after iteration)
            regret_update = self._get_regrets(key)
            for i in range(num_actions):
                regret_update[i] += opp_reach * (action_values[i] - node_value)

            # Weighted strategy update
            strategy_update = self._get_strategy_sum(key)
            for i in range(num_actions):
                strategy_update[i] += weight * reach_probs[player] * strategy[i]

            self.reach_probs[key] = reach_probs[player]

        return node_value