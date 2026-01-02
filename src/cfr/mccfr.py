"""
Monte Carlo Counterfactual Regret Minimization (MCCFR)

For games too large to enumerate completely (like NLHE), we use
sampling-based variants of CFR:

1. External Sampling: Sample opponent actions and chance, traverse all our actions
2. Outcome Sampling: Sample entire trajectory, weight by probability

This implementation uses External Sampling (most common for poker).
"""

from typing import Dict, List, Tuple, Optional, Callable
import numpy as np
from dataclasses import dataclass, field
from collections import defaultdict
from tqdm import tqdm

from ..games.base import Game, GameState, InfoSet


@dataclass
class MCCFRResult:
    """Result of MCCFR training."""
    iterations: int
    avg_strategy: Dict[str, np.ndarray]
    exploitability: Optional[float] = None
    
    # Training history
    regret_history: List[float] = field(default_factory=list)


class ExternalSamplingMCCFR:
    """
    External Sampling MCCFR.
    
    Algorithm:
    - Sample opponent and chance nodes
    - Traverse all actions at current player's nodes
    - Accumulate regrets and average strategy
    
    This is O(|A|) per iteration instead of O(|A|^d) for vanilla CFR,
    making it tractable for large games.
    """
    
    def __init__(
        self,
        game: Game,
        get_infoset_key: Optional[Callable[[GameState, int], str]] = None,
        use_cfr_plus: bool = True,
        linear_averaging: bool = True,
    ):
        """
        Args:
            game: The game to solve
            get_infoset_key: Custom function to get infoset key (for abstraction)
            use_cfr_plus: Use CFR+ (floor regrets at 0)
            linear_averaging: Weight strategies by iteration
        """
        self.game = game
        self.get_infoset_key = get_infoset_key or self._default_infoset_key
        self.use_cfr_plus = use_cfr_plus
        self.linear_averaging = linear_averaging
        
        # Regret and strategy tables
        self.regret_sum: Dict[str, np.ndarray] = defaultdict(
            lambda: np.zeros(game.num_actions, dtype=np.float64)
        )
        self.strategy_sum: Dict[str, np.ndarray] = defaultdict(
            lambda: np.zeros(game.num_actions, dtype=np.float64)
        )
        
        # Track which actions are legal at each infoset
        self.legal_actions: Dict[str, List[int]] = {}
        
        self.iteration = 0
        self._rng = np.random.default_rng()
    
    def _default_infoset_key(self, state: GameState, player: int) -> str:
        """Default infoset key generation."""
        infoset = self.game.get_infoset(state, player)
        return infoset.to_key()
    
    def get_strategy(self, infoset_key: str, legal_actions: List[int]) -> np.ndarray:
        """
        Get current strategy via regret matching.
        
        Returns uniform strategy over legal actions if no positive regrets.
        """
        regrets = self.regret_sum[infoset_key]
        
        # Only consider legal actions
        strategy = np.zeros(self.game.num_actions)
        positive_regrets = np.maximum(regrets, 0)
        
        legal_positive = positive_regrets[legal_actions]
        total = legal_positive.sum()
        
        if total > 0:
            strategy[legal_actions] = legal_positive / total
        else:
            # Uniform over legal actions
            strategy[legal_actions] = 1.0 / len(legal_actions)
        
        return strategy
    
    def get_average_strategy(self, infoset_key: str, legal_actions: List[int]) -> np.ndarray:
        """Get average strategy over all iterations."""
        strategy_sum = self.strategy_sum[infoset_key]
        
        strategy = np.zeros(self.game.num_actions)
        legal_sum = strategy_sum[legal_actions]
        total = legal_sum.sum()
        
        if total > 0:
            strategy[legal_actions] = legal_sum / total
        else:
            strategy[legal_actions] = 1.0 / len(legal_actions)
        
        return strategy
    
    def solve(
        self,
        iterations: int = 10000,
        hands_per_iteration: int = 1,
        seed: Optional[int] = None,
        verbose: bool = True,
    ) -> MCCFRResult:
        """
        Run MCCFR training.
        
        Args:
            iterations: Number of iterations (each samples one game tree)
            hands_per_iteration: Number of hands to sample per iteration
            seed: Random seed
            verbose: Show progress bar
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        
        iterator = range(iterations)
        if verbose:
            iterator = tqdm(iterator, desc="MCCFR")
        
        for i in iterator:
            self.iteration = i + 1
            
            for _ in range(hands_per_iteration):
                # Sample initial state (deal cards)
                state = self._sample_initial_state()
                
                # Traverse for each player
                for player in range(self.game.num_players):
                    self._external_cfr(state, player, 1.0, 1.0)
        
        return MCCFRResult(
            iterations=iterations,
            avg_strategy={k: v.copy() for k, v in self.strategy_sum.items()},
        )
    
    def _sample_initial_state(self) -> GameState:
        """Sample an initial state (deal cards)."""
        state = self.game.initial_state()
        
        # Handle chance nodes
        while self.game.is_chance_node(state):
            outcomes = self.game.chance_outcomes(state)
            probs = np.array([p for _, p in outcomes])
            idx = self._rng.choice(len(outcomes), p=probs)
            state = outcomes[idx][0]
        
        return state
    
    def _external_cfr(
        self,
        state: GameState,
        traversing_player: int,
        pi_traverser: float,
        pi_opponent: float,
    ) -> float:
        """
        External sampling CFR traversal.
        
        Args:
            state: Current game state
            traversing_player: Player whose regrets we're updating
            pi_traverser: Probability of traverser reaching this state
            pi_opponent: Probability of opponent reaching this state
        
        Returns:
            Expected value for traversing player
        """
        if self.game.is_terminal(state):
            payoffs = self.game.get_payoffs(state)
            return payoffs[traversing_player]
        
        if self.game.is_chance_node(state):
            # Sample one chance outcome
            outcomes = self.game.chance_outcomes(state)
            probs = np.array([p for _, p in outcomes])
            idx = self._rng.choice(len(outcomes), p=probs)
            next_state, prob = outcomes[idx]
            return self._external_cfr(next_state, traversing_player, pi_traverser, pi_opponent)
        
        current_player = self.game.current_player(state)
        infoset_key = self.get_infoset_key(state, current_player)
        legal_actions = self.game.get_actions(state)
        
        # Cache legal actions
        self.legal_actions[infoset_key] = legal_actions
        
        # Get current strategy
        strategy = self.get_strategy(infoset_key, legal_actions)
        
        if current_player == traversing_player:
            # Traverse all actions
            action_values = np.zeros(self.game.num_actions)
            
            for action in legal_actions:
                next_state = self.game.apply_action(state, action)
                action_values[action] = self._external_cfr(
                    next_state, traversing_player,
                    pi_traverser * strategy[action], pi_opponent
                )
            
            # Expected value under current strategy
            ev = np.sum(strategy * action_values)
            
            # Update regrets
            for action in legal_actions:
                regret = action_values[action] - ev
                self.regret_sum[infoset_key][action] += pi_opponent * regret
                
                if self.use_cfr_plus:
                    self.regret_sum[infoset_key][action] = max(
                        0, self.regret_sum[infoset_key][action]
                    )
            
            # Update strategy sum for averaging
            weight = self.iteration if self.linear_averaging else 1
            self.strategy_sum[infoset_key] += weight * pi_traverser * strategy
            
            return ev
        else:
            # Sample opponent action
            action = self._rng.choice(legal_actions, p=strategy[legal_actions])
            next_state = self.game.apply_action(state, action)
            
            return self._external_cfr(
                next_state, traversing_player,
                pi_traverser, pi_opponent * strategy[action]
            )
    
    def get_all_strategies(self) -> Dict[str, Tuple[List[int], np.ndarray]]:
        """Get average strategy for all infosets."""
        result = {}
        for key in self.strategy_sum:
            legal = self.legal_actions.get(key, list(range(self.game.num_actions)))
            result[key] = (legal, self.get_average_strategy(key, legal))
        return result


class OutcomeSamplingMCCFR(ExternalSamplingMCCFR):
    """
    Outcome Sampling MCCFR.
    
    Even faster than external sampling - samples entire trajectory.
    Weights updates by 1/probability of sampled trajectory.
    
    Useful for very large games where even external sampling is too slow.
    """
    
    def _outcome_cfr(
        self,
        state: GameState,
        traversing_player: int,
        reach_player: float,
        reach_opponent: float,
        sample_prob: float,
    ) -> float:
        """
        Outcome sampling CFR traversal.
        
        Samples one action at each node, weights by 1/sample_prob.
        """
        if self.game.is_terminal(state):
            payoffs = self.game.get_payoffs(state)
            return payoffs[traversing_player] / sample_prob
        
        if self.game.is_chance_node(state):
            outcomes = self.game.chance_outcomes(state)
            probs = np.array([p for _, p in outcomes])
            idx = self._rng.choice(len(outcomes), p=probs)
            next_state, prob = outcomes[idx]
            return self._outcome_cfr(
                next_state, traversing_player,
                reach_player, reach_opponent, sample_prob * prob
            )
        
        current_player = self.game.current_player(state)
        infoset_key = self.get_infoset_key(state, current_player)
        legal_actions = self.game.get_actions(state)
        
        self.legal_actions[infoset_key] = legal_actions
        strategy = self.get_strategy(infoset_key, legal_actions)
        
        # Sample action
        action = self._rng.choice(legal_actions, p=strategy[legal_actions])
        action_prob = strategy[action]
        
        next_state = self.game.apply_action(state, action)
        
        if current_player == traversing_player:
            # Recurse
            cv = self._outcome_cfr(
                next_state, traversing_player,
                reach_player * action_prob, reach_opponent,
                sample_prob * action_prob
            )
            
            # Counterfactual value for not taking this action
            # (approximated by the sample)
            ev = cv * action_prob
            
            # Update regrets (using sampling correction)
            for a in legal_actions:
                if a == action:
                    regret = cv * (1 - action_prob)
                else:
                    regret = -cv * strategy[a]
                self.regret_sum[infoset_key][a] += reach_opponent * regret
                
                if self.use_cfr_plus:
                    self.regret_sum[infoset_key][a] = max(0, self.regret_sum[infoset_key][a])
            
            # Update strategy sum
            weight = self.iteration if self.linear_averaging else 1
            self.strategy_sum[infoset_key] += weight * reach_player * strategy / sample_prob
            
            return ev
        else:
            return self._outcome_cfr(
                next_state, traversing_player,
                reach_player, reach_opponent * action_prob,
                sample_prob * action_prob
            )
    
    def solve(
        self,
        iterations: int = 10000,
        hands_per_iteration: int = 1,
        seed: Optional[int] = None,
        verbose: bool = True,
    ) -> MCCFRResult:
        """Run Outcome Sampling MCCFR."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        
        iterator = range(iterations)
        if verbose:
            iterator = tqdm(iterator, desc="OS-MCCFR")
        
        for i in iterator:
            self.iteration = i + 1
            
            for _ in range(hands_per_iteration):
                state = self._sample_initial_state()
                
                for player in range(self.game.num_players):
                    self._outcome_cfr(state, player, 1.0, 1.0, 1.0)
        
        return MCCFRResult(
            iterations=iterations,
            avg_strategy={k: v.copy() for k, v in self.strategy_sum.items()},
        )


if __name__ == "__main__":
    # Quick test with Leduc
    import sys
    sys.path.insert(0, '.')
    from src.games import LeducPoker
    
    game = LeducPoker()
    mccfr = ExternalSamplingMCCFR(game, use_cfr_plus=True)
    
    print("Running External Sampling MCCFR on Leduc...")
    result = mccfr.solve(iterations=1000, verbose=True)
    
    print(f"\nFound {len(result.avg_strategy)} infosets")
    print("Sample strategies:")
    for key in list(result.avg_strategy.keys())[:5]:
        legal = mccfr.legal_actions.get(key, [0, 1, 2])
        strat = mccfr.get_average_strategy(key, legal)
        print(f"  {key[:50]}... → {strat[legal]}")

