"""
Parameterized Leduc Poker - Scalable complexity for scaling law experiments.

Parameters:
- num_ranks: Number of card ranks (default 3: J, Q, K)
- num_suits: Suits per rank (default 2)
- max_raises: Max raises per betting round (default 2)
- num_rounds: Number of betting rounds (default 2: preflop + flop)

Complexity scaling:
- Standard Leduc (3,2,2,2): ~288 infosets
- Leduc-4 (4,2,2,2): ~512 infosets  
- Leduc-5 (5,2,2,2): ~800 infosets
- Leduc-6 (6,2,2,2): ~1,152 infosets
- Leduc-6-3suits (6,3,2,2): ~2,592 infosets
- Leduc-8-3raises (8,2,3,2): ~5,000+ infosets
- Leduc-10 (10,2,2,2): ~10,000+ infosets
"""

from typing import List, Tuple, Optional
import numpy as np

from .base import Game, GameState, InfoSet


class ParameterizedLeduc(Game):
    """
    Parameterized Leduc Poker with configurable complexity.
    
    Args:
        num_ranks: Number of distinct card ranks (e.g., 3 for J,Q,K)
        num_suits: Number of suits per rank (e.g., 2 for two of each)
        max_raises: Maximum raises allowed per betting round
        num_rounds: Number of betting rounds (1=preflop only, 2=preflop+flop)
        ante: Initial ante per player
        raise_sizes: Raise amount per round (list of length num_rounds)
    """
    
    # Action constants
    FOLD = 0
    CALL = 1
    RAISE = 2
    
    def __init__(
        self,
        num_ranks: int = 3,
        num_suits: int = 2,
        max_raises: int = 2,
        num_rounds: int = 2,
        ante: int = 1,
        raise_sizes: Optional[List[int]] = None,
    ):
        self._num_ranks = num_ranks
        self._num_suits = num_suits
        self._max_raises = max_raises
        self._num_rounds = num_rounds
        self._ante = ante
        
        # Default raise sizes: 2 for round 1, 4 for round 2, etc.
        if raise_sizes is None:
            self._raise_sizes = [2 * (r + 1) for r in range(num_rounds)]
        else:
            self._raise_sizes = raise_sizes
        
        # Build card deck: (rank, suit) tuples
        self._all_cards = [(r, s) for r in range(num_ranks) for s in range(num_suits)]
        self._num_cards = len(self._all_cards)
        
        # Rank names
        base_names = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
        if num_ranks <= len(base_names):
            self._rank_names = base_names[-num_ranks:]
        else:
            self._rank_names = [str(i) for i in range(num_ranks)]
    
    @property
    def num_players(self) -> int:
        return 2
    
    @property
    def num_actions(self) -> int:
        return 3  # Fold, Call/Check, Raise
    
    @property
    def num_ranks(self) -> int:
        return self._num_ranks
    
    @property
    def num_suits(self) -> int:
        return self._num_suits
    
    @property
    def max_raises(self) -> int:
        return self._max_raises
    
    @property
    def num_rounds(self) -> int:
        return self._num_rounds
    
    @property
    def name(self) -> str:
        return f"leduc_{self._num_ranks}r{self._num_suits}s{self._max_raises}x{self._num_rounds}rd"
    
    @property
    def complexity_name(self) -> str:
        """Human-readable complexity label."""
        return f"Leduc-{self._num_ranks}"
    
    def estimate_num_infosets(self) -> int:
        """
        Rough estimate of number of information sets.
        
        Formula approximation:
        - 2 players
        - num_ranks private cards
        - (num_ranks + 1) public states per round (including "not dealt")
        - O(3^max_raises) betting histories per round
        - num_rounds
        
        This is an upper bound; actual count depends on legal action sequences.
        """
        # Per round: roughly 2 * num_ranks * (num_ranks+1) * (betting combos)
        # Betting combos: check-check, bet-fold, bet-call, bet-raise-fold, etc.
        betting_states = sum(3**r for r in range(self._max_raises + 1))
        per_round = 2 * self._num_ranks * betting_states
        
        if self._num_rounds == 1:
            return per_round
        else:
            # Round 2 adds public card dimension
            return per_round * (self._num_ranks + 1) * 2
    
    def initial_state(self) -> GameState:
        """Return initial state (chance node for dealing)."""
        return GameState(
            player_cards=(),
            public_card=None,
            pot=2 * self._ante,
            current_player=-1,  # Chance node
            bets=(self._ante, self._ante),
            stacks=(0, 0),
            round_num=0,
            num_raises=0,
            history=(),
            is_terminal=False,
        )
    
    def is_chance_node(self, state: GameState) -> bool:
        """Check if current state is a chance node."""
        # Initial deal
        if len(state.player_cards) == 0:
            return True
        
        # Between rounds - community card not dealt
        if self._num_rounds >= 2:
            if state.public_card is None and state.round_num >= 1:
                return True
        
        return False
    
    def chance_outcomes(self, state: GameState) -> List[Tuple[GameState, float]]:
        """Get all possible chance outcomes with probabilities."""
        if not self.is_chance_node(state):
            return []
        
        outcomes = []
        
        if len(state.player_cards) == 0:
            # Initial deal: deal 2 cards
            n = self._num_cards
            prob = 1.0 / (n * (n - 1))  # Ordered dealing
            
            for p1_card in self._all_cards:
                for p2_card in self._all_cards:
                    if p1_card == p2_card:
                        continue
                    
                    new_state = state.copy()
                    new_state.player_cards = (p1_card, p2_card)
                    new_state.current_player = 0
                    outcomes.append((new_state, prob))
        
        elif state.public_card is None and state.round_num >= 1:
            # Deal community card
            dealt = set(state.player_cards)
            remaining = [c for c in self._all_cards if c not in dealt]
            prob = 1.0 / len(remaining)
            
            for card in remaining:
                new_state = state.copy()
                new_state.public_card = card
                new_state.current_player = 0
                new_state.num_raises = 0
                outcomes.append((new_state, prob))
        
        return outcomes
    
    def current_player(self, state: GameState) -> int:
        """Get current player to act."""
        if self.is_chance_node(state):
            return -1
        return state.current_player
    
    def get_infoset(self, state: GameState, player: int) -> InfoSet:
        """Get information set for a player."""
        private_card = state.player_cards[player]
        private_rank = private_card[0]
        
        public_rank = state.public_card[0] if state.public_card else -1
        
        return InfoSet(
            player=player,
            public_state=(public_rank, state.pot, state.round_num),
            private_info=(private_rank,),
            history=state.history,
        )
    
    def get_actions(self, state: GameState) -> List[int]:
        """Get legal actions at current state."""
        actions = []
        facing_bet = self._is_facing_bet(state)
        
        if facing_bet:
            actions.append(self.FOLD)
            actions.append(self.CALL)
            if state.num_raises < self._max_raises:
                actions.append(self.RAISE)
        else:
            actions.append(self.CALL)  # Check
            if state.num_raises < self._max_raises:
                actions.append(self.RAISE)
        
        return actions
    
    def _is_facing_bet(self, state: GameState) -> bool:
        """Check if current player is facing a bet/raise."""
        round_history = self._get_current_round_history(state)
        if len(round_history) == 0:
            return False
        return round_history[-1] == 'r'
    
    def _get_current_round_history(self, state: GameState) -> Tuple[str, ...]:
        """Get actions from current betting round only."""
        if state.round_num == 0:
            return state.history
        
        # Find round separator
        history = state.history
        for i, action in enumerate(history):
            if action == '/':
                return history[i+1:]
        
        return ()
    
    def apply_action(self, state: GameState, action: int) -> GameState:
        """Apply action and return new state."""
        new_state = state.copy()
        history = list(state.history)
        current = state.current_player
        opponent = 1 - current
        
        raise_size = self._raise_sizes[min(state.round_num, len(self._raise_sizes) - 1)]
        
        # Convert action to string
        action_str = {self.FOLD: 'f', self.CALL: 'c', self.RAISE: 'r'}[action]
        history.append(action_str)
        new_state.history = tuple(history)
        
        if action == self.FOLD:
            new_state.is_terminal = True
            new_state.terminal_type = "fold"
            new_state.winner = opponent
        
        elif action == self.CALL:
            bets = list(state.bets)
            bets[current] = bets[opponent]
            new_state.bets = tuple(bets)
            new_state.pot = sum(bets)
            
            round_history = self._get_current_round_history(new_state)
            
            if self._is_round_complete(round_history):
                if state.round_num < self._num_rounds - 1:
                    # More rounds to go
                    new_state.round_num = state.round_num + 1
                    new_state.num_raises = 0
                    history.append('/')
                    new_state.history = tuple(history)
                else:
                    # Final round - showdown
                    new_state.is_terminal = True
                    new_state.terminal_type = "showdown"
            else:
                new_state.current_player = opponent
        
        elif action == self.RAISE:
            bets = list(state.bets)
            bets[current] = bets[opponent] + raise_size
            new_state.bets = tuple(bets)
            new_state.pot = sum(bets)
            new_state.num_raises = state.num_raises + 1
            new_state.current_player = opponent
        
        return new_state
    
    def _is_round_complete(self, round_history: Tuple[str, ...]) -> bool:
        """Check if current betting round is complete."""
        if len(round_history) < 2:
            return False
        return round_history[-1] == 'c'
    
    def is_terminal(self, state: GameState) -> bool:
        """Check if game is over."""
        return state.is_terminal
    
    def get_payoffs(self, state: GameState) -> np.ndarray:
        """Get payoffs at terminal state."""
        if not state.is_terminal:
            raise ValueError("Cannot get payoffs for non-terminal state")
        
        payoffs = np.zeros(2)
        
        if state.terminal_type == "fold":
            winner = state.winner
            loser = 1 - winner
            loser_bet = state.bets[loser]
            payoffs[winner] = loser_bet
            payoffs[loser] = -loser_bet
        
        elif state.terminal_type == "showdown":
            winner = self._determine_winner(state)
            loser = 1 - winner
            bet_amount = state.bets[loser]
            payoffs[winner] = bet_amount
            payoffs[loser] = -bet_amount
        
        return payoffs
    
    def _determine_winner(self, state: GameState) -> int:
        """Determine winner at showdown."""
        p1_rank = state.player_cards[0][0]
        p2_rank = state.player_cards[1][0]
        
        if self._num_rounds >= 2 and state.public_card:
            public_rank = state.public_card[0]
            p1_pair = (p1_rank == public_rank)
            p2_pair = (p2_rank == public_rank)
            
            if p1_pair and not p2_pair:
                return 0
            elif p2_pair and not p1_pair:
                return 1
            # Both pair or neither - high card
        
        return 0 if p1_rank > p2_rank else 1
    
    def get_action_name(self, action: int) -> str:
        """Convert action to string."""
        names = {self.FOLD: 'fold', self.CALL: 'call/check', self.RAISE: 'raise'}
        return names.get(action, f'action_{action}')
    
    def card_name(self, card: Tuple[int, int]) -> str:
        """Convert card to string."""
        rank, suit = card
        return f"{self._rank_names[rank]}{suit}"
    
    def __repr__(self) -> str:
        return (f"ParameterizedLeduc(ranks={self._num_ranks}, suits={self._num_suits}, "
                f"raises={self._max_raises}, rounds={self._num_rounds})")


# Pre-configured complexity levels for experiments
COMPLEXITY_CONFIGS = {
    "C1": {"num_ranks": 3, "num_suits": 2, "max_raises": 2},   # ~288 infosets (standard Leduc)
    "C2": {"num_ranks": 4, "num_suits": 2, "max_raises": 2},   # ~512 infosets
    "C3": {"num_ranks": 5, "num_suits": 2, "max_raises": 2},   # ~800 infosets
    "C4": {"num_ranks": 6, "num_suits": 2, "max_raises": 2},   # ~1,200 infosets
    "C5": {"num_ranks": 6, "num_suits": 3, "max_raises": 2},   # ~2,500 infosets
    "C6": {"num_ranks": 8, "num_suits": 2, "max_raises": 3},   # ~5,000 infosets
    "C7": {"num_ranks": 10, "num_suits": 2, "max_raises": 2},  # ~8,000 infosets
    "C8": {"num_ranks": 13, "num_suits": 2, "max_raises": 2},  # ~15,000 infosets
}


def create_leduc(complexity: str = "C1") -> ParameterizedLeduc:
    """Create a Leduc variant by complexity level."""
    if complexity not in COMPLEXITY_CONFIGS:
        raise ValueError(f"Unknown complexity: {complexity}. Choose from {list(COMPLEXITY_CONFIGS.keys())}")
    
    config = COMPLEXITY_CONFIGS[complexity]
    return ParameterizedLeduc(**config)

