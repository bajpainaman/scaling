"""
No-Limit Texas Hold'em Implementation

This is a simplified but complete NLHE implementation for:
- 2 players (heads-up)
- Standard 52-card deck
- 4 betting rounds (preflop, flop, turn, river)
- No-limit betting structure

For tractability, we use:
- Action abstraction (fold, check/call, pot-sized bets)
- Future: Card abstraction for infoset encoding
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import numpy as np
from itertools import combinations

from .base import Game, GameState, InfoSet


# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
SUITS = ['c', 'd', 'h', 's']  # clubs, diamonds, hearts, spades

# Card encoding: 0-12 for 2c-Ac, 13-25 for 2d-Ad, etc.
RANK_TO_INT = {r: i for i, r in enumerate(RANKS)}
SUIT_TO_INT = {s: i for i, s in enumerate(SUITS)}

# Actions
FOLD = 0
CHECK_CALL = 1
BET_HALF_POT = 2
BET_POT = 3
BET_2X_POT = 4
ALL_IN = 5

ACTION_NAMES = ['fold', 'check/call', 'bet_0.5x', 'bet_1x', 'bet_2x', 'all_in']

# Betting rounds
PREFLOP = 0
FLOP = 1
TURN = 2
RIVER = 3

ROUND_NAMES = ['preflop', 'flop', 'turn', 'river']


# ═══════════════════════════════════════════════════════════════════════════
# HAND EVALUATION (7-card evaluator)
# ═══════════════════════════════════════════════════════════════════════════

# Hand rankings (higher = better)
HIGH_CARD = 0
ONE_PAIR = 1
TWO_PAIR = 2
THREE_OF_A_KIND = 3
STRAIGHT = 4
FLUSH = 5
FULL_HOUSE = 6
FOUR_OF_A_KIND = 7
STRAIGHT_FLUSH = 8


def card_to_int(rank: str, suit: str) -> int:
    """Convert rank and suit to integer 0-51."""
    return RANK_TO_INT[rank] * 4 + SUIT_TO_INT[suit]


def int_to_card(card_int: int) -> Tuple[int, int]:
    """Convert integer to (rank_idx, suit_idx)."""
    return card_int // 4, card_int % 4


def get_rank(card: int) -> int:
    """Get rank index (0-12) from card int."""
    return card // 4


def get_suit(card: int) -> int:
    """Get suit index (0-3) from card int."""
    return card % 4


def evaluate_hand(cards: List[int]) -> Tuple[int, List[int]]:
    """
    Evaluate a 5-7 card hand.
    
    Returns:
        (hand_rank, kickers) where hand_rank is 0-8 and kickers break ties
    """
    if len(cards) < 5:
        raise ValueError(f"Need at least 5 cards, got {len(cards)}")
    
    # Try all 5-card combinations and return best
    best_rank = -1
    best_kickers = []
    
    for combo in combinations(cards, 5):
        rank, kickers = _evaluate_5_cards(list(combo))
        if rank > best_rank or (rank == best_rank and kickers > best_kickers):
            best_rank = rank
            best_kickers = kickers
    
    return best_rank, best_kickers


def _evaluate_5_cards(cards: List[int]) -> Tuple[int, List[int]]:
    """Evaluate exactly 5 cards."""
    ranks = sorted([get_rank(c) for c in cards], reverse=True)
    suits = [get_suit(c) for c in cards]
    
    # Check flush
    is_flush = len(set(suits)) == 1
    
    # Check straight
    is_straight, straight_high = _check_straight(ranks)
    
    # Count ranks
    rank_counts = {}
    for r in ranks:
        rank_counts[r] = rank_counts.get(r, 0) + 1
    
    counts = sorted(rank_counts.values(), reverse=True)
    
    # Determine hand type
    if is_straight and is_flush:
        return STRAIGHT_FLUSH, [straight_high]
    
    if counts == [4, 1]:
        quad_rank = [r for r, c in rank_counts.items() if c == 4][0]
        kicker = [r for r, c in rank_counts.items() if c == 1][0]
        return FOUR_OF_A_KIND, [quad_rank, kicker]
    
    if counts == [3, 2]:
        trip_rank = [r for r, c in rank_counts.items() if c == 3][0]
        pair_rank = [r for r, c in rank_counts.items() if c == 2][0]
        return FULL_HOUSE, [trip_rank, pair_rank]
    
    if is_flush:
        return FLUSH, ranks
    
    if is_straight:
        return STRAIGHT, [straight_high]
    
    if counts == [3, 1, 1]:
        trip_rank = [r for r, c in rank_counts.items() if c == 3][0]
        kickers = sorted([r for r, c in rank_counts.items() if c == 1], reverse=True)
        return THREE_OF_A_KIND, [trip_rank] + kickers
    
    if counts == [2, 2, 1]:
        pairs = sorted([r for r, c in rank_counts.items() if c == 2], reverse=True)
        kicker = [r for r, c in rank_counts.items() if c == 1][0]
        return TWO_PAIR, pairs + [kicker]
    
    if counts == [2, 1, 1, 1]:
        pair_rank = [r for r, c in rank_counts.items() if c == 2][0]
        kickers = sorted([r for r, c in rank_counts.items() if c == 1], reverse=True)
        return ONE_PAIR, [pair_rank] + kickers
    
    return HIGH_CARD, ranks


def _check_straight(ranks: List[int]) -> Tuple[bool, int]:
    """Check if sorted ranks form a straight. Returns (is_straight, high_card)."""
    unique = sorted(set(ranks), reverse=True)
    if len(unique) != 5:
        return False, 0
    
    # Regular straight
    if unique[0] - unique[4] == 4:
        return True, unique[0]
    
    # Wheel (A-2-3-4-5)
    if unique == [12, 3, 2, 1, 0]:
        return True, 3  # 5-high
    
    return False, 0


def compare_hands(hand1: List[int], hand2: List[int]) -> int:
    """
    Compare two hands.
    Returns: 1 if hand1 wins, -1 if hand2 wins, 0 if tie
    """
    rank1, kickers1 = evaluate_hand(hand1)
    rank2, kickers2 = evaluate_hand(hand2)
    
    if rank1 > rank2:
        return 1
    if rank1 < rank2:
        return -1
    
    # Same hand rank, compare kickers
    for k1, k2 in zip(kickers1, kickers2):
        if k1 > k2:
            return 1
        if k1 < k2:
            return -1
    
    return 0  # Tie


# ═══════════════════════════════════════════════════════════════════════════
# GAME STATE
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class NLHEState:
    """State of an NLHE game."""
    # Cards
    hole_cards: Tuple[Tuple[int, int], Tuple[int, int]]  # ((p0_c1, p0_c2), (p1_c1, p1_c2))
    board: Tuple[int, ...]  # Community cards (0-5)
    
    # Betting
    stacks: Tuple[int, int]  # Remaining chips
    pot: int  # Total pot
    bets_this_round: Tuple[int, int]  # Current round bets
    
    # Game state
    round: int  # 0=preflop, 1=flop, 2=turn, 3=river
    current_player: int  # 0 or 1
    is_terminal: bool
    winner: Optional[int]  # None if not terminal or tie
    
    # History
    history: Tuple[Tuple[int, ...], ...]  # Actions per round
    
    def to_game_state(self) -> GameState:
        """Convert to base GameState."""
        return GameState(
            is_terminal=self.is_terminal,
            current_player=self.current_player,
            pot=self.pot,
            round_num=self.round,
            history=self.history
        )


# ═══════════════════════════════════════════════════════════════════════════
# NLHE GAME
# ═══════════════════════════════════════════════════════════════════════════

class NLHE(Game):
    """
    Heads-up No-Limit Texas Hold'em.
    
    Simplified for tractability:
    - Fixed bet sizes (0.5x, 1x, 2x pot, all-in)
    - 2 players only
    """
    
    def __init__(
        self,
        starting_stack: int = 100,
        small_blind: int = 1,
        big_blind: int = 2,
        max_raises_per_round: int = 4,
    ):
        self.starting_stack = starting_stack
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.max_raises_per_round = max_raises_per_round
        self._rng = np.random.default_rng()
        
        # Create deck
        self.deck = list(range(52))
    
    @property
    def name(self) -> str:
        return f"NLHE_{self.starting_stack}bb"
    
    @property
    def num_players(self) -> int:
        return 2
    
    @property
    def num_actions(self) -> int:
        return 6  # fold, check/call, 0.5x, 1x, 2x, all-in
    
    def initial_state(self) -> GameState:
        """Create initial state (wraps get_initial_state for base class)."""
        nlhe_state = self.get_initial_state(self._rng)
        return self._to_game_state(nlhe_state)
    
    def _to_game_state(self, state: NLHEState) -> GameState:
        """Convert NLHEState to GameState."""
        return GameState(
            player_cards=(state.hole_cards[0], state.hole_cards[1]),
            public_card=state.board,
            pot=state.pot,
            current_player=state.current_player,
            bets=state.bets_this_round,
            stacks=state.stacks,
            round_num=state.round,
            history=tuple(str(a) for round_hist in state.history for a in round_hist),
            is_terminal=state.is_terminal,
            winner=state.winner,
        )
    
    def _from_game_state(self, state: GameState) -> NLHEState:
        """Convert GameState back to NLHEState (best effort)."""
        # This is lossy - we store full NLHEState in practice
        return NLHEState(
            hole_cards=(state.player_cards[0], state.player_cards[1]),
            board=state.public_card if state.public_card else (),
            stacks=state.stacks,
            pot=state.pot,
            bets_this_round=state.bets,
            round=state.round_num,
            current_player=state.current_player,
            is_terminal=state.is_terminal,
            winner=state.winner,
            history=(tuple(int(a) for a in state.history),) if state.history else ((),),
        )
    
    def get_actions(self, state: GameState) -> List[int]:
        """Get legal actions (base class interface)."""
        nlhe_state = self._from_game_state(state)
        return self.get_legal_actions(nlhe_state)
    
    def is_terminal(self, state: GameState) -> bool:
        """Check if terminal."""
        return state.is_terminal
    
    def get_payoffs(self, state: GameState) -> np.ndarray:
        """Get payoffs for terminal state."""
        nlhe_state = self._from_game_state(state)
        return np.array([
            self.get_payoff(nlhe_state, 0),
            self.get_payoff(nlhe_state, 1)
        ])
    
    def is_chance_node(self, state: GameState) -> bool:
        """NLHE deals all cards upfront, so no mid-game chance nodes."""
        return False
    
    def chance_outcomes(self, state: GameState) -> List[Tuple[GameState, float]]:
        """No chance outcomes mid-game."""
        return []
    
    def current_player(self, state: GameState) -> int:
        """Get current player."""
        return state.current_player
    
    def get_initial_state(self, rng: Optional[np.random.Generator] = None) -> NLHEState:
        """Deal a new hand."""
        if rng is None:
            rng = np.random.default_rng()
        
        # Shuffle and deal
        deck = rng.permutation(52).tolist()
        
        hole_cards = (
            (deck[0], deck[1]),  # Player 0
            (deck[2], deck[3]),  # Player 1
        )
        
        # Preflop: SB posts small blind, BB posts big blind
        # Player 0 = SB (acts first preflop, then second postflop)
        # Player 1 = BB
        stacks = (
            self.starting_stack - self.small_blind,
            self.starting_stack - self.big_blind,
        )
        pot = self.small_blind + self.big_blind
        bets_this_round = (self.small_blind, self.big_blind)
        
        return NLHEState(
            hole_cards=hole_cards,
            board=tuple(deck[4:9]),  # Pre-deal the board for determinism
            stacks=stacks,
            pot=pot,
            bets_this_round=bets_this_round,
            round=PREFLOP,
            current_player=0,  # SB acts first preflop
            is_terminal=False,
            winner=None,
            history=((),),  # One empty tuple for preflop
        )
    
    def get_legal_actions(self, state: NLHEState) -> List[int]:
        """Get legal actions for current player."""
        if state.is_terminal:
            return []
        
        player = state.current_player
        stack = state.stacks[player]
        to_call = state.bets_this_round[1 - player] - state.bets_this_round[player]
        pot_after_call = state.pot + to_call
        
        actions = []
        
        # Fold is always legal (unless checking is free)
        if to_call > 0:
            actions.append(FOLD)
        
        # Check/Call
        if to_call <= stack:
            actions.append(CHECK_CALL)
        
        # Bet sizes (only if we have enough chips and haven't hit max raises)
        num_raises = sum(1 for a in state.history[state.round] if a >= BET_HALF_POT)
        if num_raises < self.max_raises_per_round and stack > to_call:
            remaining = stack - to_call
            
            # Half pot
            half_pot = pot_after_call // 2
            if half_pot > 0 and half_pot <= remaining:
                actions.append(BET_HALF_POT)
            
            # Pot
            pot_bet = pot_after_call
            if pot_bet > 0 and pot_bet <= remaining and pot_bet > half_pot:
                actions.append(BET_POT)
            
            # 2x pot
            two_pot = pot_after_call * 2
            if two_pot > 0 and two_pot <= remaining and two_pot > pot_bet:
                actions.append(BET_2X_POT)
            
            # All-in (if not already covered by other bets)
            if remaining > 0:
                actions.append(ALL_IN)
        
        return actions if actions else [CHECK_CALL]
    
    def apply_action(self, state: GameState, action: int) -> GameState:
        """Apply action (base class interface)."""
        nlhe_state = self._from_game_state(state)
        new_nlhe_state = self._apply_nlhe_action(nlhe_state, action)
        return self._to_game_state(new_nlhe_state)
    
    def _apply_nlhe_action(self, state: NLHEState, action: int) -> NLHEState:
        """Apply action and return new NLHE state."""
        player = state.current_player
        opponent = 1 - player
        stack = state.stacks[player]
        to_call = state.bets_this_round[opponent] - state.bets_this_round[player]
        pot_after_call = state.pot + to_call
        
        # Update history
        new_round_history = state.history[state.round] + (action,)
        new_history = state.history[:state.round] + (new_round_history,)
        if len(new_history) <= state.round:
            new_history = new_history + ((),) * (state.round + 1 - len(new_history))
        
        if action == FOLD:
            # Opponent wins
            return NLHEState(
                hole_cards=state.hole_cards,
                board=state.board,
                stacks=state.stacks,
                pot=state.pot,
                bets_this_round=state.bets_this_round,
                round=state.round,
                current_player=player,
                is_terminal=True,
                winner=opponent,
                history=new_history,
            )
        
        # Calculate bet amount
        if action == CHECK_CALL:
            bet_amount = min(to_call, stack)
        elif action == BET_HALF_POT:
            bet_amount = to_call + pot_after_call // 2
        elif action == BET_POT:
            bet_amount = to_call + pot_after_call
        elif action == BET_2X_POT:
            bet_amount = to_call + pot_after_call * 2
        elif action == ALL_IN:
            bet_amount = stack
        else:
            bet_amount = 0
        
        bet_amount = min(bet_amount, stack)
        
        # Update stacks and pot
        new_stacks = list(state.stacks)
        new_stacks[player] -= bet_amount
        new_stacks = tuple(new_stacks)
        
        new_bets = list(state.bets_this_round)
        new_bets[player] += bet_amount
        new_bets = tuple(new_bets)
        
        new_pot = state.pot + bet_amount
        
        # Check if round ends (both players acted and bets matched, or someone all-in)
        bets_matched = new_bets[0] == new_bets[1]
        both_acted = len(new_round_history) >= 2
        someone_allin = new_stacks[0] == 0 or new_stacks[1] == 0
        
        # Preflop special case: BB can still act after SB calls
        if state.round == PREFLOP and len(new_round_history) == 1:
            # SB just acted, BB still needs to act
            round_ends = False
        elif bets_matched and (both_acted or someone_allin):
            round_ends = True
        elif action >= BET_HALF_POT:
            # Raised, opponent needs to respond
            round_ends = False
        else:
            round_ends = bets_matched
        
        if round_ends:
            # Move to next round or showdown
            if state.round == RIVER or someone_allin:
                # Showdown
                return self._showdown(state, new_stacks, new_pot, new_history)
            else:
                # Next betting round
                new_round = state.round + 1
                # Pad history for new round
                while len(new_history) <= new_round:
                    new_history = new_history + ((),)
                
                return NLHEState(
                    hole_cards=state.hole_cards,
                    board=state.board,
                    stacks=new_stacks,
                    pot=new_pot,
                    bets_this_round=(0, 0),  # Reset for new round
                    round=new_round,
                    current_player=1,  # BB acts first postflop
                    is_terminal=False,
                    winner=None,
                    history=new_history,
                )
        else:
            # Same round, next player
            return NLHEState(
                hole_cards=state.hole_cards,
                board=state.board,
                stacks=new_stacks,
                pot=new_pot,
                bets_this_round=new_bets,
                round=state.round,
                current_player=opponent,
                is_terminal=False,
                winner=None,
                history=new_history,
            )
    
    def _showdown(self, state: NLHEState, stacks: Tuple[int, int], pot: int,
                  history: Tuple[Tuple[int, ...], ...]) -> NLHEState:
        """Determine winner at showdown."""
        # Get board cards for current round
        if state.round >= FLOP:
            board = list(state.board[:3])  # Flop
        else:
            board = []
        if state.round >= TURN:
            board.append(state.board[3])
        if state.round >= RIVER:
            board.append(state.board[4])
        
        # If not at river (all-in before), deal remaining cards
        full_board = list(state.board[:5])
        
        # Evaluate hands
        hand0 = list(state.hole_cards[0]) + full_board
        hand1 = list(state.hole_cards[1]) + full_board
        
        result = compare_hands(hand0, hand1)
        
        if result > 0:
            winner = 0
        elif result < 0:
            winner = 1
        else:
            winner = None  # Tie
        
        return NLHEState(
            hole_cards=state.hole_cards,
            board=state.board,
            stacks=stacks,
            pot=pot,
            bets_this_round=(0, 0),
            round=RIVER,
            current_player=state.current_player,
            is_terminal=True,
            winner=winner,
            history=history,
        )
    
    def get_payoff(self, state: NLHEState, player: int) -> float:
        """Get payoff for player (in big blinds)."""
        if not state.is_terminal:
            return 0.0
        
        if state.winner is None:
            # Tie: each player gets back half the pot
            return 0.0
        elif state.winner == player:
            # Won the pot
            return (state.pot / 2) / self.big_blind  # In BB
        else:
            # Lost
            return -(state.pot / 2) / self.big_blind
    
    def get_infoset(self, state: GameState, player: int) -> InfoSet:
        """Get information set for player (base class interface)."""
        nlhe_state = self._from_game_state(state)
        return self._get_nlhe_infoset(nlhe_state, player)
    
    def _get_nlhe_infoset(self, state: NLHEState, player: int) -> InfoSet:
        """Get information set for player from NLHE state."""
        # Private info: hole cards
        hole = state.hole_cards[player]
        
        # Public info: board, pot, stacks, history
        if state.round >= FLOP:
            visible_board = state.board[:3]
        else:
            visible_board = ()
        if state.round >= TURN:
            visible_board = state.board[:4]
        if state.round >= RIVER:
            visible_board = state.board[:5]
        
        return InfoSet(
            player=player,
            private_info=hole,
            public_state=(visible_board, state.pot, state.stacks, state.round),
            history=state.history,
        )
    
    def get_visible_board(self, state: NLHEState) -> List[int]:
        """Get currently visible board cards."""
        if state.round == PREFLOP:
            return []
        elif state.round == FLOP:
            return list(state.board[:3])
        elif state.round == TURN:
            return list(state.board[:4])
        else:
            return list(state.board[:5])


# ═══════════════════════════════════════════════════════════════════════════
# CARD ABSTRACTION (for tractable CFR)
# ═══════════════════════════════════════════════════════════════════════════

def expected_hand_strength(hole_cards: Tuple[int, int], board: List[int], 
                           num_samples: int = 1000, rng: Optional[np.random.Generator] = None) -> float:
    """
    Estimate hand strength via Monte Carlo sampling.
    
    Returns probability of winning against a random opponent hand.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Cards that are already used
    used = set(hole_cards) | set(board)
    remaining = [c for c in range(52) if c not in used]
    
    # We need 5 board cards total
    cards_to_deal_board = 5 - len(board)
    
    wins = 0
    ties = 0
    
    for _ in range(num_samples):
        # Sample remaining board + opponent hole cards
        sample = rng.choice(remaining, size=cards_to_deal_board + 2, replace=False)
        
        full_board = board + list(sample[:cards_to_deal_board])
        opp_hole = (sample[-2], sample[-1])
        
        my_hand = list(hole_cards) + full_board
        opp_hand = list(opp_hole) + full_board
        
        result = compare_hands(my_hand, opp_hand)
        if result > 0:
            wins += 1
        elif result == 0:
            ties += 1
    
    return (wins + 0.5 * ties) / num_samples


def bucket_hand_strength(ehs: float, num_buckets: int = 10) -> int:
    """Convert EHS to bucket index."""
    return min(int(ehs * num_buckets), num_buckets - 1)


if __name__ == "__main__":
    # Quick test
    game = NLHE(starting_stack=100)
    rng = np.random.default_rng(42)
    
    state = game.get_initial_state(rng)
    print(f"Initial state:")
    print(f"  P0 hole: {state.hole_cards[0]}")
    print(f"  P1 hole: {state.hole_cards[1]}")
    print(f"  Board: {state.board}")
    print(f"  Pot: {state.pot}")
    print(f"  Stacks: {state.stacks}")
    
    # Play a few actions
    while not state.is_terminal:
        actions = game.get_legal_actions(state)
        print(f"\nPlayer {state.current_player}, round {ROUND_NAMES[state.round]}")
        print(f"  Legal actions: {[ACTION_NAMES[a] for a in actions]}")
        
        # Random action
        action = rng.choice(actions)
        print(f"  Chose: {ACTION_NAMES[action]}")
        state = game.apply_action(state, action)
    
    print(f"\nGame over! Winner: {state.winner}, Pot: {state.pot}")
    print(f"P0 payoff: {game.get_payoff(state, 0):.2f} BB")
    print(f"P1 payoff: {game.get_payoff(state, 1):.2f} BB")
    
    # Test hand evaluation
    print("\n--- Hand Evaluation Test ---")
    # Royal flush
    cards = [card_to_int('A', 's'), card_to_int('K', 's'), card_to_int('Q', 's'),
             card_to_int('J', 's'), card_to_int('T', 's')]
    rank, kickers = evaluate_hand(cards)
    print(f"Royal flush: rank={rank} (STRAIGHT_FLUSH={STRAIGHT_FLUSH})")
    
    # Test EHS
    print("\n--- EHS Test ---")
    hole = (card_to_int('A', 's'), card_to_int('A', 'h'))
    ehs = expected_hand_strength(hole, [], num_samples=1000, rng=rng)
    print(f"AA preflop EHS: {ehs:.3f}")
    
    hole = (card_to_int('7', 's'), card_to_int('2', 'h'))
    ehs = expected_hand_strength(hole, [], num_samples=1000, rng=rng)
    print(f"72o preflop EHS: {ehs:.3f}")

