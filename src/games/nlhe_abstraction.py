"""
NLHE Abstraction and Encoding for Neural Networks

Card Abstraction:
- Uses Expected Hand Strength (EHS) bucketed into N buckets
- Preflop EHS is pre-computed for speed
- Reduces 1326 preflop hands to ~10-50 buckets

Infoset Encoding:
- Card bucket one-hot
- Board texture features
- Pot odds, SPR
- Action history
"""

import numpy as np
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass
from functools import lru_cache

from .nlhe import (
    NLHE, NLHEState, 
    expected_hand_strength, bucket_hand_strength,
    get_rank, get_suit, PREFLOP, FLOP, TURN, RIVER,
    compare_hands
)
from .base import InfoSet, GameState


# ═══════════════════════════════════════════════════════════════════════════
# PRE-COMPUTED PREFLOP EHS (fast lookup)
# ═══════════════════════════════════════════════════════════════════════════

# Preflop hand strength by hand type (suited, offsuit, pair)
# Format: (rank1, rank2, suited) -> approximate EHS
# These are standard preflop equity values
PREFLOP_EHS_TABLE: Dict[Tuple[int, int, bool], float] = {}

def _init_preflop_table():
    """Initialize preflop EHS lookup table."""
    # Pairs (rank, rank, False) - approximate equities
    pair_ehs = {
        12: 0.852,  # AA
        11: 0.824,  # KK
        10: 0.799,  # QQ
        9: 0.773,   # JJ
        8: 0.750,   # TT
        7: 0.723,   # 99
        6: 0.696,   # 88
        5: 0.669,   # 77
        4: 0.642,   # 66
        3: 0.615,   # 55
        2: 0.588,   # 44
        1: 0.561,   # 33
        0: 0.534,   # 22
    }
    
    for rank, ehs in pair_ehs.items():
        PREFLOP_EHS_TABLE[(rank, rank, False)] = ehs
    
    # Broadway hands (suited adds ~3%, connected adds ~2%)
    broadway = [12, 11, 10, 9, 8]  # A, K, Q, J, T
    
    for i, r1 in enumerate(broadway):
        for j, r2 in enumerate(broadway):
            if i < j:  # r1 > r2
                gap = r1 - r2
                base_ehs = 0.55 + (r1 + r2) * 0.01 - gap * 0.015
                PREFLOP_EHS_TABLE[(r1, r2, True)] = min(0.68, base_ehs + 0.03)  # Suited
                PREFLOP_EHS_TABLE[(r1, r2, False)] = min(0.65, base_ehs)  # Offsuit
    
    # Other hands - approximate formula
    for r1 in range(13):
        for r2 in range(r1):  # r1 > r2
            if (r1, r2, True) not in PREFLOP_EHS_TABLE:
                gap = r1 - r2
                base_ehs = 0.35 + (r1 + r2) * 0.008 - gap * 0.01
                PREFLOP_EHS_TABLE[(r1, r2, True)] = max(0.30, min(0.60, base_ehs + 0.025))
                PREFLOP_EHS_TABLE[(r1, r2, False)] = max(0.28, min(0.58, base_ehs))

_init_preflop_table()


def fast_preflop_ehs(card1: int, card2: int) -> float:
    """Fast preflop EHS lookup."""
    r1, s1 = card1 // 4, card1 % 4
    r2, s2 = card2 // 4, card2 % 4
    
    suited = (s1 == s2)
    high, low = max(r1, r2), min(r1, r2)
    
    if high == low:  # Pair
        return PREFLOP_EHS_TABLE.get((high, high, False), 0.5)
    else:
        return PREFLOP_EHS_TABLE.get((high, low, suited), 0.45)


def fast_postflop_ehs(hole: Tuple[int, int], board: List[int], num_samples: int = 50) -> float:
    """
    Fast postflop EHS with reduced samples.
    Uses vectorized operations where possible.
    """
    rng = np.random.default_rng()
    
    used = set(hole) | set(board)
    remaining = np.array([c for c in range(52) if c not in used])
    
    cards_to_deal = 5 - len(board)
    
    wins = 0
    ties = 0
    
    for _ in range(num_samples):
        sample = rng.choice(remaining, size=cards_to_deal + 2, replace=False)
        full_board = board + list(sample[:cards_to_deal])
        opp_hole = (sample[-2], sample[-1])
        
        my_hand = list(hole) + full_board
        opp_hand = list(opp_hole) + full_board
        
        result = compare_hands(my_hand, opp_hand)
        if result > 0:
            wins += 1
        elif result == 0:
            ties += 1
    
    return (wins + 0.5 * ties) / num_samples


@dataclass
class NLHEAbstractionConfig:
    """Configuration for NLHE abstraction."""
    num_card_buckets: int = 10  # EHS buckets
    ehs_samples: int = 50  # Monte Carlo samples for postflop EHS (reduced from 200)
    max_history_actions: int = 20  # Max actions to encode
    num_action_types: int = 6  # fold, call, 0.5x, 1x, 2x, all-in


class NLHEEncoder:
    """
    Encodes NLHE infosets into fixed-size vectors for neural networks.
    
    Encoding layout:
    - Card bucket one-hot: [num_buckets]
    - Round one-hot: [4]
    - Pot odds: [1]
    - Stack-to-pot ratio: [1]
    - Board texture: [5] (pair, two-pair, trips, flush-possible, straight-possible)
    - Action history: [max_history * num_actions]
    
    Total: num_buckets + 4 + 1 + 1 + 5 + max_history * num_actions
    """
    
    def __init__(self, config: NLHEAbstractionConfig = None):
        self.config = config or NLHEAbstractionConfig()
        self._rng = np.random.default_rng()
        
        # Pre-compute encoding dimension
        self.encoding_dim = (
            self.config.num_card_buckets +  # Card bucket
            4 +  # Round one-hot
            1 +  # Pot odds
            1 +  # SPR
            5 +  # Board texture
            self.config.max_history_actions * self.config.num_action_types  # History
        )
    
    def encode_state(self, state: NLHEState, player: int) -> np.ndarray:
        """Encode NLHE state for a player."""
        hole_cards = state.hole_cards[player]
        board = self._get_visible_board(state)
        
        encoding = []
        
        # 1. Card bucket (EHS-based) - FAST version
        if state.round == PREFLOP or len(board) == 0:
            # Use preflop lookup table (instant)
            ehs = fast_preflop_ehs(hole_cards[0], hole_cards[1])
        else:
            # Use reduced samples for postflop
            ehs = fast_postflop_ehs(hole_cards, board, self.config.ehs_samples)
        
        bucket = bucket_hand_strength(ehs, self.config.num_card_buckets)
        bucket_onehot = np.zeros(self.config.num_card_buckets)
        bucket_onehot[bucket] = 1.0
        encoding.append(bucket_onehot)
        
        # 2. Round one-hot
        round_onehot = np.zeros(4)
        round_onehot[state.round] = 1.0
        encoding.append(round_onehot)
        
        # 3. Pot odds
        to_call = state.bets_this_round[1 - player] - state.bets_this_round[player]
        pot_odds = to_call / (state.pot + to_call + 1e-8)
        encoding.append(np.array([pot_odds]))
        
        # 4. Stack-to-pot ratio (SPR)
        my_stack = state.stacks[player]
        spr = my_stack / (state.pot + 1e-8)
        spr_normalized = min(spr / 10.0, 1.0)  # Normalize to [0, 1]
        encoding.append(np.array([spr_normalized]))
        
        # 5. Board texture
        texture = self._encode_board_texture(board)
        encoding.append(texture)
        
        # 6. Action history
        history_enc = self._encode_history(state.history)
        encoding.append(history_enc)
        
        return np.concatenate(encoding)
    
    def _get_visible_board(self, state: NLHEState) -> List[int]:
        """Get visible board cards based on current round."""
        if state.round == PREFLOP:
            return []
        elif state.round == FLOP:
            return list(state.board[:3])
        elif state.round == TURN:
            return list(state.board[:4])
        else:  # RIVER
            return list(state.board[:5])
    
    def _encode_board_texture(self, board: List[int]) -> np.ndarray:
        """Encode board texture features."""
        if not board:
            return np.zeros(5)
        
        ranks = [get_rank(c) for c in board]
        suits = [get_suit(c) for c in board]
        
        # Pair on board
        rank_counts = {}
        for r in ranks:
            rank_counts[r] = rank_counts.get(r, 0) + 1
        
        has_pair = any(c >= 2 for c in rank_counts.values())
        has_two_pair = sum(1 for c in rank_counts.values() if c >= 2) >= 2
        has_trips = any(c >= 3 for c in rank_counts.values())
        
        # Flush possible (3+ of same suit)
        suit_counts = {}
        for s in suits:
            suit_counts[s] = suit_counts.get(s, 0) + 1
        flush_possible = any(c >= 3 for c in suit_counts.values())
        
        # Straight possible (3+ connected cards)
        unique_ranks = sorted(set(ranks))
        straight_possible = False
        for i in range(len(unique_ranks) - 2):
            if unique_ranks[i+2] - unique_ranks[i] <= 4:
                straight_possible = True
                break
        
        return np.array([
            float(has_pair),
            float(has_two_pair),
            float(has_trips),
            float(flush_possible),
            float(straight_possible),
        ])
    
    def _encode_history(self, history: Tuple[Tuple[int, ...], ...]) -> np.ndarray:
        """Encode action history."""
        encoding = np.zeros(self.config.max_history_actions * self.config.num_action_types)
        
        idx = 0
        for round_actions in history:
            for action in round_actions:
                if idx >= self.config.max_history_actions:
                    break
                if 0 <= action < self.config.num_action_types:
                    encoding[idx * self.config.num_action_types + action] = 1.0
                idx += 1
        
        return encoding
    
    def encode_infoset(self, infoset: InfoSet, game: NLHE) -> np.ndarray:
        """Encode an InfoSet (for compatibility with self-play trainer)."""
        # Reconstruct state from infoset (simplified)
        hole_cards = infoset.private_info
        public_state = infoset.public_state
        
        # public_state = (visible_board, pot, stacks, round)
        board, pot, stacks, round_num = public_state
        
        # Create minimal state for encoding
        state = NLHEState(
            hole_cards=(hole_cards, (0, 0)),  # Opponent cards unknown
            board=board if board else tuple(range(5)),  # Placeholder
            stacks=stacks,
            pot=pot,
            bets_this_round=(0, 0),
            round=round_num,
            current_player=infoset.player,
            is_terminal=False,
            winner=None,
            history=infoset.history,
        )
        
        return self.encode_state(state, infoset.player)


def create_nlhe_encoder(num_buckets: int = 10) -> NLHEEncoder:
    """Create an NLHE encoder with specified number of buckets."""
    config = NLHEAbstractionConfig(num_card_buckets=num_buckets)
    return NLHEEncoder(config)


if __name__ == "__main__":
    # Test encoding
    import numpy as np
    
    game = NLHE(starting_stack=100)
    encoder = NLHEEncoder()
    
    rng = np.random.default_rng(42)
    state = game.get_initial_state(rng)
    
    print(f"Encoding dimension: {encoder.encoding_dim}")
    
    enc0 = encoder.encode_state(state, player=0)
    enc1 = encoder.encode_state(state, player=1)
    
    print(f"\nPlayer 0 encoding shape: {enc0.shape}")
    print(f"Player 1 encoding shape: {enc1.shape}")
    
    # Check card buckets are different (different hands)
    bucket0 = np.argmax(enc0[:10])
    bucket1 = np.argmax(enc1[:10])
    print(f"\nPlayer 0 card bucket: {bucket0}")
    print(f"Player 1 card bucket: {bucket1}")

