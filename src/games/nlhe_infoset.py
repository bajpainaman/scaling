"""
Proper NLHE Infoset Abstraction

The key insight: an infoset must capture everything that affects optimal play.

Components:
1. Private card strength (EHS bucket)
2. Street
3. Board texture (pair/trips/flush-draw/straight-draw)
4. Pot size bucket (log scale)
5. SPR bucket (stack-to-pot ratio)
6. Action pattern (compressed history)

This gives us: 10 × 4 × 16 × 8 × 6 × ~100 = ~3M theoretical infosets
In practice we'll see 10k-100k+ unique ones.
"""

import numpy as np
from typing import Tuple
from functools import lru_cache


# ═══════════════════════════════════════════════════════════════════════════
# BUCKET FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def board_texture_bucket(board: Tuple[int, ...]) -> int:
    """
    Encode board texture into a 4-bit bucket (0-15).
    
    Bits:
    - bit 0: paired board
    - bit 1: two-pair or better texture
    - bit 2: flush possible (3+ same suit)
    - bit 3: straight possible (connected)
    """
    if not board:
        return 0
    
    ranks = [c // 4 for c in board]
    suits = [c % 4 for c in board]
    
    # Count ranks
    rank_counts = {}
    for r in ranks:
        rank_counts[r] = rank_counts.get(r, 0) + 1
    
    # Count suits
    suit_counts = {}
    for s in suits:
        suit_counts[s] = suit_counts.get(s, 0) + 1
    
    bucket = 0
    
    # Bit 0: paired
    if any(c >= 2 for c in rank_counts.values()):
        bucket |= 1
    
    # Bit 1: two-pair+ texture
    if sum(1 for c in rank_counts.values() if c >= 2) >= 2 or any(c >= 3 for c in rank_counts.values()):
        bucket |= 2
    
    # Bit 2: flush possible
    if any(c >= 3 for c in suit_counts.values()):
        bucket |= 4
    
    # Bit 3: straight possible (3+ cards within 5 ranks)
    unique_ranks = sorted(set(ranks))
    if len(unique_ranks) >= 3:
        for i in range(len(unique_ranks) - 2):
            if unique_ranks[i + 2] - unique_ranks[i] <= 4:
                bucket |= 8
                break
    
    return bucket


def pot_bucket(pot: int, starting_stack: int = 100) -> int:
    """
    Log-scale pot bucket (0-7).
    
    Buckets: [0-3], [4-10], [11-25], [26-50], [51-100], [101-200], [201-400], [400+]
    """
    if pot <= 3:
        return 0
    elif pot <= 10:
        return 1
    elif pot <= 25:
        return 2
    elif pot <= 50:
        return 3
    elif pot <= 100:
        return 4
    elif pot <= 200:
        return 5
    elif pot <= 400:
        return 6
    else:
        return 7


def spr_bucket(stack: int, pot: int) -> int:
    """
    Stack-to-Pot Ratio bucket (0-5).
    
    SPR < 1: committed
    SPR 1-2: short
    SPR 2-4: medium
    SPR 4-8: deep
    SPR 8-16: very deep
    SPR 16+: ultra deep
    """
    if pot <= 0:
        return 5
    
    spr = stack / pot
    
    if spr < 1:
        return 0
    elif spr < 2:
        return 1
    elif spr < 4:
        return 2
    elif spr < 8:
        return 3
    elif spr < 16:
        return 4
    else:
        return 5


def action_pattern_bucket(history: Tuple[Tuple[int, ...], ...], max_patterns: int = 128) -> int:
    """
    Compress action history into a pattern bucket.
    
    We encode: (num_bets_preflop, num_bets_flop, num_bets_turn, num_bets_river, last_action)
    This captures the aggression pattern without memorizing exact sequences.
    """
    # Count bets/raises per street
    bets_per_street = [0, 0, 0, 0]
    last_action = 0
    
    for street_idx, street_actions in enumerate(history):
        if street_idx >= 4:
            break
        for action in street_actions:
            if action >= 2:  # Bet/raise actions
                bets_per_street[street_idx] = min(3, bets_per_street[street_idx] + 1)
            last_action = action
    
    # Encode: bets_preflop(2 bits) + bets_flop(2 bits) + bets_turn(2 bits) + last_action(3 bits)
    # = 9 bits = 512 max, but we cap at max_patterns
    pattern = (
        (min(3, bets_per_street[0]) << 7) |
        (min(3, bets_per_street[1]) << 5) |
        (min(3, bets_per_street[2]) << 3) |
        (min(5, last_action))
    )
    
    return pattern % max_patterns


def to_call_bucket(to_call: int, pot: int) -> int:
    """
    To-call relative to pot bucket (0-5).
    
    0: nothing to call (checking)
    1: < 25% pot
    2: 25-50% pot
    3: 50-100% pot
    4: 100-200% pot
    5: > 200% pot (overbet)
    """
    if to_call <= 0:
        return 0
    
    if pot <= 0:
        return 5
    
    ratio = to_call / pot
    
    if ratio < 0.25:
        return 1
    elif ratio < 0.5:
        return 2
    elif ratio < 1.0:
        return 3
    elif ratio < 2.0:
        return 4
    else:
        return 5


# ═══════════════════════════════════════════════════════════════════════════
# FULL INFOSET KEY
# ═══════════════════════════════════════════════════════════════════════════

def make_infoset_key(
    player: int,
    card_bucket: int,
    street: int,
    board: Tuple[int, ...],
    pot: int,
    stack: int,
    to_call: int,
    history: Tuple[Tuple[int, ...], ...],
) -> str:
    """
    Build a proper infoset key with all relevant dimensions.
    
    Format: p{player}|c{card}|s{street}|t{texture}|pot{pot}|spr{spr}|tc{to_call}|h{history}
    
    Theoretical max: 2 × 10 × 4 × 16 × 8 × 6 × 6 × 128 = 11,796,480
    Practical: 10k-100k+ unique keys
    """
    texture = board_texture_bucket(board)
    pot_b = pot_bucket(pot)
    spr_b = spr_bucket(stack, pot)
    tc_b = to_call_bucket(to_call, pot)
    hist_b = action_pattern_bucket(history)
    
    return f"p{player}|c{card_bucket}|s{street}|t{texture}|pot{pot_b}|spr{spr_b}|tc{tc_b}|h{hist_b}"


def make_infoset_key_from_state(state, player: int, card_bucket: int) -> str:
    """Build infoset key from NLHEState."""
    # Get visible board
    if state.round == 0:  # PREFLOP
        board = tuple()
    elif state.round == 1:  # FLOP
        board = tuple(state.board[:3])
    elif state.round == 2:  # TURN
        board = tuple(state.board[:4])
    else:  # RIVER
        board = tuple(state.board[:5])
    
    to_call = state.bets_this_round[1 - player] - state.bets_this_round[player]
    
    return make_infoset_key(
        player=player,
        card_bucket=card_bucket,
        street=state.round,
        board=board,
        pot=state.pot,
        stack=state.stacks[player],
        to_call=max(0, to_call),
        history=state.history,
    )


# ═══════════════════════════════════════════════════════════════════════════
# TESTING
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')
    
    from src.games.nlhe import NLHE
    import numpy as np
    
    game = NLHE(starting_stack=100, max_raises_per_round=2)
    rng = np.random.default_rng(42)
    
    # Play many games, collect unique infosets
    unique_keys = set()
    
    print("Testing infoset key diversity...")
    
    for game_num in range(1000):
        state = game.get_initial_state(rng)
        
        while not state.is_terminal:
            player = state.current_player
            legal = game.get_legal_actions(state)
            
            # Fake bucket (random for testing)
            bucket = rng.integers(0, 10)
            key = make_infoset_key_from_state(state, player, bucket)
            unique_keys.add(key)
            
            # Random action
            action = rng.choice(legal)
            state = game._apply_nlhe_action(state, action)
    
    print(f"\n1,000 games → {len(unique_keys)} unique infosets")
    
    # Show some samples
    print("\nSample keys:")
    for key in list(unique_keys)[:10]:
        print(f"  {key}")

