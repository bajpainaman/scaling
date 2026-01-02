#!/usr/bin/env python3
"""
🎯 Deep CFR Training on NLHE with Rust EHS
"""
import sys
sys.path.insert(0, '.')

import numpy as np
import torch
import time
from functools import lru_cache
from tqdm import tqdm

from src.games.nlhe import NLHE, PREFLOP, FLOP, TURN, RIVER
from src.training.deep_cfr import DeepCFRTrainer, DeepCFRConfig

# Import Rust module for fast EHS
import poker_ehs
print("🦀 Using Rust EHS")


# ═══════════════════════════════════════════════════════════════════════════
# FAST ENCODER (uses Rust for EHS)
# ═══════════════════════════════════════════════════════════════════════════

NUM_BUCKETS = 10
MAX_HISTORY = 20
NUM_ACTIONS = 6

# Encoding dim: bucket(10) + round(4) + pot_odds(1) + spr(1) + texture(5) + history(20*6)
ENCODING_DIM = NUM_BUCKETS + 4 + 1 + 1 + 5 + MAX_HISTORY * NUM_ACTIONS

# EHS samples by street
EHS_SAMPLES = {PREFLOP: 0, FLOP: 32, TURN: 64, RIVER: 0}


@lru_cache(maxsize=500_000)
def cached_bucket(h0: int, h1: int, board_tuple: tuple, street: int) -> int:
    """Cached EHS bucket using Rust."""
    board = list(board_tuple)
    samples = EHS_SAMPLES.get(street, 50)
    seed = hash((h0, h1, board_tuple, street)) & 0xFFFFFFFF
    return poker_ehs.fast_bucket([h0, h1], board, NUM_BUCKETS, samples, seed)


def encode_state(state, player: int) -> np.ndarray:
    """Fast state encoding using Rust EHS."""
    hole = state.hole_cards[player]
    
    # Get visible board
    if state.round == PREFLOP:
        board = []
    elif state.round == FLOP:
        board = list(state.board[:3])
    elif state.round == TURN:
        board = list(state.board[:4])
    else:
        board = list(state.board[:5])
    
    encoding = []
    
    # 1. Card bucket (cached Rust EHS)
    bucket = cached_bucket(int(hole[0]), int(hole[1]), tuple(board), state.round)
    bucket_onehot = np.zeros(NUM_BUCKETS)
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
    
    # 4. SPR (normalized)
    my_stack = state.stacks[player]
    spr = min(my_stack / (state.pot + 1e-8) / 10.0, 1.0)
    encoding.append(np.array([spr]))
    
    # 5. Board texture (simplified)
    texture = np.zeros(5)
    if board:
        ranks = [c // 4 for c in board]
        suits = [c % 4 for c in board]
        rank_counts = {}
        for r in ranks:
            rank_counts[r] = rank_counts.get(r, 0) + 1
        suit_counts = {}
        for s in suits:
            suit_counts[s] = suit_counts.get(s, 0) + 1
        
        texture[0] = float(any(c >= 2 for c in rank_counts.values()))
        texture[1] = float(sum(1 for c in rank_counts.values() if c >= 2) >= 2)
        texture[2] = float(any(c >= 3 for c in rank_counts.values()))
        texture[3] = float(any(c >= 3 for c in suit_counts.values()))
        texture[4] = float(max(ranks) - min(ranks) <= 4 if ranks else 0)
    encoding.append(texture)
    
    # 6. Action history
    history_enc = np.zeros(MAX_HISTORY * NUM_ACTIONS)
    idx = 0
    for round_actions in state.history:
        for action in round_actions:
            if idx < MAX_HISTORY and 0 <= action < NUM_ACTIONS:
                history_enc[idx * NUM_ACTIONS + action] = 1.0
                idx += 1
    encoding.append(history_enc)
    
    return np.concatenate(encoding)


def main():
    print(f"\n{'='*60}")
    print(f"🎯 DEEP CFR ON NLHE")
    print(f"{'='*60}")
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Encoding dim: {ENCODING_DIM}")

    # Setup game
    game = NLHE(starting_stack=100, max_raises_per_round=2)
    
    # Deep CFR config
    config = DeepCFRConfig(
        num_cfr_iterations=50,
        num_traversals_per_iter=200,
        train_steps_per_iter=100,
        batch_size=256,
        learning_rate=1e-4,
        advantage_memory_size=500_000,
        strategy_memory_size=500_000,
        epsilon=0.1,
        linear_cfr=True,
        device="mps" if torch.backends.mps.is_available() else "cpu",
    )
    
    print(f"\nConfig:")
    print(f"  CFR iterations: {config.num_cfr_iterations}")
    print(f"  Traversals/iter: {config.num_traversals_per_iter}")
    print(f"  Total traversals: {config.num_cfr_iterations * config.num_traversals_per_iter}")
    
    # Create trainer
    trainer = DeepCFRTrainer(
        game=game,
        encode_state=encode_state,
        input_dim=ENCODING_DIM,
        num_actions=NUM_ACTIONS,
        config=config,
    )
    
    print(f"\nAdvantage net params: {sum(p.numel() for p in trainer.advantage_nets[0].parameters()):,}")
    print(f"Strategy net params: {sum(p.numel() for p in trainer.strategy_net.parameters()):,}")
    
    # Train
    start = time.time()
    metrics = trainer.train(verbose=True)
    elapsed = time.time() - start
    
    print(f"\n{'='*60}")
    print(f"🏆 DEEP CFR COMPLETE")
    print(f"{'='*60}")
    print(f"Total time: {elapsed:.1f}s")
    print(f"Cache stats: {cached_bucket.cache_info()}")
    
    # Test strategy
    print("\nTesting strategy network...")
    rng = np.random.default_rng(123)
    test_states = [game.get_initial_state(rng) for _ in range(5)]
    
    print("\nSample strategies (fold, call, small, pot, big, allin):")
    for i, state in enumerate(test_states):
        enc = encode_state(state, 0)
        strategy = trainer.get_strategy(enc)
        print(f"  Hand {i+1}: [{', '.join(f'{p:.2f}' for p in strategy)}]")


if __name__ == "__main__":
    main()

