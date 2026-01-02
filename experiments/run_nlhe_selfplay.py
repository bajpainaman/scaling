#!/usr/bin/env python3
"""
🦀 BLAZING FAST NLHE Self-Play with Rust EHS
"""
import sys
sys.path.insert(0, '.')

import numpy as np
import torch
from collections import defaultdict
import time
from functools import lru_cache
from tqdm import tqdm

from src.games.nlhe import NLHE, PREFLOP, FLOP, TURN, RIVER
from src.games.nlhe_infoset import make_infoset_key_from_state
from src.models import MLP

# Import Rust module for fast EHS
import poker_ehs
print("🦀 Using Rust EHS (245x faster than Python)")


# ═══════════════════════════════════════════════════════════════════════════
# FAST ENCODER (uses Rust for EHS)
# ═══════════════════════════════════════════════════════════════════════════

NUM_BUCKETS = 10
MAX_HISTORY = 20
NUM_ACTIONS = 6

# Encoding dim: bucket(10) + round(4) + pot_odds(1) + spr(1) + texture(5) + history(20*6)
ENCODING_DIM = NUM_BUCKETS + 4 + 1 + 1 + 5 + MAX_HISTORY * NUM_ACTIONS

# EHS samples by street (less for earlier streets)
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
        
        texture[0] = float(any(c >= 2 for c in rank_counts.values()))  # pair
        texture[1] = float(sum(1 for c in rank_counts.values() if c >= 2) >= 2)  # two pair
        texture[2] = float(any(c >= 3 for c in rank_counts.values()))  # trips
        texture[3] = float(any(c >= 3 for c in suit_counts.values()))  # flush draw
        texture[4] = float(max(ranks) - min(ranks) <= 4 if ranks else 0)  # connected
    encoding.append(texture)
    
    # 6. Action history (simplified)
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
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Encoding dim: {ENCODING_DIM}")

    # Setup
    game = NLHE(starting_stack=100, max_raises_per_round=2)
    
    model = MLP.from_name("m", input_dim=ENCODING_DIM, num_actions=NUM_ACTIONS).to(device)
    print(f"Model params: {model.num_parameters():,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    rng = np.random.default_rng(42)

    # CFR tables
    regret_sum = defaultdict(lambda: np.zeros(NUM_ACTIONS, dtype=np.float64))
    strategy_sum = defaultdict(lambda: np.zeros(NUM_ACTIONS, dtype=np.float64))

    # Replay buffer
    buffer = []
    buffer_max = 100000

    def get_strategy(cfv, legal_actions):
        """Get strategy from CFV via regret matching."""
        regrets = cfv - np.mean(cfv[legal_actions])
        positive = np.maximum(regrets, 0)
        strategy = np.zeros(NUM_ACTIONS)
        legal_positive = positive[legal_actions]
        total = legal_positive.sum()
        if total > 0:
            strategy[legal_actions] = legal_positive / total
        else:
            strategy[legal_actions] = 1.0 / len(legal_actions)
        return strategy

    def predict_cfv(enc):
        """Get CFV from model."""
        x = torch.tensor(enc, dtype=torch.float32).unsqueeze(0).to(device)
        model.eval()
        with torch.no_grad():
            return model(x).squeeze(0).cpu().numpy()

    def external_cfr(state, traverser, epsilon=0.1):
        """External sampling CFR traversal."""
        if state.is_terminal:
            return game.get_payoff(state, traverser)
        
        player = state.current_player
        legal = game.get_legal_actions(state)
        
        # Encode state
        enc = encode_state(state, player)
        
        # Get strategy
        cfv = predict_cfv(enc)
        strategy = get_strategy(cfv, legal)
        
        # Epsilon-greedy exploration
        if rng.random() < epsilon:
            strategy[legal] = 1.0 / len(legal)
        
        # Proper infoset key with all dimensions
        bucket = np.argmax(enc[:NUM_BUCKETS])
        key = make_infoset_key_from_state(state, player, bucket)
        
        if player == traverser:
            # Traverse all actions
            action_values = np.zeros(NUM_ACTIONS)
            for action in legal:
                next_state = game._apply_nlhe_action(state, action)
                action_values[action] = external_cfr(next_state, traverser, epsilon)
            
            ev = np.dot(strategy, action_values)
            regrets = action_values - ev
            
            # CFR+ update
            regret_sum[key] += regrets
            regret_sum[key] = np.maximum(0, regret_sum[key])
            strategy_sum[key] += strategy
            
            # Add to buffer (normalized regrets as target)
            target = regret_sum[key].copy()
            max_abs = np.max(np.abs(target)) + 1e-8
            target = target / max_abs
            
            if len(buffer) < buffer_max:
                buffer.append((enc.copy(), target))
            else:
                idx = rng.integers(len(buffer))
                buffer[idx] = (enc.copy(), target)
            
            return ev
        else:
            # Sample opponent action
            probs = strategy[legal]
            probs = probs / (probs.sum() + 1e-8)
            action = rng.choice(legal, p=probs)
            next_state = game._apply_nlhe_action(state, action)
            return external_cfr(next_state, traverser, epsilon)

    def train_step(batch_size=128):
        """Train model on buffer."""
        if len(buffer) < batch_size:
            return 0.0
        
        indices = rng.choice(len(buffer), size=batch_size, replace=False)
        X = torch.tensor(np.array([buffer[i][0] for i in indices]), dtype=torch.float32).to(device)
        Y = torch.tensor(np.array([buffer[i][1] for i in indices]), dtype=torch.float32).to(device)
        
        model.train()
        optimizer.zero_grad()
        pred = model(X)
        loss = ((pred - Y) ** 2).mean()
        loss.backward()
        optimizer.step()
        
        return loss.item()

    # Training loop - 20k games!
    num_epochs = 5
    games_per_epoch = 4000
    train_steps = 200

    print(f"\n{'='*60}")
    print(f"🦀 NLHE SELF-PLAY (20k games)")
    print(f"{'='*60}")

    total_games = 0
    for epoch in range(num_epochs):
        start = time.time()
        
        # Collect data with progress bar
        pbar = tqdm(range(games_per_epoch), desc=f"Epoch {epoch+1} games", 
                    ncols=80, leave=True)
        for g in pbar:
            state = game.get_initial_state(rng)
            for player in range(2):
                external_cfr(state, player)
            
            if (g+1) % 500 == 0:
                cache_info = cached_bucket.cache_info()
                hit_rate = cache_info.hits / (cache_info.hits + cache_info.misses + 1) * 100
                pbar.set_postfix({"buf": len(buffer), "cache": f"{hit_rate:.0f}%"})
        
        total_games += games_per_epoch
        elapsed = time.time() - start
        
        # Train with progress bar
        losses = []
        for _ in tqdm(range(train_steps), desc="Training", ncols=80, leave=False):
            loss = train_step()
            losses.append(loss)
        
        avg_loss = np.mean(losses) if losses else 0.0
        ms_per_game = elapsed / games_per_epoch * 1000
        games_per_sec = games_per_epoch / elapsed
        
        print(f"✅ Epoch {epoch+1}: {total_games} games | buffer={len(buffer)} | infosets={len(regret_sum)}")
        print(f"   Loss: {avg_loss:.4f} | {games_per_sec:.1f} games/s ({ms_per_game:.1f}ms/game)\n")

    print(f"\n{'='*60}")
    print(f"🏆 NLHE SELF-PLAY COMPLETE")
    print(f"{'='*60}")
    print(f"Total games: {total_games}")
    print(f"Total infosets: {len(regret_sum)}")
    print(f"Buffer size: {len(buffer)}")
    print(f"Cache stats: {cached_bucket.cache_info()}")
    
    # Show sample strategies
    print("\nSample learned strategies (fold, call, bet_small, bet_pot, bet_big, allin):")
    for i, key in enumerate(list(strategy_sum.keys())[:8]):
        strat = strategy_sum[key]
        total = strat.sum()
        if total > 0:
            strat = strat / total
            print(f"  {key}: [{strat[0]:.2f}, {strat[1]:.2f}, {strat[2]:.2f}, {strat[3]:.2f}, {strat[4]:.2f}, {strat[5]:.2f}]")


if __name__ == "__main__":
    main()
