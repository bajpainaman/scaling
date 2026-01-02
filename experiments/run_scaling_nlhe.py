#!/usr/bin/env python3
"""
NLHE Scaling Laws: MLP vs Transformer at Real Scale

This is THE experiment: measure how CFV prediction loss scales with:
- D (data): 100k → 1M → 5M samples
- N (model): tiny → small → medium → large

Run on M4 Pro for quick iteration, then H100 for full sweep.
"""

import sys
sys.path.insert(0, '.')

import torch
import torch.nn as nn
import numpy as np
import json
import time
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple
from tqdm import tqdm

import poker_ehs

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════

DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
ENCODING_DIM = 141
NUM_ACTIONS = 6

# Model sizes (parameter counts)
MODEL_CONFIGS = {
    "tiny":   {"hidden": 64,  "layers": 2},   # ~10k params
    "small":  {"hidden": 128, "layers": 3},   # ~50k params
    "medium": {"hidden": 256, "layers": 4},   # ~250k params
    "large":  {"hidden": 512, "layers": 4},   # ~1M params
}

# Data sizes (start small for M4, scale up on H100)
DATA_SIZES = [50_000, 200_000, 500_000]  # samples

@dataclass
class ExperimentResult:
    model_name: str
    model_params: int
    data_size: int
    train_loss: float
    val_loss: float
    train_time: float

# ═══════════════════════════════════════════════════════════════════════════
# MODELS
# ═══════════════════════════════════════════════════════════════════════════

class MLP(nn.Module):
    def __init__(self, hidden: int, layers: int):
        super().__init__()
        dims = [ENCODING_DIM] + [hidden] * layers + [NUM_ACTIONS]
        self.layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                self.layers.append(nn.ReLU())
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class GroupedTransformer(nn.Module):
    """
    Transformer with semantic token groups (24 tokens, not 141).
    
    Token structure: [CLS] [CARD] [ROUND] [STATE] [ACT_1..20]
    Attention cost: O(24²) = 576 vs O(141²) = 19,881 → 34× faster!
    """
    def __init__(self, hidden: int, layers: int, num_heads: int = 4):
        super().__init__()
        self.hidden = hidden
        self.num_actions = NUM_ACTIONS
        self.max_history = 20
        
        # Token embeddings
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden) * 0.02)
        self.bucket_embed = nn.Embedding(10, hidden)
        self.round_embed = nn.Embedding(4, hidden)
        self.action_embed = nn.Embedding(NUM_ACTIONS + 1, hidden)  # +1 for pad
        self.state_proj = nn.Linear(6, hidden)
        self.pos_embed = nn.Embedding(24, hidden)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=num_heads, dim_feedforward=hidden*4,
            dropout=0.0, activation='gelu', batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, NUM_ACTIONS))
    
    def forward(self, x):
        B, device = x.shape[0], x.device
        
        # Parse encoding
        bucket_idx = x[:, :10].argmax(dim=1)
        round_idx = x[:, 10:14].argmax(dim=1)
        state_feats = torch.cat([x[:, 14:16], x[:, 16:20]], dim=1)  # pot_odds, spr, texture
        
        # History
        history_onehot = x[:, 21:].reshape(B, self.max_history, self.num_actions)
        history_mask = history_onehot.sum(dim=2) > 0
        history_idx = history_onehot.argmax(dim=2)
        history_idx = torch.where(history_mask, history_idx, 
                                   torch.full_like(history_idx, self.num_actions))
        
        # Build tokens: [CLS, CARD, ROUND, STATE, ACT_1..20]
        tokens = torch.cat([
            self.cls_token.expand(B, -1, -1),
            self.bucket_embed(bucket_idx).unsqueeze(1),
            self.round_embed(round_idx).unsqueeze(1),
            self.state_proj(state_feats).unsqueeze(1),
            self.action_embed(history_idx),
        ], dim=1)  # (B, 24, H)
        
        tokens = tokens + self.pos_embed(torch.arange(24, device=device))
        tokens = self.encoder(tokens)
        return self.head(tokens[:, 0])  # CLS token

# ═══════════════════════════════════════════════════════════════════════════
# DATA GENERATION
# ═══════════════════════════════════════════════════════════════════════════

def generate_data(num_samples: int, seed: int = 42) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate training data from Rust CFR engine."""
    # Estimate games needed (avg ~70 samples/game)
    num_games = max(1000, num_samples // 50)
    
    print(f"  Generating {num_games:,} games for ~{num_samples:,} samples...")
    start = time.time()
    
    enc_flat, adv_flat, players, keys, n, stats = poker_ehs.run_cfr_games(
        num_games=num_games,
        starting_stack=100,
        max_raises=2,
        epsilon=0.1,
        seed=seed,
    )
    
    elapsed = time.time() - start
    print(f"  Generated {n:,} samples in {elapsed:.1f}s ({n/elapsed:,.0f} samples/sec)")
    
    # Convert to tensors
    X = torch.tensor(enc_flat, dtype=torch.float32).reshape(n, ENCODING_DIM)
    y = torch.tensor(adv_flat, dtype=torch.float32).reshape(n, NUM_ACTIONS)
    
    # NORMALIZE TARGETS (critical for stable training!)
    # Clip extreme values and scale to reasonable range
    y = torch.clamp(y, -100, 100)  # Clip outliers
    y_std = y.std() + 1e-8
    y = y / y_std  # Normalize to unit variance
    
    print(f"  Target stats: mean={y.mean():.3f}, std={y.std():.3f}, max={y.abs().max():.1f}")
    
    # Subsample if we got too many
    if n > num_samples:
        idx = torch.randperm(n)[:num_samples]
        X, y = X[idx], y[idx]
    
    return X, y

# ═══════════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════════

def train_model(model: nn.Module, X: torch.Tensor, y: torch.Tensor, 
                epochs: int = 10, batch_size: int = 4096, lr: float = 1e-3,
                model_name: str = "model") -> Tuple[float, float, float]:
    """Train model and return (train_loss, val_loss, time)."""
    model = model.to(DEVICE)
    
    # Split
    n = len(X)
    perm = torch.randperm(n)
    train_n = int(0.9 * n)
    train_idx, val_idx = perm[:train_n], perm[train_n:]
    
    X_train, y_train = X[train_idx].to(DEVICE), y[train_idx].to(DEVICE)
    X_val, y_val = X[val_idx].to(DEVICE), y[val_idx].to(DEVICE)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    start = time.time()
    
    pbar = tqdm(range(epochs), desc=f"{model_name}", leave=False)
    for epoch in pbar:
        model.train()
        perm_train = torch.randperm(train_n)
        total_loss = 0.0
        n_batches = 0
        
        for i in range(0, train_n, batch_size):
            idx = perm_train[i:i+batch_size]
            xb, yb = X_train[idx], y_train[idx]
            
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        train_loss = total_loss / n_batches
        pbar.set_postfix({"loss": f"{train_loss:.4f}"})
    
    # Final validation
    model.eval()
    with torch.no_grad():
        val_pred = model(X_val)
        val_loss = criterion(val_pred, y_val).item()
    
    elapsed = time.time() - start
    
    return train_loss, val_loss, elapsed

# ═══════════════════════════════════════════════════════════════════════════
# MAIN EXPERIMENT
# ═══════════════════════════════════════════════════════════════════════════

def run_experiment():
    log.info("="*60)
    log.info("🧪 NLHE SCALING LAWS: MLP vs Transformer (Grouped)")
    log.info(f"   Device: {DEVICE}")
    log.info("="*60)
    
    results: List[ExperimentResult] = []
    
    for data_size in tqdm(DATA_SIZES, desc="Data sizes"):
        log.info(f"\n📊 Data size: {data_size:,}")
        
        # Generate data once per size
        X, y = generate_data(data_size)
        
        for model_name, config in tqdm(list(MODEL_CONFIGS.items()), desc="Models", leave=False):
            # MLP
            mlp = MLP(**config)
            mlp_params = sum(p.numel() for p in mlp.parameters())
            
            train_loss, val_loss, elapsed = train_model(mlp, X, y, model_name=f"MLP-{model_name}")
            results.append(ExperimentResult(
                f"MLP-{model_name}", mlp_params, data_size, train_loss, val_loss, elapsed
            ))
            log.info(f"  MLP-{model_name:6s} | {mlp_params:>8,} params | val={val_loss:.4f} | {elapsed:.1f}s")
            
            # Transformer (grouped tokenization - 24 tokens, not 141!)
            trans = GroupedTransformer(config["hidden"], config["layers"])
            trans_params = sum(p.numel() for p in trans.parameters())
            
            train_loss, val_loss, elapsed = train_model(trans, X, y, model_name=f"Trans-{model_name}")
            results.append(ExperimentResult(
                f"Trans-{model_name}", trans_params, data_size, train_loss, val_loss, elapsed
            ))
            log.info(f"  Trans-{model_name:6s} | {trans_params:>8,} params | val={val_loss:.4f} | {elapsed:.1f}s")
    
    # Save results
    output_dir = Path("results/nlhe_scaling")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_dict = [
        {
            "model": r.model_name,
            "params": r.model_params,
            "data_size": r.data_size,
            "train_loss": r.train_loss,
            "val_loss": r.val_loss,
            "train_time": r.train_time,
        }
        for r in results
    ]
    
    with open(output_dir / "scaling_results.json", "w") as f:
        json.dump(results_dict, f, indent=2)
    
    log.info("\n" + "="*60)
    log.info("📈 RESULTS SUMMARY")
    log.info("="*60)
    
    # Print table
    log.info(f"{'Model':<15} {'Params':>10} {'Data':>10} {'Val Loss':>12} {'Time':>8}")
    log.info("-"*55)
    for r in results:
        log.info(f"{r.model_name:<15} {r.model_params:>10,} {r.data_size:>10,} {r.val_loss:>12.4f} {r.train_time:>7.1f}s")
    
    log.info(f"\n✅ Results saved to {output_dir}/scaling_results.json")
    
    # Key insight
    log.info("\n🔑 KEY INSIGHTS:")
    for data_size in DATA_SIZES:
        mlp_losses = [r.val_loss for r in results if "MLP" in r.model_name and r.data_size == data_size]
        trans_losses = [r.val_loss for r in results if "Trans" in r.model_name and r.data_size == data_size]
        if mlp_losses and trans_losses:
            avg_mlp = sum(mlp_losses) / len(mlp_losses)
            avg_trans = sum(trans_losses) / len(trans_losses)
            winner = "Trans" if avg_trans < avg_mlp else "MLP"
            log.info(f"   D={data_size//1000}k: {winner} wins (MLP={avg_mlp:.4f}, Trans={avg_trans:.4f})")

if __name__ == "__main__":
    run_experiment()

