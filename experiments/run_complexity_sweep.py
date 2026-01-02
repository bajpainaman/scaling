#!/usr/bin/env python3
"""
COMPLEXITY SWEEP EXPERIMENT

Runs bulletproof scaling protocol at multiple game complexities to show:
1. Scaling extends beyond the 288 infoset ceiling of standard Leduc
2. The same power-law relationship holds at higher complexity
3. Coverage saturation shifts right with more infosets

Complexity levels:
- C1: 3 ranks, 288 infosets (standard Leduc)
- C2: 4 ranks, 504 infosets
- C3: 5 ranks, 780 infosets

Runtime: ~8-10 min on M4 Pro (5 seeds, 3 complexities)
"""

import json
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List, Dict, Any, Tuple, Set
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from tqdm import tqdm

# Optimize for M4 Pro
torch.set_num_threads(10)
if hasattr(torch.backends, 'mps'):
    torch.mps.set_per_process_memory_fraction(0.0)

import sys
sys.path.insert(0, '.')

from src.data.oracle_generator import OracleGenerator, get_game, get_input_dim, encode_infoset
from src.data.schema import DatasetConfig
from src.cfr.dcfr import CFRPlus


@dataclass
class ComplexitySweepConfig:
    """Configuration for complexity sweep experiment."""
    # Complexity levels to test
    complexity_levels: List[str] = field(default_factory=lambda: ["leduc_C1", "leduc_C2", "leduc_C3"])
    
    # Model sizes to test (keep small for speed)
    model_sizes: List[str] = field(default_factory=lambda: ["s", "m", "l"])
    
    # D as fraction of max infosets (to compare across complexities)
    d_fractions: List[float] = field(default_factory=lambda: [0.25, 0.50, 0.75, 1.0])
    
    # Training
    cfr_iterations: int = 200
    epochs: int = 200
    batch_size: int = 128
    lr: float = 2e-3
    early_stop: int = 30
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456, 789, 999])


# Model configs
MODEL_CONFIGS = {
    "s":  {"width": 128, "depth": 3},
    "m":  {"width": 256, "depth": 4},
    "l":  {"width": 384, "depth": 5},
}


class FastMLP(nn.Module):
    """Simple MLP with configurable input dim."""
    def __init__(self, input_dim: int, output_dim: int, size: str):
        super().__init__()
        cfg = MODEL_CONFIGS[size]
        
        layers = [nn.Linear(input_dim, cfg["width"]), nn.ReLU()]
        for _ in range(cfg["depth"] - 1):
            layers.extend([nn.Linear(cfg["width"], cfg["width"]), nn.ReLU()])
        layers.append(nn.Linear(cfg["width"], output_dim))
        
        self.net = nn.Sequential(*layers)
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.net(x)


def generate_oracle_data(game_name: str, cfr_iterations: int) -> Tuple[np.ndarray, np.ndarray, List[str], int]:
    """Generate oracle data for a game, return X, Y, infoset_keys, num_actions."""
    config = DatasetConfig(
        game=game_name,
        cfr_iterations=cfr_iterations,
        num_samples=None,
        solver_type="cfr+",
        seed=42,
    )
    generator = OracleGenerator(config)
    datapoints, metadata = generator.generate()
    
    X = np.stack([dp.infoset_encoding for dp in datapoints])
    Y = np.stack([dp.cfv for dp in datapoints])
    keys = [dp.infoset_key for dp in datapoints]
    
    return X, Y, keys, Y.shape[1]


def create_group_split(X, Y, keys, train_frac, seed):
    """Split by infoset_key (group split) to prevent leakage."""
    np.random.seed(seed)
    unique_keys = list(set(keys))
    np.random.shuffle(unique_keys)
    
    n_train = int(len(unique_keys) * train_frac)
    train_keys = set(unique_keys[:n_train])
    
    train_mask = np.array([k in train_keys for k in keys])
    test_mask = ~train_mask
    
    return X[train_mask], Y[train_mask], X[test_mask], Y[test_mask]


def train_model(model, train_loader, epochs, lr, early_stop, device):
    """Train and return best validation loss (using last batch as pseudo-val)."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    model.train()
    best_loss = float('inf')
    patience = 0
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for X_batch, Y_batch in train_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = nn.functional.mse_loss(pred, Y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
        
        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience = 0
        else:
            patience += 1
            if patience >= early_stop:
                break
    
    return model


def evaluate_on_frozen_set(model, X_eval, Y_eval, device):
    """Evaluate model on frozen eval set."""
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X_eval, dtype=torch.float32).to(device)
        Y_t = torch.tensor(Y_eval, dtype=torch.float32).to(device)
        pred = model(X_t)
        mse = nn.functional.mse_loss(pred, Y_t).item()
        
        # Strategy L1 (regret matching)
        def regret_match(cfv):
            regrets = cfv - cfv.mean(dim=-1, keepdim=True)
            pos = torch.clamp(regrets, min=0)
            total = pos.sum(dim=-1, keepdim=True)
            return torch.where(total > 0, pos / total, torch.ones_like(pos) / pos.shape[-1])
        
        pred_strat = regret_match(pred)
        true_strat = regret_match(Y_t)
        l1 = (pred_strat - true_strat).abs().sum(dim=-1).mean().item()
    
    return mse, l1


def run_complexity_sweep(config: ComplexitySweepConfig):
    """Run the full complexity sweep experiment."""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    results_dir = Path("results/complexity_sweep")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    
    for complexity in config.complexity_levels:
        print(f"\n{'='*60}")
        print(f"COMPLEXITY: {complexity}")
        print("="*60)
        
        # Generate oracle data
        print("Generating oracle data...")
        X, Y, keys, num_actions = generate_oracle_data(complexity, config.cfr_iterations)
        
        # Normalize CFVs
        cfv_scale = np.abs(Y).max() + 1e-8
        Y_norm = Y / cfv_scale
        
        num_infosets = len(set(keys))
        input_dim = X.shape[1]
        
        print(f"  Infosets: {num_infosets}, Input dim: {input_dim}, Actions: {num_actions}")
        
        # Frozen eval set = all infosets
        X_eval, Y_eval = X.copy(), Y_norm.copy()
        
        complexity_results = {
            'game': complexity,
            'num_infosets': num_infosets,
            'input_dim': input_dim,
            'd_results': [],
        }
        
        # D-scaling at this complexity
        for d_frac in config.d_fractions:
            D = max(int(num_infosets * d_frac), 20)
            
            losses = []
            l1s = []
            
            for seed in tqdm(config.seeds, desc=f"D={D} ({d_frac*100:.0f}%)"):
                # Sample D infosets for training
                np.random.seed(seed)
                if D >= num_infosets:
                    X_train, Y_train = X.copy(), Y_norm.copy()
                else:
                    unique_keys = list(set(keys))
                    np.random.shuffle(unique_keys)
                    train_keys = set(unique_keys[:D])
                    train_mask = np.array([k in train_keys for k in keys])
                    X_train, Y_train = X[train_mask], Y_norm[train_mask]
                
                # Train model (use 'm' for all D comparisons)
                train_dataset = TensorDataset(
                    torch.tensor(X_train, dtype=torch.float32),
                    torch.tensor(Y_train, dtype=torch.float32)
                )
                train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
                
                torch.manual_seed(seed)
                model = FastMLP(input_dim, num_actions, "m").to(device)
                model = train_model(model, train_loader, config.epochs, config.lr, config.early_stop, device)
                
                # Eval on frozen set
                mse, l1 = evaluate_on_frozen_set(model, X_eval, Y_eval, device)
                losses.append(mse)
                l1s.append(l1)
            
            d_result = {
                'D': D,
                'd_fraction': d_frac,
                'coverage': D / num_infosets,
                'loss_mean': float(np.mean(losses)),
                'loss_std': float(np.std(losses)),
                'l1_mean': float(np.mean(l1s)),
                'l1_std': float(np.std(l1s)),
            }
            complexity_results['d_results'].append(d_result)
            
            print(f"  D={D:4d} ({d_frac*100:3.0f}%): loss={np.mean(losses):.6f}±{np.std(losses):.6f}")
        
        all_results[complexity] = complexity_results
    
    # Save results
    with open(results_dir / "complexity_sweep.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Plot
    create_plots(all_results, results_dir)
    
    return all_results


def create_plots(results: Dict, results_dir: Path):
    """Create comparison plots across complexity levels."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    colors = {'leduc_C1': '#e74c3c', 'leduc_C2': '#3498db', 'leduc_C3': '#2ecc71'}
    labels = {'leduc_C1': 'C1 (288)', 'leduc_C2': 'C2 (504)', 'leduc_C3': 'C3 (780)'}
    
    # --- Plot 1: Loss vs D (absolute) ---
    ax1 = axes[0]
    for complexity, data in results.items():
        Ds = [r['D'] for r in data['d_results']]
        means = [r['loss_mean'] for r in data['d_results']]
        stds = [r['loss_std'] for r in data['d_results']]
        ax1.errorbar(Ds, means, yerr=stds, fmt='o-', capsize=4, 
                     color=colors[complexity], label=labels[complexity], linewidth=2, markersize=8)
    
    ax1.set_xlabel('Dataset Size (D)', fontsize=12)
    ax1.set_ylabel('Eval CFV MSE', fontsize=12)
    ax1.set_title('D-Scaling Across Complexities\n(Absolute D)', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    
    # --- Plot 2: Loss vs Coverage (normalized) ---
    ax2 = axes[1]
    for complexity, data in results.items():
        covs = [r['coverage'] for r in data['d_results']]
        means = [r['loss_mean'] for r in data['d_results']]
        stds = [r['loss_std'] for r in data['d_results']]
        ax2.errorbar(covs, means, yerr=stds, fmt='o-', capsize=4,
                     color=colors[complexity], label=labels[complexity], linewidth=2, markersize=8)
    
    ax2.set_xlabel('Coverage (D / Total Infosets)', fontsize=12)
    ax2.set_ylabel('Eval CFV MSE', fontsize=12)
    ax2.set_title('D-Scaling by Coverage\n(Normalized)', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    
    # --- Plot 3: Infosets vs Final Loss ---
    ax3 = axes[2]
    infosets = []
    final_losses = []
    final_stds = []
    complexity_labels = []
    
    for complexity, data in results.items():
        infosets.append(data['num_infosets'])
        final_losses.append(data['d_results'][-1]['loss_mean'])
        final_stds.append(data['d_results'][-1]['loss_std'])
        complexity_labels.append(labels[complexity])
    
    bars = ax3.bar(complexity_labels, final_losses, yerr=final_stds, capsize=8,
                   color=[colors[c] for c in results.keys()], alpha=0.8)
    
    # Add infoset counts on bars
    for bar, n in zip(bars, infosets):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005,
                f'{n} infosets', ha='center', va='bottom', fontsize=10)
    
    ax3.set_ylabel('Eval CFV MSE (at 100% coverage)', fontsize=12)
    ax3.set_title('Final Loss by Complexity\n(Full Coverage)', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(results_dir / "complexity_sweep.png", dpi=150, bbox_inches='tight')
    print(f"\n📊 Saved: {results_dir / 'complexity_sweep.png'}")


if __name__ == "__main__":
    start_time = time.time()
    
    config = ComplexitySweepConfig()
    print("🔬 COMPLEXITY SWEEP EXPERIMENT")
    print("="*60)
    print(f"Complexity levels: {config.complexity_levels}")
    print(f"D fractions: {config.d_fractions}")
    print(f"Seeds: {len(config.seeds)}")
    
    results = run_complexity_sweep(config)
    
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"✅ COMPLETE in {elapsed/60:.1f} minutes")
    print("="*60)
    
    # Summary
    print("\n📋 SUMMARY:")
    for complexity, data in results.items():
        first_loss = data['d_results'][0]['loss_mean']
        last_loss = data['d_results'][-1]['loss_mean']
        improvement = first_loss / last_loss
        print(f"  {complexity}: {data['num_infosets']} infosets, "
              f"loss {first_loss:.4f} → {last_loss:.6f} ({improvement:.0f}× improvement)")

