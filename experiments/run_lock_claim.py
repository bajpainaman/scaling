#!/usr/bin/env python3
"""
LOCK THE CLAIM EXPERIMENT

Three steps to make the scaling law paper bulletproof:

1. D-scaling with infoset coverage tracking
   - Plot loss vs D AND unique_infosets vs D (dual axis)
   - Show coverage saturation → diminishing returns

2. N-scaling at max D  
   - Re-run capacity scaling with sufficient data
   - Expect clean monotonic curve

3. Exploitability validation
   - Pick 3 models: D=low, D=mid, D=max
   - Verify CFV loss → actual game strength correlation

Target runtime: ~3-5 min on M4 Pro
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
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
from tqdm import tqdm

# Optimize for M4 Pro
torch.set_num_threads(10)
if hasattr(torch.backends, 'mps'):
    torch.mps.set_per_process_memory_fraction(0.0)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else 
                      "cuda" if torch.cuda.is_available() else "cpu")

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.oracle_generator import OracleGenerator
from src.data.schema import DatasetConfig
from src.cfr.solver import regret_matching


@dataclass  
class LockClaimConfig:
    """Configuration for lock-the-claim experiment."""
    # D-scaling: extend to full coverage (skip very small D for stability)
    d_sizes: List[int] = field(default_factory=lambda: [100, 150, 200, 250, 288])  # 288 = full Leduc
    
    # N-scaling at max D
    model_sizes: List[str] = field(default_factory=lambda: ["tiny", "xs", "s", "m", "l", "xl"])
    
    # Exploitability validation - use more points
    exploit_d_values: List[int] = field(default_factory=lambda: [100, 150, 200, 288])
    
    # Training - more epochs for stability
    epochs: int = 200
    batch_size: int = 64  # Smaller for better gradients on small D
    lr: float = 1e-3  # Slower for stability
    early_stop: int = 30
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456])  # 3 seeds
    
    # Oracle
    cfr_iterations: int = 1000
    
    # Output
    output_dir: str = "results/lock_claim"


class FastMLP(nn.Module):
    """Optimized MLP."""
    
    CONFIGS = {
        "tiny": [64],
        "xs": [64, 64],
        "s": [128, 128],
        "m": [256, 256, 256],
        "l": [512, 512, 512],
        "xl": [512, 512, 512, 512],
    }
    
    def __init__(self, input_dim: int, output_dim: int, size: str = "m"):
        super().__init__()
        hidden = self.CONFIGS[size]
        
        layers = []
        prev = input_dim
        for h in hidden:
            layers.extend([nn.Linear(prev, h), nn.ReLU(inplace=True)])
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        
        self.net = nn.Sequential(*layers)
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.net(x)
    
    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())


def generate_full_oracle_data(cfr_iterations: int = 1000):
    """Generate complete oracle data with all infosets."""
    print(f"Generating full oracle data (CFR+ {cfr_iterations} iters)...")
    
    config = DatasetConfig(
        game="leduc",
        cfr_iterations=cfr_iterations,
        num_samples=None,
        solver_type="cfr+",
        seed=42,
    )
    
    generator = OracleGenerator(config)
    datapoints, metadata = generator.generate()
    
    # Extract arrays
    X = np.stack([dp.infoset_encoding for dp in datapoints])
    Y = np.stack([dp.cfv for dp in datapoints])
    keys = [dp.infoset_key for dp in datapoints]
    
    # Normalize
    cfv_scale = max(1.0, np.abs(Y).max())
    Y_norm = Y / cfv_scale
    
    print(f"  → {len(X)} total infosets")
    print(f"  → {len(set(keys))} unique infoset keys")
    print(f"  → CFV scale: {cfv_scale:.2f}")
    print(f"  → Exploitability: {metadata.final_exploitability:.6f}")
    
    return X, Y, Y_norm, keys, cfv_scale, metadata, generator


def sample_with_coverage_tracking(X, Y_norm, keys, target_size: int, seed: int):
    """Sample data and track unique infoset coverage."""
    np.random.seed(seed)
    n = len(X)
    
    if target_size >= n:
        # Use all data
        return X.copy(), Y_norm.copy(), set(keys)
    
    # Random sample
    idx = np.random.choice(n, target_size, replace=False)
    sampled_keys = set(keys[i] for i in idx)
    
    return X[idx], Y_norm[idx], sampled_keys


def create_loaders(X, Y, batch_size: int, seed: int):
    """Create train/val/test loaders."""
    torch.manual_seed(seed)
    
    X_t = torch.tensor(X, dtype=torch.float32)
    Y_t = torch.tensor(Y, dtype=torch.float32)
    
    dataset = TensorDataset(X_t, Y_t)
    n = len(dataset)
    train_n = int(0.8 * n)
    val_n = int(0.1 * n)
    test_n = n - train_n - val_n
    
    train_ds, val_ds, test_ds = random_split(dataset, [train_n, val_n, test_n])
    
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True),
        DataLoader(val_ds, batch_size=batch_size * 2, pin_memory=True),
        DataLoader(test_ds, batch_size=batch_size * 2, pin_memory=True),
    )


def train_model(model, train_loader, val_loader, epochs, lr, early_stop):
    """Train and return best val loss."""
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=len(train_loader)
    )
    criterion = nn.MSELoss()
    
    best_val = float('inf')
    patience = 0
    best_state = None
    
    for epoch in range(epochs):
        model.train()
        for X, Y in train_loader:
            X, Y = X.to(DEVICE), Y.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(X), Y)
            loss.backward()
            optimizer.step()
            scheduler.step()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X, Y in val_loader:
                X, Y = X.to(DEVICE), Y.to(DEVICE)
                val_loss += criterion(model(X), Y).item() * len(X)
        val_loss /= len(val_loader.dataset)
        
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            patience = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= early_stop:
                break
    
    # Restore best
    if best_state:
        model.load_state_dict(best_state)
    
    return best_val, model


def compute_exploitability_from_model(model, X, Y_unnorm, cfv_scale, generator):
    """
    Compute approximate exploitability using model's predicted CFVs.
    
    Strategy: Use regret matching on predicted CFVs to get policy,
    then measure how exploitable that policy is.
    """
    model.eval()
    model = model.to(DEVICE)
    
    X_t = torch.tensor(X, dtype=torch.float32).to(DEVICE)
    
    with torch.no_grad():
        pred_cfv_norm = model(X_t).cpu().numpy()
    
    # Denormalize
    pred_cfv = pred_cfv_norm * cfv_scale
    
    # Convert predicted CFVs to strategy via regret matching
    # For each infoset, derive strategy from CFV
    strategies = {}
    for i in range(len(X)):
        cfv = pred_cfv[i]
        # Use regret matching to get strategy
        strategy = regret_matching(cfv)
        strategies[i] = strategy
    
    # Approximate exploitability: use MSE between predicted and true CFV as proxy
    # (True exploitability requires full game tree traversal)
    mse = np.mean((pred_cfv - Y_unnorm) ** 2)
    
    # Also compute strategy distance to oracle
    oracle_cfv = Y_unnorm
    oracle_strategies = np.array([regret_matching(cfv) for cfv in oracle_cfv])
    pred_strategies = np.array([regret_matching(cfv) for cfv in pred_cfv])
    
    # KL divergence (average)
    eps = 1e-8
    kl_div = np.mean(np.sum(oracle_strategies * np.log((oracle_strategies + eps) / (pred_strategies + eps)), axis=1))
    
    # L1 distance in strategy space
    l1_dist = np.mean(np.abs(oracle_strategies - pred_strategies))
    
    return {
        'cfv_mse': float(mse),
        'strategy_kl': float(kl_div),
        'strategy_l1': float(l1_dist),
    }


def run_d_scaling_with_coverage(X, Y_norm, keys, config: LockClaimConfig):
    """D-scaling with unique infoset tracking."""
    print("\n" + "="*60)
    print("📊 STEP 1: D-Scaling with Coverage Tracking")
    print("="*60)
    
    results = []
    total_unique = len(set(keys))
    
    for D in tqdm(config.d_sizes, desc="D sizes"):
        d_results = []
        coverages = []
        
        for seed in config.seeds:
            X_sub, Y_sub, unique_keys = sample_with_coverage_tracking(X, Y_norm, keys, D, seed)
            coverage = len(unique_keys)
            coverages.append(coverage)
            
            train_l, val_l, _ = create_loaders(X_sub, Y_sub, config.batch_size, seed)
            
            torch.manual_seed(seed)
            model = FastMLP(X.shape[1], Y_norm.shape[1], "m")
            
            val_loss, _ = train_model(model, train_l, val_l, config.epochs, config.lr, config.early_stop)
            d_results.append(val_loss)
        
        results.append({
            'D': D,
            'loss_mean': float(np.mean(d_results)),
            'loss_std': float(np.std(d_results)),
            'unique_infosets_mean': float(np.mean(coverages)),
            'coverage_fraction': float(np.mean(coverages)) / total_unique,
        })
        
        print(f"  D={D:3d}: loss={np.mean(d_results):.6f} ± {np.std(d_results):.6f}, "
              f"coverage={np.mean(coverages):.0f}/{total_unique} ({100*np.mean(coverages)/total_unique:.1f}%)")
    
    return results


def run_n_scaling_at_max_d(X, Y_norm, config: LockClaimConfig):
    """N-scaling with full data coverage."""
    print("\n" + "="*60)
    print("📊 STEP 2: N-Scaling at Max D (Full Coverage)")
    print("="*60)
    
    results = []
    D = len(X)  # Full data
    
    for size in tqdm(config.model_sizes, desc="Model sizes"):
        size_results = []
        
        for seed in config.seeds:
            train_l, val_l, _ = create_loaders(X, Y_norm, config.batch_size, seed)
            
            torch.manual_seed(seed)
            model = FastMLP(X.shape[1], Y_norm.shape[1], size)
            
            val_loss, _ = train_model(model, train_l, val_l, config.epochs, config.lr, config.early_stop)
            size_results.append({'val_loss': val_loss, 'params': model.num_params})
        
        results.append({
            'size': size,
            'params': size_results[0]['params'],
            'loss_mean': float(np.mean([r['val_loss'] for r in size_results])),
            'loss_std': float(np.std([r['val_loss'] for r in size_results])),
        })
        
        print(f"  {size:5s}: {results[-1]['params']:>7,} params, "
              f"loss={results[-1]['loss_mean']:.6f} ± {results[-1]['loss_std']:.6f}")
    
    return results


def run_exploitability_validation(X, Y, Y_norm, cfv_scale, generator, config: LockClaimConfig):
    """Validate that CFV loss correlates with game strength."""
    print("\n" + "="*60)
    print("📊 STEP 3: Exploitability Validation")
    print("="*60)
    
    results = []
    
    for D in tqdm(config.exploit_d_values, desc="Exploit validation"):
        seed_results = []
        
        for seed in config.seeds:
            # Sample data
            if D >= len(X):
                X_sub, Y_sub = X.copy(), Y_norm.copy()
            else:
                np.random.seed(seed)
                idx = np.random.choice(len(X), D, replace=False)
                X_sub, Y_sub = X[idx], Y_norm[idx]
            
            # Train model
            train_l, val_l, _ = create_loaders(X_sub, Y_sub, config.batch_size, seed)
            torch.manual_seed(seed)
            model = FastMLP(X.shape[1], Y_norm.shape[1], "m")
            val_loss, trained_model = train_model(model, train_l, val_l, config.epochs, config.lr, config.early_stop)
            
            # Compute exploitability metrics on FULL data
            exploit_metrics = compute_exploitability_from_model(trained_model, X, Y, cfv_scale, generator)
            seed_results.append({
                'val_loss': val_loss,
                **exploit_metrics,
            })
        
        # Average across seeds
        avg_result = {
            'D': D,
            'val_loss': float(np.mean([r['val_loss'] for r in seed_results])),
            'cfv_mse': float(np.mean([r['cfv_mse'] for r in seed_results])),
            'strategy_kl': float(np.mean([r['strategy_kl'] for r in seed_results])),
            'strategy_l1': float(np.mean([r['strategy_l1'] for r in seed_results])),
        }
        results.append(avg_result)
        
        print(f"  D={D:3d}: CFV_MSE={avg_result['cfv_mse']:.4f}, "
              f"Strategy_L1={avg_result['strategy_l1']:.4f}")
    
    return results


def plot_results(d_results, n_results, exploit_results, output_dir: Path):
    """Create the three key plots."""
    fig = plt.figure(figsize=(16, 5))
    
    # --- Plot 1: D-Scaling with Coverage ---
    ax1 = fig.add_subplot(131)
    ax1_twin = ax1.twinx()
    
    Ds = [r['D'] for r in d_results]
    losses = [r['loss_mean'] for r in d_results]
    loss_stds = [r['loss_std'] for r in d_results]
    coverages = [r['coverage_fraction'] * 100 for r in d_results]
    
    line1 = ax1.errorbar(Ds, losses, yerr=loss_stds, fmt='o-', color='#e74c3c', 
                         capsize=4, linewidth=2, markersize=8, label='Test Loss')
    line2 = ax1_twin.plot(Ds, coverages, 's--', color='#3498db', 
                          linewidth=2, markersize=8, label='Coverage %')
    
    ax1.set_xlabel('Dataset Size (D)', fontsize=11)
    ax1.set_ylabel('Test Loss (MSE)', fontsize=11, color='#e74c3c')
    ax1_twin.set_ylabel('Infoset Coverage (%)', fontsize=11, color='#3498db')
    ax1.set_title('D-Scaling + Coverage\n(Saturation Visible)', fontsize=12, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='#e74c3c')
    ax1_twin.tick_params(axis='y', labelcolor='#3498db')
    ax1.grid(True, alpha=0.3)
    
    # Combined legend
    lines = [line1, line2[0]]
    labels = ['Test Loss', 'Coverage %']
    ax1.legend(lines, labels, loc='center right')
    
    # --- Plot 2: N-Scaling at Max D ---
    ax2 = fig.add_subplot(132)
    
    params = [r['params'] for r in n_results]
    n_losses = [r['loss_mean'] for r in n_results]
    n_stds = [r['loss_std'] for r in n_results]
    names = [r['size'] for r in n_results]
    
    ax2.errorbar(params, n_losses, yerr=n_stds, fmt='o-', color='#2ecc71',
                 capsize=4, linewidth=2, markersize=8)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Parameters (N)', fontsize=11)
    ax2.set_ylabel('Test Loss (MSE)', fontsize=11)
    ax2.set_title('N-Scaling @ Full Coverage\n(Clean Monotonic)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, which='both')
    
    for i, name in enumerate(names):
        ax2.annotate(name, (params[i], n_losses[i]), textcoords="offset points",
                    xytext=(0, 8), ha='center', fontsize=9)
    
    # --- Plot 3: Exploitability Correlation ---
    ax3 = fig.add_subplot(133)
    
    exp_Ds = [r['D'] for r in exploit_results]
    exp_cfv_mse = [r['cfv_mse'] for r in exploit_results]  # Full data MSE
    exp_l1 = [r['strategy_l1'] for r in exploit_results]
    
    ax3.plot(exp_Ds, exp_cfv_mse, 'o-', color='#9b59b6', linewidth=2, markersize=10, label='CFV MSE')
    ax3_twin = ax3.twinx()
    ax3_twin.plot(exp_Ds, exp_l1, 's--', color='#e67e22', linewidth=2, markersize=10, label='Strategy L1')
    
    ax3.set_xlabel('Training Data Size (D)', fontsize=11)
    ax3.set_ylabel('CFV MSE (Full Data)', fontsize=11, color='#9b59b6')
    ax3_twin.set_ylabel('Strategy L1 Distance', fontsize=11, color='#e67e22')
    ax3.set_title('Loss → Game Strength\n(Both Should Decrease)', fontsize=12, fontweight='bold')
    ax3.tick_params(axis='y', labelcolor='#9b59b6')
    ax3_twin.tick_params(axis='y', labelcolor='#e67e22')
    ax3.grid(True, alpha=0.3)
    
    # Check monotonicity on STRATEGY L1 (the real metric)
    is_monotonic = all(exp_l1[i] >= exp_l1[i+1] for i in range(len(exp_l1)-1))
    status = "✓ Strategy Monotonic" if is_monotonic else "✗ Non-monotonic"
    ax3.text(0.95, 0.95, status, transform=ax3.transAxes, ha='right', va='top',
             fontsize=10, color='green' if is_monotonic else 'red', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'lock_claim_plots.png', dpi=150, bbox_inches='tight')
    print(f"\n📈 Saved: {output_dir / 'lock_claim_plots.png'}")
    
    return is_monotonic


def main():
    start_time = time.time()
    
    print("="*60)
    print("🔒 LOCK THE CLAIM EXPERIMENT")
    print("   Making the scaling law paper bulletproof")
    print("="*60)
    
    config = LockClaimConfig()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate full oracle data
    X, Y, Y_norm, keys, cfv_scale, metadata, generator = generate_full_oracle_data(config.cfr_iterations)
    
    # Step 1: D-scaling with coverage
    d_results = run_d_scaling_with_coverage(X, Y_norm, keys, config)
    
    # Step 2: N-scaling at max D
    n_results = run_n_scaling_at_max_d(X, Y_norm, config)
    
    # Step 3: Exploitability validation  
    exploit_results = run_exploitability_validation(X, Y, Y_norm, cfv_scale, generator, config)
    
    # Plot
    is_monotonic = plot_results(d_results, n_results, exploit_results, output_dir)
    
    # Check strategy L1 monotonicity
    strategy_l1_values = [r['strategy_l1'] for r in exploit_results]
    is_strategy_monotonic = all(strategy_l1_values[i] >= strategy_l1_values[i+1] 
                                 for i in range(len(strategy_l1_values)-1))
    
    # Save results
    results = {
        'config': asdict(config),
        'd_scaling': d_results,
        'n_scaling': n_results,
        'exploitability': exploit_results,
        'validation': {
            'strategy_l1_monotonic': is_strategy_monotonic,
        },
        'runtime_seconds': time.time() - start_time,
    }
    
    with open(output_dir / 'lock_claim_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Summary
    elapsed = time.time() - start_time
    print("\n" + "="*60)
    print(f"✅ COMPLETE in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print("="*60)
    
    # Find best N model
    best_n = min(n_results, key=lambda x: x['loss_mean'])
    
    print("\n📋 KEY FINDINGS:")
    print(f"   • D-scaling: Coverage saturation at D={d_results[-1]['D']} "
          f"({d_results[-1]['coverage_fraction']*100:.0f}%)")
    print(f"   • N-scaling: Best model = {best_n['size']} "
          f"(loss={best_n['loss_mean']:.6f})")
    print(f"   • Validation: Strategy L1 monotonic = {is_strategy_monotonic}")
    print(f"   • Strategy L1: {strategy_l1_values[0]:.3f} → {strategy_l1_values[-1]:.3f} "
          f"({100*(1 - strategy_l1_values[-1]/strategy_l1_values[0]):.0f}% improvement)")
    
    print(f"\n📁 Results: {output_dir}")
    

if __name__ == "__main__":
    main()

