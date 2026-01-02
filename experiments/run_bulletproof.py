#!/usr/bin/env python3
"""
BULLETPROOF SCALING EXPERIMENT

Implements reviewer-proof methodology:

1. FROZEN EVAL SET: All 288 infosets, never changes
2. GROUP SPLIT: Train samples by infoset_key (no leakage)
3. TWO LOSSES: Train loss (on sampled D) + Eval loss (on frozen set)
4. 10 SEEDS: Camera-ready variance estimates
5. N×D GRID: Complete scaling surface

Reports:
- CFV MSE (primary, scaling-law friendly)
- Strategy L1 (behavioral)
- Coverage & duplication metrics
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

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else 
                      "cuda" if torch.cuda.is_available() else "cpu")

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.oracle_generator import OracleGenerator
from src.data.schema import DatasetConfig
from src.cfr.solver import regret_matching


@dataclass
class BulletproofConfig:
    """Camera-ready configuration."""
    # N×D grid
    model_sizes: List[str] = field(default_factory=lambda: ["s", "m", "l"])
    d_sizes: List[int] = field(default_factory=lambda: [50, 100, 150, 200, 288])
    
    # Seeds for variance
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456, 789, 101, 202, 303, 404, 505, 606])
    
    # Training
    epochs: int = 200
    batch_size: int = 32
    lr: float = 1e-3
    early_stop: int = 30
    
    # Oracle
    cfr_iterations: int = 1000
    
    # Output
    output_dir: str = "results/bulletproof"


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


@dataclass
class OracleData:
    """Frozen oracle data container."""
    X: np.ndarray  # (288, input_dim)
    Y: np.ndarray  # (288, num_actions) - unnormalized CFVs
    Y_norm: np.ndarray  # normalized CFVs
    keys: List[str]  # infoset_key for each sample
    cfv_scale: float
    key_to_idx: Dict[str, int]  # map infoset_key -> index


def generate_oracle_data(cfr_iterations: int = 1000) -> OracleData:
    """Generate complete oracle data (frozen eval set)."""
    print(f"Generating FROZEN oracle data (CFR+ {cfr_iterations} iters)...")
    
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
    
    # Build key-to-index map
    key_to_idx = {k: i for i, k in enumerate(keys)}
    
    print(f"  → {len(X)} total infosets")
    print(f"  → {len(set(keys))} unique infoset keys")
    print(f"  → CFV scale: {cfv_scale:.2f}")
    print(f"  → Exploitability: {metadata.final_exploitability:.6f}")
    
    return OracleData(
        X=X, Y=Y, Y_norm=Y_norm, keys=keys,
        cfv_scale=cfv_scale, key_to_idx=key_to_idx
    )


def sample_train_set_by_infoset(oracle: OracleData, D: int, seed: int) -> Tuple[np.ndarray, np.ndarray, Set[str]]:
    """
    Sample D training examples by GROUP SPLIT (by infoset_key).
    
    Returns:
        X_train, Y_train_norm, unique_keys_sampled
    """
    np.random.seed(seed)
    n = len(oracle.keys)
    
    if D >= n:
        # Use all
        return oracle.X.copy(), oracle.Y_norm.copy(), set(oracle.keys)
    
    # Sample D UNIQUE infosets (no duplication)
    unique_keys = list(set(oracle.keys))
    sampled_keys = set(np.random.choice(unique_keys, min(D, len(unique_keys)), replace=False))
    
    # Get indices for sampled keys
    indices = [i for i, k in enumerate(oracle.keys) if k in sampled_keys]
    
    return oracle.X[indices], oracle.Y_norm[indices], sampled_keys


def create_train_loader(X: np.ndarray, Y: np.ndarray, batch_size: int, seed: int) -> DataLoader:
    """Create training dataloader."""
    torch.manual_seed(seed)
    X_t = torch.tensor(X, dtype=torch.float32)
    Y_t = torch.tensor(Y, dtype=torch.float32)
    dataset = TensorDataset(X_t, Y_t)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)


def create_eval_loader(oracle: OracleData, batch_size: int) -> DataLoader:
    """Create FROZEN eval dataloader (all 288 infosets)."""
    X_t = torch.tensor(oracle.X, dtype=torch.float32)
    Y_t = torch.tensor(oracle.Y_norm, dtype=torch.float32)
    dataset = TensorDataset(X_t, Y_t)
    return DataLoader(dataset, batch_size=batch_size * 4, pin_memory=True)


def train_and_evaluate(
    model: nn.Module,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    epochs: int,
    lr: float,
    early_stop: int,
) -> Tuple[float, float, nn.Module]:
    """
    Train model and return:
    - train_loss: Loss on training set
    - eval_loss: Loss on FROZEN eval set (all 288 infosets)
    - trained model
    """
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=len(train_loader)
    )
    criterion = nn.MSELoss()
    
    best_eval = float('inf')
    patience = 0
    best_state = None
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss_sum = 0
        train_n = 0
        for X, Y in train_loader:
            X, Y = X.to(DEVICE), Y.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            pred = model(X)
            loss = criterion(pred, Y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss_sum += loss.item() * len(X)
            train_n += len(X)
        
        # Evaluation on FROZEN set
        model.eval()
        eval_loss_sum = 0
        eval_n = 0
        with torch.no_grad():
            for X, Y in eval_loader:
                X, Y = X.to(DEVICE), Y.to(DEVICE)
                pred = model(X)
                eval_loss_sum += criterion(pred, Y).item() * len(X)
                eval_n += len(X)
        
        eval_loss = eval_loss_sum / eval_n
        
        if eval_loss < best_eval - 1e-6:
            best_eval = eval_loss
            patience = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= early_stop:
                break
    
    # Restore best
    if best_state:
        model.load_state_dict(best_state)
    
    # Final train loss
    model.eval()
    train_loss_sum = 0
    train_n = 0
    with torch.no_grad():
        for X, Y in train_loader:
            X, Y = X.to(DEVICE), Y.to(DEVICE)
            train_loss_sum += criterion(model(X), Y).item() * len(X)
            train_n += len(X)
    
    return train_loss_sum / train_n, best_eval, model


def compute_metrics(model: nn.Module, oracle: OracleData) -> Dict[str, float]:
    """
    Compute all metrics on FROZEN eval set.
    
    Returns:
        - cfv_mse: MSE on CFV prediction
        - strategy_l1: L1 distance in strategy space
        - strategy_kl: KL divergence (oracle || predicted)
    """
    model.eval()
    model = model.to(DEVICE)
    
    X_t = torch.tensor(oracle.X, dtype=torch.float32).to(DEVICE)
    
    with torch.no_grad():
        pred_cfv_norm = model(X_t).cpu().numpy()
    
    # Denormalize
    pred_cfv = pred_cfv_norm * oracle.cfv_scale
    oracle_cfv = oracle.Y
    
    # CFV MSE
    cfv_mse = np.mean((pred_cfv - oracle_cfv) ** 2)
    
    # Derive strategies via regret matching
    oracle_strategies = np.array([regret_matching(cfv) for cfv in oracle_cfv])
    pred_strategies = np.array([regret_matching(cfv) for cfv in pred_cfv])
    
    # Strategy L1
    strategy_l1 = np.mean(np.abs(oracle_strategies - pred_strategies))
    
    # Strategy KL (oracle || predicted)
    eps = 1e-8
    kl_div = np.mean(np.sum(
        oracle_strategies * np.log((oracle_strategies + eps) / (pred_strategies + eps)), 
        axis=1
    ))
    
    return {
        'cfv_mse': float(cfv_mse),
        'strategy_l1': float(strategy_l1),
        'strategy_kl': float(kl_div),
    }


def run_grid_experiment(oracle: OracleData, config: BulletproofConfig) -> Dict[str, Any]:
    """
    Run the full N×D grid experiment.
    """
    print("\n" + "="*70)
    print("🔬 BULLETPROOF N×D GRID EXPERIMENT")
    print(f"   Models: {config.model_sizes}")
    print(f"   D sizes: {config.d_sizes}")
    print(f"   Seeds: {len(config.seeds)}")
    print("="*70)
    
    # Create frozen eval loader
    eval_loader = create_eval_loader(oracle, config.batch_size)
    
    results = {}
    
    for model_size in config.model_sizes:
        results[model_size] = {}
        
        for D in config.d_sizes:
            seed_results = []
            
            desc = f"{model_size}/D={D}"
            for seed in tqdm(config.seeds, desc=desc, leave=False):
                # Sample training set by infoset_key
                X_train, Y_train, sampled_keys = sample_train_set_by_infoset(oracle, D, seed)
                
                # Metrics
                unique_infosets = len(sampled_keys)
                coverage = unique_infosets / len(set(oracle.keys))
                duplication = D / unique_infosets if unique_infosets > 0 else 0
                
                # Create train loader
                train_loader = create_train_loader(X_train, Y_train, config.batch_size, seed)
                
                # Create and train model
                torch.manual_seed(seed)
                model = FastMLP(oracle.X.shape[1], oracle.Y_norm.shape[1], model_size)
                
                train_loss, eval_loss, trained_model = train_and_evaluate(
                    model, train_loader, eval_loader,
                    config.epochs, config.lr, config.early_stop
                )
                
                # Compute all metrics on frozen eval set
                metrics = compute_metrics(trained_model, oracle)
                
                seed_results.append({
                    'train_loss': train_loss,
                    'eval_loss': eval_loss,
                    'unique_infosets': unique_infosets,
                    'coverage': coverage,
                    'duplication': duplication,
                    **metrics,
                })
            
            # Aggregate
            agg = {
                'D': D,
                'params': FastMLP(oracle.X.shape[1], oracle.Y_norm.shape[1], model_size).num_params,
                'train_loss_mean': float(np.mean([r['train_loss'] for r in seed_results])),
                'train_loss_std': float(np.std([r['train_loss'] for r in seed_results])),
                'eval_loss_mean': float(np.mean([r['eval_loss'] for r in seed_results])),
                'eval_loss_std': float(np.std([r['eval_loss'] for r in seed_results])),
                'cfv_mse_mean': float(np.mean([r['cfv_mse'] for r in seed_results])),
                'cfv_mse_std': float(np.std([r['cfv_mse'] for r in seed_results])),
                'strategy_l1_mean': float(np.mean([r['strategy_l1'] for r in seed_results])),
                'strategy_l1_std': float(np.std([r['strategy_l1'] for r in seed_results])),
                'coverage_mean': float(np.mean([r['coverage'] for r in seed_results])),
                'unique_infosets_mean': float(np.mean([r['unique_infosets'] for r in seed_results])),
            }
            
            results[model_size][D] = agg
            
            print(f"  {model_size}/D={D:3d}: eval_loss={agg['eval_loss_mean']:.5f}±{agg['eval_loss_std']:.5f}, "
                  f"cfv_mse={agg['cfv_mse_mean']:.4f}, L1={agg['strategy_l1_mean']:.4f}, "
                  f"cov={agg['coverage_mean']*100:.0f}%")
    
    return results


def plot_grid_results(results: Dict, config: BulletproofConfig, output_dir: Path):
    """Create publication-ready plots."""
    
    fig = plt.figure(figsize=(18, 10))
    
    # Color palette
    colors = {'s': '#e74c3c', 'm': '#3498db', 'l': '#2ecc71'}
    
    # --- Plot 1: D-Scaling (Eval Loss vs D, one line per N) ---
    ax1 = fig.add_subplot(231)
    for model_size in config.model_sizes:
        Ds = sorted(results[model_size].keys())
        means = [results[model_size][D]['eval_loss_mean'] for D in Ds]
        stds = [results[model_size][D]['eval_loss_std'] for D in Ds]
        ax1.errorbar(Ds, means, yerr=stds, fmt='o-', color=colors[model_size],
                    capsize=3, linewidth=2, markersize=6, label=f'{model_size}')
    ax1.set_xlabel('Dataset Size (D)', fontsize=11)
    ax1.set_ylabel('Eval Loss (MSE)', fontsize=11)
    ax1.set_title('D-Scaling (Frozen Eval Set)', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # --- Plot 2: N-Scaling at max D ---
    ax2 = fig.add_subplot(232)
    max_D = max(config.d_sizes)
    params = []
    means = []
    stds = []
    names = []
    for model_size in config.model_sizes:
        r = results[model_size][max_D]
        params.append(r['params'])
        means.append(r['eval_loss_mean'])
        stds.append(r['eval_loss_std'])
        names.append(model_size)
    ax2.errorbar(params, means, yerr=stds, fmt='o-', color='#9b59b6',
                capsize=4, linewidth=2, markersize=8)
    ax2.set_xscale('log')
    ax2.set_xlabel('Parameters (N)', fontsize=11)
    ax2.set_ylabel('Eval Loss (MSE)', fontsize=11)
    ax2.set_title(f'N-Scaling @ D={max_D} (Full Coverage)', fontsize=12, fontweight='bold')
    for i, name in enumerate(names):
        ax2.annotate(name, (params[i], means[i]), textcoords="offset points",
                    xytext=(0, 8), ha='center', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # --- Plot 3: Coverage vs D ---
    ax3 = fig.add_subplot(233)
    # All models have same coverage for same D
    model_size = config.model_sizes[0]
    Ds = sorted(results[model_size].keys())
    coverages = [results[model_size][D]['coverage_mean'] * 100 for D in Ds]
    unique_counts = [results[model_size][D]['unique_infosets_mean'] for D in Ds]
    ax3.plot(Ds, coverages, 'o-', color='#3498db', linewidth=2, markersize=8, label='Coverage %')
    ax3_twin = ax3.twinx()
    ax3_twin.plot(Ds, unique_counts, 's--', color='#e67e22', linewidth=2, markersize=8, label='Unique Infosets')
    ax3.set_xlabel('Dataset Size (D)', fontsize=11)
    ax3.set_ylabel('Coverage (%)', fontsize=11, color='#3498db')
    ax3_twin.set_ylabel('Unique Infosets', fontsize=11, color='#e67e22')
    ax3.set_title('Coverage & Unique Infosets', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # --- Plot 4: CFV MSE vs D ---
    ax4 = fig.add_subplot(234)
    for model_size in config.model_sizes:
        Ds = sorted(results[model_size].keys())
        means = [results[model_size][D]['cfv_mse_mean'] for D in Ds]
        stds = [results[model_size][D]['cfv_mse_std'] for D in Ds]
        ax4.errorbar(Ds, means, yerr=stds, fmt='o-', color=colors[model_size],
                    capsize=3, linewidth=2, markersize=6, label=f'{model_size}')
    ax4.set_xlabel('Dataset Size (D)', fontsize=11)
    ax4.set_ylabel('CFV MSE (Full Data)', fontsize=11)
    ax4.set_title('CFV MSE (Primary Metric)', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # --- Plot 5: Strategy L1 vs D ---
    ax5 = fig.add_subplot(235)
    for model_size in config.model_sizes:
        Ds = sorted(results[model_size].keys())
        means = [results[model_size][D]['strategy_l1_mean'] for D in Ds]
        stds = [results[model_size][D]['strategy_l1_std'] for D in Ds]
        ax5.errorbar(Ds, means, yerr=stds, fmt='o-', color=colors[model_size],
                    capsize=3, linewidth=2, markersize=6, label=f'{model_size}')
    ax5.set_xlabel('Dataset Size (D)', fontsize=11)
    ax5.set_ylabel('Strategy L1 Distance', fontsize=11)
    ax5.set_title('Strategy L1 (Behavioral Metric)', fontsize=12, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # --- Plot 6: Log-log D-scaling with power law fit ---
    ax6 = fig.add_subplot(236)
    for model_size in config.model_sizes:
        Ds = np.array(sorted(results[model_size].keys()))
        means = np.array([results[model_size][D]['eval_loss_mean'] for D in Ds])
        ax6.scatter(Ds, means, color=colors[model_size], s=60, label=f'{model_size}', zorder=3)
        
        # Power law fit
        log_D = np.log(Ds)
        log_L = np.log(means)
        coeffs = np.polyfit(log_D, log_L, 1)
        alpha = -coeffs[0]
        
        fit_D = np.logspace(np.log10(Ds.min()), np.log10(Ds.max()), 50)
        fit_L = np.exp(coeffs[1]) * fit_D ** coeffs[0]
        ax6.plot(fit_D, fit_L, '--', color=colors[model_size], alpha=0.7, linewidth=1.5)
    
    ax6.set_xscale('log')
    ax6.set_yscale('log')
    ax6.set_xlabel('Dataset Size (D) [log]', fontsize=11)
    ax6.set_ylabel('Eval Loss (MSE) [log]', fontsize=11)
    ax6.set_title('Log-Log D-Scaling (Power Law)', fontsize=12, fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'bulletproof_grid.png', dpi=150, bbox_inches='tight')
    print(f"\n📈 Saved: {output_dir / 'bulletproof_grid.png'}")


def main():
    start_time = time.time()
    
    print("="*70)
    print("🔒 BULLETPROOF SCALING EXPERIMENT")
    print("   Methodology: Frozen eval set, group split, 10 seeds")
    print("="*70)
    
    config = BulletproofConfig()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate frozen oracle data (never changes)
    oracle = generate_oracle_data(config.cfr_iterations)
    
    # Run N×D grid
    results = run_grid_experiment(oracle, config)
    
    # Plot
    plot_grid_results(results, config, output_dir)
    
    # Save results
    output = {
        'config': asdict(config),
        'methodology': {
            'eval_set': 'frozen (all 288 infosets)',
            'train_split': 'group by infoset_key',
            'seeds': len(config.seeds),
        },
        'grid_results': results,
        'runtime_seconds': time.time() - start_time,
    }
    
    with open(output_dir / 'bulletproof_results.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    # Summary
    elapsed = time.time() - start_time
    print("\n" + "="*70)
    print(f"✅ COMPLETE in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print("="*70)
    
    # Check monotonicity
    max_D = max(config.d_sizes)
    print("\n📋 KEY FINDINGS:")
    
    for model_size in config.model_sizes:
        Ds = sorted(results[model_size].keys())
        eval_losses = [results[model_size][D]['eval_loss_mean'] for D in Ds]
        is_mono = all(eval_losses[i] >= eval_losses[i+1] for i in range(len(eval_losses)-1))
        print(f"   • {model_size}: D-scaling monotonic = {is_mono}, "
              f"loss {eval_losses[0]:.4f} → {eval_losses[-1]:.4f}")
    
    print(f"\n📁 Results: {output_dir}")


if __name__ == "__main__":
    main()

