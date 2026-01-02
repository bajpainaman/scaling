#!/usr/bin/env python3
"""
Scaling D Experiment: Dataset size vs CFV prediction loss.

The diagnosis from N-scaling:
- Performance plateaus at m (50K params)
- Suggests we're D-limited, not N-limited

This experiment tests that hypothesis:
- Fix model size at N=m (the stable regime)
- Vary oracle data quantity D
- Plot loss vs D

Expected: If D-limited, loss should drop with more data.
"""

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np

import torch
from tqdm import tqdm

from src.data import CFVDataset
from src.data.schema import DatasetMetadata, DatasetConfig
from src.models import MLP, get_config
from src.training import Trainer, TrainingConfig, TrainingResult


@dataclass
class DScalingConfig:
    """Configuration for D-scaling experiment."""
    # Game
    game: str = "leduc"
    
    # Fixed model size (in stable regime from N-scaling)
    model_size: str = "m"
    
    # Oracle quality (higher = closer to Nash)
    cfr_iterations: int = 1000
    
    # Dataset sizes to test
    # Full Leduc has ~288 infosets, so we need to vary CFR iterations or sampling
    # For now, we'll generate different amounts of oracle data
    dataset_sizes: Tuple[int, ...] = (100, 250, 500, 1000, 2000, 5000)
    
    # Seeds for statistical significance
    seeds: Tuple[int, ...] = (42, 123, 456)
    
    # Training
    epochs: int = 200
    batch_size: int = 32
    learning_rate: float = 1e-3
    early_stopping_patience: int = 30
    
    # Output
    output_dir: str = "results/scaling_D"


def generate_data_with_size(
    game: str,
    target_size: int,
    cfr_iterations: int,
    seed: int,
) -> Tuple[List, DatasetMetadata, float]:
    """
    Generate oracle data, potentially with augmentation to reach target size.
    
    For small games like Leduc (288 infosets), we can:
    1. Use all infosets if target <= actual
    2. Add noise/perturbation for larger datasets (data augmentation)
    
    For now, we'll use subsampling and repetition with noise.
    """
    from src.data.oracle_generator import OracleGenerator
    from src.data.schema import DatasetConfig
    
    # Set seed for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Create config for generator
    config = DatasetConfig(
        game=game,
        cfr_iterations=cfr_iterations,
        num_samples=None,  # Full enumeration of infosets
        solver_type="cfr+",
        seed=seed,
    )
    
    generator = OracleGenerator(config)
    
    # Generate base oracle data
    all_datapoints, metadata = generator.generate()
    base_size = len(all_datapoints)
    
    # Compute cfv_scale from data (max abs CFV for normalization)
    all_cfvs = np.concatenate([dp.cfv for dp in all_datapoints])
    cfv_scale = max(1.0, np.abs(all_cfvs).max())
    
    # Adjust to target size
    if target_size <= base_size:
        # Subsample
        indices = np.random.choice(base_size, size=target_size, replace=False)
        datapoints = [all_datapoints[i] for i in indices]
    else:
        # Repeat with small noise (simple augmentation)
        datapoints = list(all_datapoints)
        while len(datapoints) < target_size:
            # Add copies with tiny noise to CFV (simulates solver variance)
            for dp in all_datapoints:
                if len(datapoints) >= target_size:
                    break
                # Create noisy copy
                noisy_dp = type(dp)(
                    infoset_encoding=dp.infoset_encoding.copy(),
                    cfv=dp.cfv + np.random.normal(0, 0.01, dp.cfv.shape),
                    strategy=dp.strategy.copy(),
                    regrets=dp.regrets.copy(),
                    infoset_key=dp.infoset_key,
                    player=dp.player,
                    reach_prob=dp.reach_prob,
                    weight=dp.weight,
                )
                datapoints.append(noisy_dp)
    
    # Update metadata
    metadata.num_datapoints = len(datapoints)
    
    return datapoints, metadata, cfv_scale


def run_single_experiment(
    model_size: str,
    dataset_size: int,
    datapoints: List,
    metadata: DatasetMetadata,
    cfv_scale: float,
    config: DScalingConfig,
    seed: int,
) -> Dict[str, Any]:
    """Run a single training run."""
    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create dataset
    dataset = CFVDataset(datapoints, cfv_scale)
    
    # Split: 80/10/10
    n = len(dataset)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    n_test = n - n_train - n_val
    
    train_ds, val_ds, test_ds = torch.utils.data.random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(seed)
    )
    
    # Create dataloaders
    batch_size = min(config.batch_size, n_train)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size)
    
    # Create model
    model = MLP.from_name(
        model_size,
        input_dim=metadata.input_dim,
        num_actions=metadata.num_actions,
    )
    
    # Training config
    train_config = TrainingConfig(
        epochs=config.epochs,
        learning_rate=config.learning_rate,
        early_stopping_patience=config.early_stopping_patience,
        verbose=False,
    )
    
    # Train
    trainer = Trainer(model, train_loader, val_loader, train_config)
    result = trainer.train()
    
    # Evaluate on test set
    model.eval()
    test_losses = []
    with torch.no_grad():
        for batch in test_loader:
            pred = model(batch['encoding'])
            loss = torch.nn.functional.mse_loss(pred, batch['cfv'])
            test_losses.append(loss.item())
    test_loss = np.mean(test_losses) if test_losses else result.best_val_loss
    
    return {
        'model_size': model_size,
        'dataset_size': dataset_size,
        'seed': seed,
        'params': sum(p.numel() for p in model.parameters()),
        'train_loss': result.final_train_loss,
        'val_loss': result.best_val_loss,
        'test_loss': test_loss,
        'epochs_trained': result.total_epochs,
        'training_time': result.training_time_seconds,
    }


def run_experiment(config: DScalingConfig) -> Dict[str, Any]:
    """Run the full D-scaling experiment."""
    print("=" * 60)
    print("D-SCALING EXPERIMENT")
    print("=" * 60)
    print(f"Game: {config.game}")
    print(f"Model size: {config.model_size}")
    print(f"Dataset sizes: {config.dataset_sizes}")
    print(f"Seeds per size: {len(config.seeds)}")
    print(f"Total runs: {len(config.dataset_sizes) * len(config.seeds)}")
    print("=" * 60)
    
    results = []
    start_time = time.time()
    
    # For each dataset size
    for d_size in tqdm(config.dataset_sizes, desc="Dataset sizes"):
        print(f"\n--- Dataset size: {d_size} ---")
        
        for seed in config.seeds:
            # Generate data with target size
            datapoints, metadata, cfv_scale = generate_data_with_size(
                config.game,
                d_size,
                config.cfr_iterations,
                seed,
            )
            
            # Run training
            result = run_single_experiment(
                config.model_size,
                d_size,
                datapoints,
                metadata,
                cfv_scale,
                config,
                seed,
            )
            results.append(result)
            
            print(f"  seed={seed}: test_loss={result['test_loss']:.6f}")
    
    total_time = time.time() - start_time
    
    # Aggregate results
    size_results = {}
    for d_size in config.dataset_sizes:
        size_data = [r for r in results if r['dataset_size'] == d_size]
        losses = [r['test_loss'] for r in size_data]
        size_results[d_size] = {
            'dataset_size': d_size,
            'loss_mean': float(np.mean(losses)),
            'loss_std': float(np.std(losses)),
            'loss_min': float(np.min(losses)),
            'loss_max': float(np.max(losses)),
            'runs': len(size_data),
        }
    
    return {
        'config': asdict(config),
        'size_results': size_results,
        'all_results': results,
        'total_time': total_time,
        'metadata': {
            'game': config.game,
            'model_size': config.model_size,
            'model_params': results[0]['params'] if results else 0,
        },
    }


def plot_results(results: Dict[str, Any], output_path: Path):
    """Generate the scaling curve plot."""
    import matplotlib.pyplot as plt
    
    size_results = results['size_results']
    
    # Extract data
    sizes = sorted([int(k) for k in size_results.keys()])
    means = [size_results[s]['loss_mean'] for s in sizes]
    stds = [size_results[s]['loss_std'] for s in sizes]
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Linear scale
    ax1 = axes[0]
    ax1.errorbar(sizes, means, yerr=stds, fmt='o-', capsize=5, 
                 color='#2E86AB', markersize=8, linewidth=2)
    ax1.set_xlabel('Dataset Size (D)', fontsize=12)
    ax1.set_ylabel('Test Loss (MSE)', fontsize=12)
    ax1.set_title(f'D-Scaling: {results["metadata"]["game"].title()} Poker\n'
                  f'Model: {results["metadata"]["model_size"]} ({results["metadata"]["model_params"]:,} params)',
                  fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Log-log scale (to see power law)
    ax2 = axes[1]
    ax2.errorbar(sizes, means, yerr=stds, fmt='o-', capsize=5,
                 color='#E94F37', markersize=8, linewidth=2)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Dataset Size (D) [log]', fontsize=12)
    ax2.set_ylabel('Test Loss (MSE) [log]', fontsize=12)
    ax2.set_title('Log-Log Scale (Power Law Check)', fontsize=14)
    ax2.grid(True, alpha=0.3, which='both')
    
    # Fit power law: L = A * D^(-alpha)
    log_sizes = np.log(sizes)
    log_means = np.log(means)
    
    # Simple linear regression in log space
    slope, intercept = np.polyfit(log_sizes, log_means, 1)
    alpha = -slope  # power law exponent
    A = np.exp(intercept)
    
    # Plot fit line
    fit_sizes = np.linspace(min(sizes), max(sizes), 100)
    fit_loss = A * fit_sizes ** (-alpha)
    ax2.plot(fit_sizes, fit_loss, '--', color='gray', alpha=0.7, 
             label=f'Fit: L ∝ D^{-alpha:.3f}')
    ax2.legend()
    
    # Add R² value
    from scipy.stats import pearsonr
    r, _ = pearsonr(log_sizes, log_means)
    r_squared = r ** 2
    ax2.text(0.05, 0.95, f'R² = {r_squared:.3f}\nα = {alpha:.3f}', 
             transform=ax2.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return alpha, r_squared


def print_summary(results: Dict[str, Any]):
    """Print a summary table."""
    print("\n" + "=" * 60)
    print("D-SCALING RESULTS SUMMARY")
    print("=" * 60)
    
    size_results = results['size_results']
    
    # Header
    print(f"{'Size':<10} {'Loss Mean':<12} {'Loss Std':<12}")
    print("-" * 34)
    
    for d_size in sorted([int(k) for k in size_results.keys()]):
        data = size_results[d_size]
        print(f"{d_size:<10} {data['loss_mean']:<12.6f} {data['loss_std']:<12.6f}")
    
    print("-" * 34)
    print(f"Total time: {results['total_time']:.1f}s")


def main():
    # Configuration
    config = DScalingConfig(
        game="leduc",
        model_size="m",  # Fixed at stable regime
        cfr_iterations=1000,
        dataset_sizes=(100, 250, 500, 1000, 2000, 5000),
        seeds=(42, 123, 456),
        epochs=200,
        batch_size=32,
        learning_rate=1e-3,
        early_stopping_patience=30,
    )
    
    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run experiment
    results = run_experiment(config)
    
    # Print summary
    print_summary(results)
    
    # Plot
    plot_path = output_dir / f"scaling_D_{config.game}.png"
    try:
        alpha, r_squared = plot_results(results, plot_path)
        print(f"\nPower law fit: L ∝ D^{-alpha:.3f} (R² = {r_squared:.3f})")
        print(f"Plot saved to: {plot_path}")
    except ImportError:
        print("matplotlib not available for plotting")
    
    # Save results
    json_path = output_dir / f"scaling_D_{config.game}.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to: {json_path}")


if __name__ == "__main__":
    main()

