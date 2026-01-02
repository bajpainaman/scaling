#!/usr/bin/env python3
"""
Scaling N Experiment: Model capacity vs CFV prediction loss.

The smallest meaningful experiment:
- 1 game (Leduc)
- All model sizes (tiny → xxl)
- 3 seeds per size
- Log-log plot of params vs loss

Expected output: Power law L(N) ≈ A/N^α + E
"""

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

import torch
from tqdm import tqdm

from src.data import generate_oracle_data, CFVDataset, create_dataloaders
from src.models import MLP, list_configs, get_config
from src.training import Trainer, TrainingConfig, TrainingResult


@dataclass
class ExperimentConfig:
    """Configuration for scaling N experiment."""
    # Game
    game: str = "leduc"
    
    # Oracle quality (higher = closer to Nash)
    cfr_iterations: int = 1000
    
    # Model sizes to test
    model_sizes: List[str] = None
    
    # Seeds for error bars
    seeds: List[int] = None
    
    # Training
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    early_stopping_patience: int = 15
    
    # Output
    output_dir: str = "results/scaling_N"
    
    def __post_init__(self):
        if self.model_sizes is None:
            self.model_sizes = ["tiny", "xs", "s", "m", "l", "xl"]
        if self.seeds is None:
            self.seeds = [42, 123, 456]


@dataclass
class RunResult:
    """Result from a single training run."""
    model_size: str
    seed: int
    num_params: int
    best_val_loss: float
    test_loss: float
    test_mae: float
    best_epoch: int
    training_time: float


def run_single(
    model_size: str,
    seed: int,
    train_loader,
    val_loader,
    test_loader,
    input_dim: int,
    num_actions: int,
    config: ExperimentConfig,
) -> RunResult:
    """Run a single training configuration."""
    
    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create model
    model = MLP.from_name(model_size, input_dim, num_actions)
    num_params = model.num_parameters()
    
    # Train
    train_config = TrainingConfig(
        epochs=config.epochs,
        learning_rate=config.learning_rate,
        early_stopping_patience=config.early_stopping_patience,
        verbose=False,
    )
    
    trainer = Trainer(model, train_loader, val_loader, train_config)
    result = trainer.train()
    
    # Evaluate on test set
    model.eval()
    test_loss = 0.0
    test_mae = 0.0
    n = 0
    
    with torch.no_grad():
        for batch in test_loader:
            x = batch['encoding']
            y = batch['cfv']
            pred = model(x)
            test_loss += ((pred - y) ** 2).mean().item()
            test_mae += (pred - y).abs().mean().item()
            n += 1
    
    return RunResult(
        model_size=model_size,
        seed=seed,
        num_params=num_params,
        best_val_loss=result.best_val_loss,
        test_loss=test_loss / n,
        test_mae=test_mae / n,
        best_epoch=result.best_epoch,
        training_time=result.training_time_seconds,
    )


def run_experiment(config: ExperimentConfig = None) -> Dict[str, Any]:
    """
    Run the full scaling N experiment.
    
    Returns:
        Dictionary with all results and metadata
    """
    if config is None:
        config = ExperimentConfig()
    
    print("=" * 60)
    print("SCALING N EXPERIMENT")
    print("=" * 60)
    print(f"Game: {config.game}")
    print(f"Model sizes: {config.model_sizes}")
    print(f"Seeds: {config.seeds}")
    print(f"CFR iterations: {config.cfr_iterations}")
    print()
    
    # Generate oracle data (once)
    print("Generating oracle data...")
    datapoints, metadata = generate_oracle_data(
        game=config.game,
        cfr_iterations=config.cfr_iterations,
        solver_type="cfr+",
        seed=config.seeds[0],
    )
    print(f"  Dataset size: {len(datapoints)}")
    print(f"  Exploitability: {metadata.final_exploitability:.6f}")
    print()
    
    # Create dataset
    dataset = CFVDataset(datapoints, normalize_cfv=True)
    cfv_scale = dataset.get_cfv_scale()
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset,
        batch_size=config.batch_size,
        seed=config.seeds[0],
    )
    
    input_dim = metadata.input_dim
    num_actions = metadata.num_actions
    
    # Run all configurations
    results: List[RunResult] = []
    total_runs = len(config.model_sizes) * len(config.seeds)
    
    print(f"Running {total_runs} training runs...")
    print()
    
    with tqdm(total=total_runs, desc="Training") as pbar:
        for model_size in config.model_sizes:
            for seed in config.seeds:
                result = run_single(
                    model_size=model_size,
                    seed=seed,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    test_loader=test_loader,
                    input_dim=input_dim,
                    num_actions=num_actions,
                    config=config,
                )
                results.append(result)
                pbar.update(1)
                pbar.set_postfix({
                    'size': model_size,
                    'loss': f'{result.test_loss:.4f}',
                })
    
    # Aggregate results by model size
    aggregated = {}
    for size in config.model_sizes:
        size_results = [r for r in results if r.model_size == size]
        losses = [r.test_loss for r in size_results]
        
        aggregated[size] = {
            'num_params': size_results[0].num_params,
            'test_loss_mean': np.mean(losses),
            'test_loss_std': np.std(losses),
            'test_loss_all': losses,
            'seeds': [r.seed for r in size_results],
        }
    
    # Print summary
    print()
    print("=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Size':<8} {'Params':>10} {'Loss Mean':>12} {'Loss Std':>10}")
    print("-" * 42)
    for size in config.model_sizes:
        agg = aggregated[size]
        print(f"{size:<8} {agg['num_params']:>10,} {agg['test_loss_mean']:>12.6f} {agg['test_loss_std']:>10.6f}")
    
    # Prepare output
    output = {
        'experiment': 'scaling_N',
        'config': asdict(config),
        'metadata': {
            'game': config.game,
            'dataset_size': len(datapoints),
            'exploitability': float(metadata.final_exploitability),
            'cfv_scale': float(cfv_scale),
            'input_dim': input_dim,
            'num_actions': num_actions,
        },
        'results': aggregated,
        'raw_results': [asdict(r) for r in results],
    }
    
    # Save results
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"scaling_N_{config.game}.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    print()
    print(f"Results saved to: {output_file}")
    
    return output


def plot_scaling_curve(results: Dict[str, Any], save_path: str = None):
    """
    Plot log-log scaling curve.
    
    X-axis: log(params)
    Y-axis: log(loss)
    Expected: linear relationship (power law)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plot")
        return
    
    aggregated = results['results']
    
    # Extract data
    sizes = list(aggregated.keys())
    params = [aggregated[s]['num_params'] for s in sizes]
    losses = [aggregated[s]['test_loss_mean'] for s in sizes]
    stds = [aggregated[s]['test_loss_std'] for s in sizes]
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # Plot with error bars
    ax.errorbar(params, losses, yerr=stds, fmt='o-', capsize=5, 
                markersize=8, linewidth=2, label='Test Loss')
    
    # Log-log scale
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Labels
    ax.set_xlabel('Number of Parameters (N)', fontsize=12)
    ax.set_ylabel('Test MSE Loss', fontsize=12)
    ax.set_title(f"CFV Prediction Scaling: {results['config']['game'].title()} Poker", fontsize=14)
    
    # Add model size labels
    for size, p, l in zip(sizes, params, losses):
        ax.annotate(size, (p, l), textcoords="offset points", 
                    xytext=(5, 5), fontsize=9)
    
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()


if __name__ == "__main__":
    # Run minimal experiment
    config = ExperimentConfig(
        game="leduc",
        cfr_iterations=1000,
        model_sizes=["tiny", "xs", "s", "m", "l", "xl"],
        seeds=[42, 123, 456],
        epochs=100,
    )
    
    results = run_experiment(config)
    
    # Plot
    plot_scaling_curve(
        results,
        save_path="results/scaling_N/scaling_N_leduc.png"
    )

