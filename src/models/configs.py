"""
Model size configurations for scaling experiments.

Phase 3A: MLP family with clean parameter scaling.
Start simple, measure scaling, then upgrade architecture.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class MLPConfig:
    """Configuration for MLP model."""
    name: str
    hidden_dims: List[int]
    dropout: float = 0.0
    activation: str = "relu"
    
    @property
    def num_layers(self) -> int:
        return len(self.hidden_dims)
    
    def compute_params(self, input_dim: int, output_dim: int) -> int:
        """Compute approximate parameter count."""
        total = 0
        prev_dim = input_dim
        for hidden_dim in self.hidden_dims:
            total += prev_dim * hidden_dim + hidden_dim  # weights + bias
            prev_dim = hidden_dim
        total += prev_dim * output_dim + output_dim  # output layer
        return total


# ============================================================================
# MLP Model Configs - Clean scaling ladder
# ============================================================================
# 
# Design principles:
# 1. Parameter count scales ~4x between sizes
# 2. Depth increases with width (prevents bottlenecks)
# 3. No fancy tricks (LayerNorm, skip connections) for baseline
#
# These are the "control experiment" models for measuring scaling laws.

MODEL_CONFIGS = {
    # Debugging (very small)
    "tiny": MLPConfig(
        name="tiny",
        hidden_dims=[32, 16],
        # ~1K params for Leduc (32→32→16→3)
    ),
    
    # Scaling ladder
    "xs": MLPConfig(
        name="xs", 
        hidden_dims=[64, 32],
        # ~4K params
    ),
    
    "s": MLPConfig(
        name="s",
        hidden_dims=[128, 64],
        # ~15K params
    ),
    
    "m": MLPConfig(
        name="m",
        hidden_dims=[256, 128, 64],
        # ~60K params
    ),
    
    "l": MLPConfig(
        name="l",
        hidden_dims=[512, 256, 128],
        # ~250K params
    ),
    
    "xl": MLPConfig(
        name="xl",
        hidden_dims=[1024, 512, 256, 128],
        # ~1M params
    ),
    
    "xxl": MLPConfig(
        name="xxl",
        hidden_dims=[2048, 1024, 512, 256],
        # ~4M params
    ),
}


def get_config(name: str) -> MLPConfig:
    """Get model config by name."""
    if name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {name}. Available: {list(MODEL_CONFIGS.keys())}")
    return MODEL_CONFIGS[name]


def list_configs() -> List[str]:
    """List all available config names."""
    return list(MODEL_CONFIGS.keys())


# ============================================================================
# Dataset sizes for scaling D experiments
# ============================================================================

DATASET_SIZES = [100, 500, 1_000, 5_000, 10_000, 50_000, 100_000]


# ============================================================================
# Search budgets for scaling C experiments (Phase 4)
# ============================================================================

SEARCH_BUDGETS = [0, 10, 50, 100, 500, 1000]

