"""Neural network models for CFV prediction."""

from .base import CFVPredictor
from .configs import (
    MLPConfig,
    MODEL_CONFIGS,
    get_config,
    list_configs,
    DATASET_SIZES,
    SEARCH_BUDGETS,
)
from .mlp import MLP, create_model, print_model_sizes
from .transformer import (
    SetTransformer,
    GroupedTransformer,
    TransformerConfig,
    TRANSFORMER_CONFIGS,
    print_transformer_sizes,
    compare_model_sizes,
)

__all__ = [
    # Base
    'CFVPredictor',
    # Configs
    'MLPConfig',
    'MODEL_CONFIGS',
    'get_config',
    'list_configs',
    'DATASET_SIZES',
    'SEARCH_BUDGETS',
    # MLP
    'MLP',
    'create_model',
    'print_model_sizes',
    # Transformer
    'SetTransformer',
    'GroupedTransformer',
    'TransformerConfig',
    'TRANSFORMER_CONFIGS',
    'print_transformer_sizes',
    'compare_model_sizes',
]
