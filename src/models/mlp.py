"""
MLP model family for CFV prediction.

Phase 3A baseline: Plain MLPs with clean parameter scaling.
No fancy tricks - this is the control experiment.
"""

from typing import List, Optional
import torch
import torch.nn as nn

from .base import CFVPredictor
from .configs import MLPConfig, get_config


class MLP(CFVPredictor):
    """
    Multi-layer perceptron for CFV prediction.
    
    Simple feedforward network:
    Input → Linear → ReLU → ... → Linear → Output
    
    Designed for clean scaling experiments:
    - No skip connections
    - No layer norm (can add later as ablation)
    - Optional dropout for regularization
    """
    
    def __init__(
        self,
        input_dim: int,
        num_actions: int,
        hidden_dims: List[int],
        dropout: float = 0.0,
        activation: str = "relu",
    ):
        """
        Args:
            input_dim: Dimension of input encoding
            num_actions: Number of output actions (CFV per action)
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability (0 = no dropout)
            activation: Activation function ("relu", "gelu", "tanh")
        """
        super().__init__(input_dim, num_actions)
        
        self.hidden_dims = hidden_dims
        self.dropout_p = dropout
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(self._get_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer (no activation)
        layers.append(nn.Linear(prev_dim, num_actions))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _get_activation(self, name: str) -> nn.Module:
        """Get activation module by name."""
        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "tanh": nn.Tanh(),
            "leaky_relu": nn.LeakyReLU(0.1),
        }
        if name not in activations:
            raise ValueError(f"Unknown activation: {name}")
        return activations[name]
    
    def _init_weights(self):
        """Initialize weights with Xavier/Glorot initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input encoding [batch, input_dim]
        
        Returns:
            cfv: Predicted counterfactual values [batch, num_actions]
        """
        return self.network(x)
    
    @classmethod
    def from_config(
        cls,
        config: MLPConfig,
        input_dim: int,
        num_actions: int,
    ) -> "MLP":
        """
        Create MLP from config.
        
        Args:
            config: MLPConfig instance
            input_dim: Input dimension (game-specific)
            num_actions: Number of actions (game-specific)
        """
        return cls(
            input_dim=input_dim,
            num_actions=num_actions,
            hidden_dims=config.hidden_dims,
            dropout=config.dropout,
            activation=config.activation,
        )
    
    @classmethod
    def from_name(
        cls,
        name: str,
        input_dim: int,
        num_actions: int,
    ) -> "MLP":
        """
        Create MLP from config name.
        
        Args:
            name: Config name ("tiny", "xs", "s", "m", "l", "xl", "xxl")
            input_dim: Input dimension
            num_actions: Number of actions
        """
        config = get_config(name)
        return cls.from_config(config, input_dim, num_actions)


def create_model(
    name: str,
    input_dim: int,
    num_actions: int,
) -> CFVPredictor:
    """
    Factory function to create a model by name.
    
    Args:
        name: Model config name
        input_dim: Input dimension
        num_actions: Number of actions
    
    Returns:
        CFVPredictor instance
    """
    return MLP.from_name(name, input_dim, num_actions)


def print_model_sizes(input_dim: int = 32, num_actions: int = 3):
    """Print parameter counts for all model sizes."""
    from .configs import MODEL_CONFIGS
    
    print(f"Model sizes (input_dim={input_dim}, num_actions={num_actions}):")
    print("-" * 50)
    
    for name, config in MODEL_CONFIGS.items():
        model = MLP.from_config(config, input_dim, num_actions)
        params = model.num_parameters()
        print(f"  {name:6s}: {params:>10,} params | layers={config.hidden_dims}")

