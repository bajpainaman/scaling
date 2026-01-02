"""
Base class for CFV prediction models.

All models predict counterfactual values (CFV) from encoded infosets.
Strategy is derived via regret matching, not a separate head.
"""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class CFVPredictor(nn.Module, ABC):
    """
    Base class for counterfactual value prediction models.
    
    Input: Encoded infoset [batch, input_dim]
    Output: Predicted CFVs [batch, num_actions]
    
    Strategy is derived from CFV via regret matching (not learned separately).
    """
    
    def __init__(self, input_dim: int, num_actions: int):
        super().__init__()
        self.input_dim = input_dim
        self.num_actions = num_actions
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict counterfactual values.
        
        Args:
            x: Encoded infoset [batch, input_dim]
        
        Returns:
            cfv: Predicted CFVs [batch, num_actions]
        """
        pass
    
    def predict_strategy(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert CFV predictions to strategy via regret matching.
        
        This mirrors how strategies are derived from CFV in CFR:
        - Compute "regrets" as CFV - mean(CFV)
        - Take positive part
        - Normalize to probability distribution
        
        Args:
            x: Encoded infoset [batch, input_dim]
        
        Returns:
            strategy: Action probabilities [batch, num_actions]
        """
        cfv = self.forward(x)
        return self.cfv_to_strategy(cfv)
    
    @staticmethod
    def cfv_to_strategy(cfv: torch.Tensor) -> torch.Tensor:
        """
        Convert CFV to strategy via regret matching.
        
        Args:
            cfv: Counterfactual values [batch, num_actions]
        
        Returns:
            strategy: Action probabilities [batch, num_actions]
        """
        # Regrets relative to mean action value
        regrets = cfv - cfv.mean(dim=-1, keepdim=True)
        
        # Positive regrets only
        positive_regrets = torch.clamp(regrets, min=0)
        
        # Normalize (uniform if all regrets non-positive)
        total = positive_regrets.sum(dim=-1, keepdim=True)
        
        uniform = torch.ones_like(positive_regrets) / cfv.shape[-1]
        
        strategy = torch.where(
            total > 0,
            positive_regrets / (total + 1e-8),
            uniform,
        )
        
        return strategy
    
    def num_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def summary(self) -> str:
        """Return model summary string."""
        return (
            f"{self.__class__.__name__}("
            f"input_dim={self.input_dim}, "
            f"num_actions={self.num_actions}, "
            f"params={self.num_parameters():,})"
        )

