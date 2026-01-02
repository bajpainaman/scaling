"""
Loss functions for CFV prediction training.

Primary loss: MSE on counterfactual values (reach-weighted)
Keep it simple for Phase 3A.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def cfv_mse_loss(
    pred_cfv: torch.Tensor,
    target_cfv: torch.Tensor,
    weights: torch.Tensor = None,
) -> torch.Tensor:
    """
    Mean squared error on counterfactual values.
    
    This is the primary loss for scaling experiments.
    
    Args:
        pred_cfv: Predicted CFVs [batch, num_actions]
        target_cfv: Target CFVs from oracle [batch, num_actions]
        weights: Optional sample weights [batch] or [batch, 1]
    
    Returns:
        Scalar loss
    """
    mse = F.mse_loss(pred_cfv, target_cfv, reduction='none')
    
    if weights is not None:
        # Ensure weights broadcast correctly
        if weights.dim() == 1:
            weights = weights.unsqueeze(-1)
        mse = mse * weights
    
    return mse.mean()


def cfv_huber_loss(
    pred_cfv: torch.Tensor,
    target_cfv: torch.Tensor,
    delta: float = 1.0,
    weights: torch.Tensor = None,
) -> torch.Tensor:
    """
    Huber loss on CFVs. More robust to outliers.
    
    Args:
        pred_cfv: Predicted CFVs [batch, num_actions]
        target_cfv: Target CFVs [batch, num_actions]
        delta: Huber delta parameter
        weights: Optional sample weights
    
    Returns:
        Scalar loss
    """
    huber = F.huber_loss(pred_cfv, target_cfv, delta=delta, reduction='none')
    
    if weights is not None:
        if weights.dim() == 1:
            weights = weights.unsqueeze(-1)
        huber = huber * weights
    
    return huber.mean()


def strategy_kl_loss(
    pred_strategy: torch.Tensor,
    target_strategy: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    KL divergence between predicted and target strategies.
    
    Optional auxiliary loss - use sparingly in Phase 3A.
    
    Args:
        pred_strategy: Predicted strategy [batch, num_actions]
        target_strategy: Target strategy [batch, num_actions]
        eps: Small constant for numerical stability
    
    Returns:
        Scalar KL divergence
    """
    # Ensure valid distributions
    pred_strategy = pred_strategy + eps
    target_strategy = target_strategy + eps
    
    # Normalize
    pred_strategy = pred_strategy / pred_strategy.sum(dim=-1, keepdim=True)
    target_strategy = target_strategy / target_strategy.sum(dim=-1, keepdim=True)
    
    # KL(target || pred) = sum(target * log(target/pred))
    kl = target_strategy * (torch.log(target_strategy) - torch.log(pred_strategy))
    
    return kl.sum(dim=-1).mean()


class CFVLoss(nn.Module):
    """
    Combined loss module for CFV prediction.
    
    Primary: MSE on CFV (reach-weighted)
    Optional: KL on strategy (auxiliary)
    """
    
    def __init__(
        self,
        loss_type: str = "mse",
        use_reach_weighting: bool = True,
        strategy_weight: float = 0.0,  # 0 = no strategy loss
        huber_delta: float = 1.0,
    ):
        """
        Args:
            loss_type: "mse" or "huber"
            use_reach_weighting: Weight samples by reach probability
            strategy_weight: Weight for auxiliary strategy KL loss
            huber_delta: Delta for Huber loss
        """
        super().__init__()
        self.loss_type = loss_type
        self.use_reach_weighting = use_reach_weighting
        self.strategy_weight = strategy_weight
        self.huber_delta = huber_delta
    
    def forward(
        self,
        pred_cfv: torch.Tensor,
        target_cfv: torch.Tensor,
        target_strategy: torch.Tensor = None,
        reach_prob: torch.Tensor = None,
    ) -> dict:
        """
        Compute loss.
        
        Args:
            pred_cfv: Predicted CFVs [batch, num_actions]
            target_cfv: Target CFVs [batch, num_actions]
            target_strategy: Optional target strategy [batch, num_actions]
            reach_prob: Optional reach probabilities [batch]
        
        Returns:
            Dictionary with 'loss' (total) and component losses
        """
        # Compute weights
        weights = None
        if self.use_reach_weighting and reach_prob is not None:
            weights = reach_prob / (reach_prob.mean() + 1e-8)
        
        # Primary CFV loss
        if self.loss_type == "mse":
            cfv_loss = cfv_mse_loss(pred_cfv, target_cfv, weights)
        elif self.loss_type == "huber":
            cfv_loss = cfv_huber_loss(pred_cfv, target_cfv, self.huber_delta, weights)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        total_loss = cfv_loss
        result = {'cfv_loss': cfv_loss}
        
        # Optional strategy loss
        if self.strategy_weight > 0 and target_strategy is not None:
            from ..models.base import CFVPredictor
            pred_strategy = CFVPredictor.cfv_to_strategy(pred_cfv)
            strat_loss = strategy_kl_loss(pred_strategy, target_strategy)
            total_loss = total_loss + self.strategy_weight * strat_loss
            result['strategy_loss'] = strat_loss
        
        result['loss'] = total_loss
        return result

