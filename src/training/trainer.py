"""
Training loop for CFV prediction models.

Simple and clean for Phase 3A:
- AdamW optimizer
- Cosine LR schedule
- Early stopping
- Basic logging
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Callable
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from ..models.base import CFVPredictor
from .losses import CFVLoss


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Model
    model_size: str = "s"
    
    # Optimization
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    epochs: int = 100
    
    # Loss
    loss_type: str = "mse"  # "mse" or "huber"
    use_reach_weighting: bool = True
    strategy_weight: float = 0.0  # Auxiliary strategy loss weight
    
    # Early stopping
    early_stopping_patience: int = 10
    
    # Logging
    eval_every: int = 1  # Evaluate every N epochs
    verbose: bool = True
    
    # Reproducibility
    seed: int = 42
    
    # Device
    device: str = "cpu"  # "cpu" or "cuda" or "mps"


@dataclass
class TrainingResult:
    """Training results."""
    final_train_loss: float
    final_val_loss: float
    best_val_loss: float
    best_epoch: int
    total_epochs: int
    training_time_seconds: float
    history: Dict[str, List[float]] = field(default_factory=dict)


class Trainer:
    """
    Trainer for CFV prediction models.
    
    Simple training loop with:
    - AdamW optimizer
    - Cosine LR schedule
    - Early stopping
    - Loss logging
    """
    
    def __init__(
        self,
        model: CFVPredictor,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainingConfig,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # Move model to device
        self.device = torch.device(config.device)
        self.model = self.model.to(self.device)
        
        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        # LR scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.epochs,
        )
        
        # Loss function
        self.loss_fn = CFVLoss(
            loss_type=config.loss_type,
            use_reach_weighting=config.use_reach_weighting,
            strategy_weight=config.strategy_weight,
        )
        
        # Tracking
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_cfv_mae': [],
            'lr': [],
        }
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.patience_counter = 0
        self.best_state_dict = None
    
    def train(self) -> TrainingResult:
        """
        Run training loop.
        
        Returns:
            TrainingResult with final metrics and history
        """
        start_time = time.time()
        
        iterator = range(self.config.epochs)
        if self.config.verbose:
            iterator = tqdm(iterator, desc="Training")
        
        for epoch in iterator:
            # Train one epoch
            train_loss = self._train_epoch()
            
            # Evaluate
            if epoch % self.config.eval_every == 0 or epoch == self.config.epochs - 1:
                val_metrics = self._evaluate()
                val_loss = val_metrics['loss']
                
                # Track history
                self.history['train_loss'].append(train_loss)
                self.history['val_loss'].append(val_loss)
                self.history['val_cfv_mae'].append(val_metrics['cfv_mae'])
                self.history['lr'].append(self.scheduler.get_last_lr()[0])
                
                # Update progress bar
                if self.config.verbose:
                    iterator.set_postfix({
                        'train': f'{train_loss:.4f}',
                        'val': f'{val_loss:.4f}',
                    })
                
                # Early stopping check
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_epoch = epoch
                    self.patience_counter = 0
                    self.best_state_dict = {
                        k: v.cpu().clone() for k, v in self.model.state_dict().items()
                    }
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.config.early_stopping_patience:
                        if self.config.verbose:
                            print(f"\nEarly stopping at epoch {epoch}")
                        break
            
            # Step scheduler
            self.scheduler.step()
        
        training_time = time.time() - start_time
        
        # Restore best model
        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)
        
        return TrainingResult(
            final_train_loss=self.history['train_loss'][-1] if self.history['train_loss'] else 0,
            final_val_loss=self.history['val_loss'][-1] if self.history['val_loss'] else 0,
            best_val_loss=self.best_val_loss,
            best_epoch=self.best_epoch,
            total_epochs=len(self.history['train_loss']),
            training_time_seconds=training_time,
            history=self.history,
        )
    
    def _train_epoch(self) -> float:
        """Train for one epoch. Returns average loss."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in self.train_loader:
            # Move to device
            x = batch['encoding'].to(self.device)
            target_cfv = batch['cfv'].to(self.device)
            target_strategy = batch['strategy'].to(self.device)
            reach_prob = batch['reach_prob'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            pred_cfv = self.model(x)
            
            # Compute loss
            loss_dict = self.loss_fn(
                pred_cfv=pred_cfv,
                target_cfv=target_cfv,
                target_strategy=target_strategy,
                reach_prob=reach_prob,
            )
            loss = loss_dict['loss']
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / max(num_batches, 1)
    
    def _evaluate(self) -> Dict[str, float]:
        """Evaluate on validation set. Returns metrics dict."""
        self.model.eval()
        total_loss = 0.0
        total_mae = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                x = batch['encoding'].to(self.device)
                target_cfv = batch['cfv'].to(self.device)
                target_strategy = batch['strategy'].to(self.device)
                reach_prob = batch['reach_prob'].to(self.device)
                
                pred_cfv = self.model(x)
                
                loss_dict = self.loss_fn(
                    pred_cfv=pred_cfv,
                    target_cfv=target_cfv,
                    target_strategy=target_strategy,
                    reach_prob=reach_prob,
                )
                
                total_loss += loss_dict['loss'].item()
                total_mae += torch.abs(pred_cfv - target_cfv).mean().item()
                num_batches += 1
        
        n = max(num_batches, 1)
        return {
            'loss': total_loss / n,
            'cfv_mae': total_mae / n,
        }
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch,
            'history': self.history,
        }, path)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_epoch = checkpoint['best_epoch']
        self.history = checkpoint['history']


def train_model(
    model: CFVPredictor,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainingConfig = None,
    **kwargs,
) -> TrainingResult:
    """
    Convenience function to train a model.
    
    Args:
        model: CFVPredictor to train
        train_loader: Training data
        val_loader: Validation data
        config: TrainingConfig (or pass kwargs)
        **kwargs: Override config values
    
    Returns:
        TrainingResult
    """
    if config is None:
        config = TrainingConfig(**kwargs)
    else:
        # Override with kwargs
        for k, v in kwargs.items():
            if hasattr(config, k):
                setattr(config, k, v)
    
    trainer = Trainer(model, train_loader, val_loader, config)
    return trainer.train()

