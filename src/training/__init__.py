"""Training infrastructure for CFV prediction."""

from .losses import CFVLoss, cfv_mse_loss, cfv_huber_loss, strategy_kl_loss
from .trainer import Trainer, TrainingConfig, TrainingResult, train_model
from .self_play import SelfPlayTrainer, SelfPlayConfig, SelfPlayBuffer
from .deep_cfr import DeepCFRTrainer, DeepCFRConfig, ReservoirBuffer

__all__ = [
    # Losses
    'CFVLoss',
    'cfv_mse_loss',
    'cfv_huber_loss',
    'strategy_kl_loss',
    # Trainer
    'Trainer',
    'TrainingConfig',
    'TrainingResult',
    'train_model',
    # Self-play
    'SelfPlayTrainer',
    'SelfPlayConfig',
    'SelfPlayBuffer',
    # Deep CFR
    'DeepCFRTrainer',
    'DeepCFRConfig',
    'ReservoirBuffer',
]
