"""
Self-Play Training for Neural CFV Prediction

This module implements self-play fine-tuning:
1. Start with a pre-trained model (from oracle data)
2. Generate games using the model's policy
3. Compute regrets using CFR-style updates
4. Fine-tune model on the new regrets

This is a simplified version of Deep CFR / DREAM.
"""

from typing import Dict, List, Tuple, Optional, Callable
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from collections import defaultdict
from tqdm import tqdm

from ..games.base import Game, GameState, InfoSet
from ..models.base import CFVPredictor


@dataclass
class SelfPlayConfig:
    """Configuration for self-play training."""
    # Sampling
    games_per_epoch: int = 1000
    traversals_per_game: int = 2  # One per player
    
    # CFR updates
    use_cfr_plus: bool = True
    regret_temperature: float = 1.0  # For softmax sampling
    
    # Training
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    train_steps_per_epoch: int = 100
    
    # Regularization
    use_reservoir_sampling: bool = True
    reservoir_size: int = 100000
    
    # Exploration
    epsilon: float = 0.1  # Epsilon-greedy exploration


@dataclass
class SelfPlayBuffer:
    """Reservoir buffer for self-play data."""
    encodings: List[np.ndarray] = field(default_factory=list)
    targets: List[np.ndarray] = field(default_factory=list)
    weights: List[float] = field(default_factory=list)
    
    max_size: int = 100000
    count: int = 0
    
    def add(self, encoding: np.ndarray, target: np.ndarray, weight: float = 1.0):
        """Add sample with reservoir sampling."""
        self.count += 1
        
        if len(self.encodings) < self.max_size:
            self.encodings.append(encoding)
            self.targets.append(target)
            self.weights.append(weight)
        else:
            # Reservoir sampling
            idx = np.random.randint(0, self.count)
            if idx < self.max_size:
                self.encodings[idx] = encoding
                self.targets[idx] = target
                self.weights[idx] = weight
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample a batch."""
        if len(self.encodings) == 0:
            return None, None, None
        
        indices = np.random.choice(len(self.encodings), size=min(batch_size, len(self.encodings)))
        
        return (
            np.array([self.encodings[i] for i in indices]),
            np.array([self.targets[i] for i in indices]),
            np.array([self.weights[i] for i in indices]),
        )
    
    def __len__(self):
        return len(self.encodings)


class SelfPlayTrainer:
    """
    Self-play trainer for CFV prediction models.
    
    Uses External Sampling CFR with neural network for regret/strategy.
    """
    
    def __init__(
        self,
        game: Game,
        model: CFVPredictor,
        encode_infoset: Callable[[InfoSet, Game], np.ndarray],
        config: SelfPlayConfig,
        device: torch.device = None,
    ):
        self.game = game
        self.model = model
        self.encode_infoset = encode_infoset
        self.config = config
        self.device = device or torch.device("cpu")
        
        self.model.to(self.device)
        
        # Accumulated regrets (for CFR averaging)
        self.regret_sum: Dict[str, np.ndarray] = defaultdict(
            lambda: np.zeros(game.num_actions, dtype=np.float64)
        )
        self.strategy_sum: Dict[str, np.ndarray] = defaultdict(
            lambda: np.zeros(game.num_actions, dtype=np.float64)
        )
        
        # Replay buffer
        self.buffer = SelfPlayBuffer(max_size=config.reservoir_size)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        self._rng = np.random.default_rng()
        self.iteration = 0
    
    def get_strategy(self, cfv: np.ndarray, legal_actions: List[int]) -> np.ndarray:
        """
        Convert CFV to strategy via regret matching.
        
        This mirrors the CFR derivation.
        """
        # Regrets relative to mean
        mean_cfv = np.mean(cfv[legal_actions])
        regrets = cfv - mean_cfv
        
        # Positive regrets only
        positive = np.maximum(regrets, 0)
        
        # Normalize over legal actions
        strategy = np.zeros(len(cfv))
        legal_positive = positive[legal_actions]
        total = legal_positive.sum()
        
        if total > 0:
            strategy[legal_actions] = legal_positive / total
        else:
            strategy[legal_actions] = 1.0 / len(legal_actions)
        
        return strategy
    
    def predict_cfv(self, state: GameState, player: int) -> np.ndarray:
        """Get CFV prediction from model."""
        infoset = self.game.get_infoset(state, player)
        encoding = self.encode_infoset(infoset, self.game)
        
        x = torch.tensor(encoding, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            cfv = self.model(x).squeeze(0).cpu().numpy()
        
        return cfv
    
    def _sample_initial_state(self) -> GameState:
        """Sample initial state."""
        state = self.game.initial_state()
        
        while self.game.is_chance_node(state):
            outcomes = self.game.chance_outcomes(state)
            probs = np.array([p for _, p in outcomes])
            idx = self._rng.choice(len(outcomes), p=probs)
            state = outcomes[idx][0]
        
        return state
    
    def _external_cfr_traverse(
        self,
        state: GameState,
        traversing_player: int,
        pi_player: float,
        pi_opponent: float,
    ) -> float:
        """
        External sampling CFR traversal using neural network.
        
        Returns expected value for traversing player.
        """
        if self.game.is_terminal(state):
            payoffs = self.game.get_payoffs(state)
            return payoffs[traversing_player]
        
        if self.game.is_chance_node(state):
            outcomes = self.game.chance_outcomes(state)
            probs = np.array([p for _, p in outcomes])
            idx = self._rng.choice(len(outcomes), p=probs)
            return self._external_cfr_traverse(
                outcomes[idx][0], traversing_player, pi_player, pi_opponent
            )
        
        current_player = self.game.current_player(state)
        legal_actions = self.game.get_actions(state)
        
        # Get strategy from neural network
        cfv = self.predict_cfv(state, current_player)
        strategy = self.get_strategy(cfv, legal_actions)
        
        # Epsilon-greedy exploration
        if self._rng.random() < self.config.epsilon:
            strategy[legal_actions] = 1.0 / len(legal_actions)
        
        infoset = self.game.get_infoset(state, current_player)
        infoset_key = infoset.to_key()
        
        if current_player == traversing_player:
            # Traverse all actions
            action_values = np.zeros(self.game.num_actions)
            
            for action in legal_actions:
                next_state = self.game.apply_action(state, action)
                action_values[action] = self._external_cfr_traverse(
                    next_state, traversing_player,
                    pi_player * strategy[action], pi_opponent
                )
            
            # Expected value
            ev = np.dot(strategy, action_values)
            
            # Compute and store regrets
            regrets = action_values - ev
            
            # Update regret sum
            self.regret_sum[infoset_key] += pi_opponent * regrets
            if self.config.use_cfr_plus:
                self.regret_sum[infoset_key] = np.maximum(0, self.regret_sum[infoset_key])
            
            # Update strategy sum
            self.strategy_sum[infoset_key] += pi_player * strategy
            
            # Add to buffer: encoding → accumulated regrets (normalized)
            encoding = self.encode_infoset(infoset, self.game)
            target = self.regret_sum[infoset_key].copy()
            
            # Normalize target
            max_abs = np.max(np.abs(target)) + 1e-8
            target = target / max_abs
            
            self.buffer.add(encoding, target, weight=pi_opponent)
            
            return ev
        else:
            # Sample opponent action
            probs = strategy[legal_actions]
            probs = probs / (probs.sum() + 1e-8)  # Normalize
            action = self._rng.choice(legal_actions, p=probs)
            next_state = self.game.apply_action(state, action)
            
            return self._external_cfr_traverse(
                next_state, traversing_player,
                pi_player, pi_opponent * strategy[action]
            )
    
    def collect_epoch(self) -> int:
        """
        Run one epoch of self-play data collection.
        
        Returns number of samples collected.
        """
        samples_before = len(self.buffer)
        
        for _ in tqdm(range(self.config.games_per_epoch), desc="Self-play"):
            state = self._sample_initial_state()
            
            for player in range(self.game.num_players):
                self._external_cfr_traverse(state, player, 1.0, 1.0)
        
        self.iteration += 1
        return len(self.buffer) - samples_before
    
    def train_epoch(self) -> float:
        """
        Train model on collected data.
        
        Returns average loss.
        """
        if len(self.buffer) < self.config.batch_size:
            return 0.0
        
        self.model.train()
        total_loss = 0.0
        
        for _ in range(self.config.train_steps_per_epoch):
            encodings, targets, weights = self.buffer.sample(self.config.batch_size)
            
            x = torch.tensor(encodings, dtype=torch.float32).to(self.device)
            y = torch.tensor(targets, dtype=torch.float32).to(self.device)
            w = torch.tensor(weights, dtype=torch.float32).to(self.device)
            
            self.optimizer.zero_grad()
            pred = self.model(x)
            
            # Weighted MSE
            loss = (w.unsqueeze(1) * (pred - y) ** 2).mean()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / self.config.train_steps_per_epoch
    
    def train(
        self,
        num_epochs: int = 10,
        verbose: bool = True,
    ) -> List[float]:
        """
        Run full self-play training loop.
        
        Returns list of losses per epoch.
        """
        losses = []
        
        for epoch in range(num_epochs):
            # Collect data
            new_samples = self.collect_epoch()
            
            # Train
            loss = self.train_epoch()
            losses.append(loss)
            
            if verbose:
                print(f"Epoch {epoch+1}/{num_epochs}: "
                      f"samples={len(self.buffer)}, new={new_samples}, loss={loss:.6f}")
        
        return losses


if __name__ == "__main__":
    # Quick test with Leduc
    import sys
    sys.path.insert(0, '.')
    from src.games import LeducPoker
    from src.models import MLP
    from src.data.oracle_generator import encode_leduc_infoset
    
    game = LeducPoker()
    model = MLP.from_name("s", input_dim=40, num_actions=3)
    
    config = SelfPlayConfig(
        games_per_epoch=100,
        train_steps_per_epoch=50,
    )
    
    trainer = SelfPlayTrainer(
        game=game,
        model=model,
        encode_infoset=encode_leduc_infoset,
        config=config,
    )
    
    print("Running self-play training on Leduc...")
    losses = trainer.train(num_epochs=3, verbose=True)
    
    print(f"\nFinal buffer size: {len(trainer.buffer)}")
    print(f"Infosets with regrets: {len(trainer.regret_sum)}")

