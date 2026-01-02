"""
Deep CFR Implementation

Deep Counterfactual Regret Minimization:
- Uses neural networks to predict advantages (regrets)
- Separate advantage network and strategy network
- Reservoir sampling for memory management
- Linear CFR weighting for better convergence

Reference: Brown et al. "Deep Counterfactual Regret Minimization" (2019)
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass, field
from collections import defaultdict
import time
from tqdm import tqdm


@dataclass
class DeepCFRConfig:
    """Configuration for Deep CFR training."""
    # CFR iterations
    num_cfr_iterations: int = 100
    num_traversals_per_iter: int = 500
    
    # Network training
    train_steps_per_iter: int = 200
    batch_size: int = 256
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    
    # Reservoir sampling
    advantage_memory_size: int = 2_000_000
    strategy_memory_size: int = 2_000_000
    
    # Exploration
    epsilon: float = 0.1
    
    # Linear CFR weighting
    linear_cfr: bool = True
    
    # Device
    device: str = "mps"


class ReservoirBuffer:
    """
    Reservoir sampling buffer for Deep CFR.
    Maintains a fixed-size buffer with uniform sampling probability.
    """
    
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.buffer: List[Tuple[np.ndarray, np.ndarray, float]] = []
        self.total_seen = 0
    
    def add(self, encoding: np.ndarray, target: np.ndarray, weight: float = 1.0):
        """Add sample with reservoir sampling."""
        self.total_seen += 1
        
        if len(self.buffer) < self.max_size:
            self.buffer.append((encoding.copy(), target.copy(), weight))
        else:
            # Reservoir sampling: replace with probability max_size / total_seen
            idx = np.random.randint(0, self.total_seen)
            if idx < self.max_size:
                self.buffer[idx] = (encoding.copy(), target.copy(), weight)
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample a batch from the buffer."""
        if len(self.buffer) < batch_size:
            indices = np.arange(len(self.buffer))
        else:
            indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        
        encodings = np.array([self.buffer[i][0] for i in indices])
        targets = np.array([self.buffer[i][1] for i in indices])
        weights = np.array([self.buffer[i][2] for i in indices])
        
        return encodings, targets, weights
    
    def __len__(self) -> int:
        return len(self.buffer)
    
    def clear(self):
        """Clear the buffer."""
        self.buffer = []
        self.total_seen = 0


class AdvantageNetwork(nn.Module):
    """
    Network that predicts advantages (regrets) for each action.
    One network per player in 2-player games.
    """
    
    def __init__(self, input_dim: int, num_actions: int, hidden_dims: List[int] = [256, 256]):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.Dropout(0.1),
            ])
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, num_actions))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize last layer to zero for stable training
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class StrategyNetwork(nn.Module):
    """
    Network that predicts average strategy (action probabilities).
    Uses softmax output.
    """
    
    def __init__(self, input_dim: int, num_actions: int, hidden_dims: List[int] = [256, 256]):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.Dropout(0.1),
            ])
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, num_actions))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.network(x)
        return torch.softmax(logits, dim=-1)


class DeepCFRTrainer:
    """
    Deep CFR trainer for imperfect-information games.
    
    Training loop:
    1. For each CFR iteration t:
       a. Traverse game tree with external sampling
       b. Collect advantage samples (weighted by t for Linear CFR)
       c. Train advantage networks on collected samples
       d. Store strategy samples (weighted by reach probability)
    2. After all iterations, train strategy network on strategy samples
    """
    
    def __init__(
        self,
        game: Any,
        encode_state: Callable,
        input_dim: int,
        num_actions: int,
        config: DeepCFRConfig = None,
    ):
        self.game = game
        self.encode_state = encode_state
        self.input_dim = input_dim
        self.num_actions = num_actions
        self.config = config or DeepCFRConfig()
        
        self.device = torch.device(self.config.device if torch.backends.mps.is_available() else "cpu")
        
        # Advantage networks (one per player)
        self.advantage_nets = [
            AdvantageNetwork(input_dim, num_actions).to(self.device)
            for _ in range(2)
        ]
        self.advantage_optimizers = [
            torch.optim.Adam(net.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
            for net in self.advantage_nets
        ]
        
        # Strategy network
        self.strategy_net = StrategyNetwork(input_dim, num_actions).to(self.device)
        self.strategy_optimizer = torch.optim.Adam(
            self.strategy_net.parameters(), 
            lr=self.config.learning_rate, 
            weight_decay=self.config.weight_decay
        )
        
        # Memory buffers (one advantage buffer per player)
        self.advantage_buffers = [
            ReservoirBuffer(self.config.advantage_memory_size)
            for _ in range(2)
        ]
        self.strategy_buffer = ReservoirBuffer(self.config.strategy_memory_size)
        
        # RNG
        self.rng = np.random.default_rng()
    
    def get_strategy_from_advantages(self, advantages: np.ndarray, legal_actions: np.ndarray) -> np.ndarray:
        """Convert advantages to strategy via regret matching."""
        strategy = np.zeros(self.num_actions)
        positive = np.maximum(advantages, 0)
        legal_positive = positive[legal_actions]
        total = legal_positive.sum()
        
        if total > 0:
            strategy[legal_actions] = legal_positive / total
        else:
            strategy[legal_actions] = 1.0 / len(legal_actions)
        
        return strategy
    
    def predict_advantages(self, encoding: np.ndarray, player: int) -> np.ndarray:
        """Get advantages from network."""
        x = torch.tensor(encoding, dtype=torch.float32).unsqueeze(0).to(self.device)
        self.advantage_nets[player].eval()
        with torch.no_grad():
            return self.advantage_nets[player](x).squeeze(0).cpu().numpy()
    
    def external_cfr_traversal(
        self, 
        state: Any, 
        traverser: int, 
        iteration: int,
        pi_t: float = 1.0,
        pi_o: float = 1.0,
    ) -> float:
        """
        External sampling CFR traversal.
        
        Args:
            state: Current game state
            traverser: Player whose strategy we're updating
            iteration: Current CFR iteration (for Linear CFR weighting)
            pi_t: Traverser's reach probability
            pi_o: Opponent's reach probability
        
        Returns:
            Expected value for traverser
        """
        if state.is_terminal:
            return self.game.get_payoff(state, traverser)
        
        player = state.current_player
        legal = self.game.get_legal_actions(state)
        
        # Encode state
        encoding = self.encode_state(state, player)
        
        # Get advantages from network
        advantages = self.predict_advantages(encoding, player)
        strategy = self.get_strategy_from_advantages(advantages, legal)
        
        # Epsilon-greedy exploration
        if self.rng.random() < self.config.epsilon:
            strategy[legal] = 1.0 / len(legal)
        
        if player == traverser:
            # Traverse all actions for traverser
            action_values = np.zeros(self.num_actions)
            for action in legal:
                next_state = self.game._apply_nlhe_action(state, action)
                action_values[action] = self.external_cfr_traversal(
                    next_state, traverser, iteration,
                    pi_t * strategy[action], pi_o
                )
            
            # Compute counterfactual values
            ev = np.dot(strategy, action_values)
            cfv = action_values - ev  # Advantages
            
            # Add to advantage buffer with Linear CFR weighting
            weight = iteration if self.config.linear_cfr else 1.0
            self.advantage_buffers[player].add(encoding, cfv, weight)
            
            # Add to strategy buffer (weighted by reach probability)
            self.strategy_buffer.add(encoding, strategy, pi_o)
            
            return ev
        else:
            # Sample opponent's action
            probs = strategy[legal]
            probs = probs / (probs.sum() + 1e-8)
            action = self.rng.choice(legal, p=probs)
            
            next_state = self.game._apply_nlhe_action(state, action)
            return self.external_cfr_traversal(
                next_state, traverser, iteration,
                pi_t, pi_o * strategy[action]
            )
    
    def train_advantage_network(self, player: int) -> float:
        """Train advantage network for one player."""
        buffer = self.advantage_buffers[player]
        if len(buffer) < self.config.batch_size:
            return 0.0
        
        self.advantage_nets[player].train()
        
        total_loss = 0.0
        for _ in range(self.config.train_steps_per_iter):
            encodings, targets, weights = buffer.sample(self.config.batch_size)
            
            X = torch.tensor(encodings, dtype=torch.float32).to(self.device)
            Y = torch.tensor(targets, dtype=torch.float32).to(self.device)
            W = torch.tensor(weights, dtype=torch.float32).to(self.device)
            W = W / W.sum()  # Normalize weights
            
            self.advantage_optimizers[player].zero_grad()
            pred = self.advantage_nets[player](X)
            
            # Weighted MSE loss
            loss = (W.unsqueeze(1) * (pred - Y) ** 2).sum()
            loss.backward()
            self.advantage_optimizers[player].step()
            
            total_loss += loss.item()
        
        return total_loss / self.config.train_steps_per_iter
    
    def train_strategy_network(self) -> float:
        """Train strategy network on accumulated samples."""
        if len(self.strategy_buffer) < self.config.batch_size:
            return 0.0
        
        self.strategy_net.train()
        
        total_loss = 0.0
        num_steps = self.config.train_steps_per_iter * 2  # More training for strategy
        
        for _ in range(num_steps):
            encodings, targets, weights = self.strategy_buffer.sample(self.config.batch_size)
            
            X = torch.tensor(encodings, dtype=torch.float32).to(self.device)
            Y = torch.tensor(targets, dtype=torch.float32).to(self.device)
            W = torch.tensor(weights, dtype=torch.float32).to(self.device)
            W = W / W.sum()
            
            self.strategy_optimizer.zero_grad()
            pred = self.strategy_net(X)
            
            # Cross-entropy style loss (KL divergence approximation)
            loss = -(W.unsqueeze(1) * Y * torch.log(pred + 1e-8)).sum()
            loss.backward()
            self.strategy_optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / num_steps
    
    def train(self, verbose: bool = True) -> Dict[str, List[float]]:
        """
        Run full Deep CFR training.
        
        Returns:
            Dictionary of training metrics
        """
        metrics = {
            "adv_loss_p0": [],
            "adv_loss_p1": [],
            "strategy_loss": [],
            "buffer_sizes": [],
        }
        
        print(f"\n{'='*60}")
        print(f"🎯 DEEP CFR TRAINING")
        print(f"{'='*60}")
        print(f"Iterations: {self.config.num_cfr_iterations}")
        print(f"Traversals/iter: {self.config.num_traversals_per_iter}")
        print(f"Device: {self.device}")
        
        start_time = time.time()
        
        for iteration in tqdm(range(1, self.config.num_cfr_iterations + 1), desc="CFR Iterations"):
            # Collect samples via external sampling
            for _ in range(self.config.num_traversals_per_iter):
                state = self.game.get_initial_state(self.rng)
                
                # Traverse for each player
                for player in range(2):
                    self.external_cfr_traversal(state, player, iteration)
            
            # Train advantage networks
            loss_p0 = self.train_advantage_network(0)
            loss_p1 = self.train_advantage_network(1)
            
            metrics["adv_loss_p0"].append(loss_p0)
            metrics["adv_loss_p1"].append(loss_p1)
            metrics["buffer_sizes"].append((
                len(self.advantage_buffers[0]),
                len(self.advantage_buffers[1]),
                len(self.strategy_buffer),
            ))
            
            if verbose and iteration % 10 == 0:
                elapsed = time.time() - start_time
                print(f"\n  Iter {iteration}: adv_loss=[{loss_p0:.4f}, {loss_p1:.4f}]")
                print(f"    Buffers: adv0={len(self.advantage_buffers[0])}, "
                      f"adv1={len(self.advantage_buffers[1])}, "
                      f"strat={len(self.strategy_buffer)}")
                print(f"    Time: {elapsed:.1f}s")
        
        # Final strategy network training
        print("\n📊 Training final strategy network...")
        strategy_loss = self.train_strategy_network()
        metrics["strategy_loss"].append(strategy_loss)
        
        total_time = time.time() - start_time
        print(f"\n✅ Deep CFR complete in {total_time:.1f}s")
        print(f"Final strategy loss: {strategy_loss:.4f}")
        
        return metrics
    
    def get_strategy(self, encoding: np.ndarray) -> np.ndarray:
        """Get strategy from trained strategy network."""
        x = torch.tensor(encoding, dtype=torch.float32).unsqueeze(0).to(self.device)
        self.strategy_net.eval()
        with torch.no_grad():
            return self.strategy_net(x).squeeze(0).cpu().numpy()


if __name__ == "__main__":
    # Quick test with Leduc
    import sys
    sys.path.insert(0, '.')
    
    from src.games import LeducPoker
    from src.data.oracle_generator import encode_leduc_infoset
    
    game = LeducPoker()
    
    def encode_state(state, player):
        infoset = game.get_infoset(state, player)
        return encode_leduc_infoset(infoset, game)
    
    config = DeepCFRConfig(
        num_cfr_iterations=20,
        num_traversals_per_iter=100,
        train_steps_per_iter=50,
    )
    
    trainer = DeepCFRTrainer(
        game=game,
        encode_state=encode_state,
        input_dim=32,  # Leduc encoding dim
        num_actions=3,
        config=config,
    )
    
    print("Testing Deep CFR on Leduc...")
    
    # Adapt for Leduc's apply_action
    original_apply = trainer.game._apply_nlhe_action
    trainer.game._apply_nlhe_action = lambda s, a: game.apply_action(s, a)
    
    metrics = trainer.train(verbose=True)
    
    print("\n✅ Deep CFR test complete!")

