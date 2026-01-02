# SIGMA-GO: Complete Build Context for Claude Code

**Purpose:** This document contains EVERYTHING needed to build the Sigma-Go scaling experiments. Feed this to Claude Code as context.

---

## 1. PROJECT OVERVIEW

### What We're Building
A research codebase to empirically test whether **counterfactual value (CFV) prediction** in imperfect-information games follows predictable **scaling laws** with respect to:
- **N**: Model capacity (parameters)
- **D**: Oracle data (CFR-generated labels)
- **C**: Online compute (inference-time search)

### Core Hypothesis
> Loss on CFV prediction scales as a power law: L(N,D) ≈ A/N^α + B/D^β + E
> 
> Online search (C) only helps when N > N_critical (competence threshold)

### Why This Matters
- If true → equilibrium reasoning can be "amortized" like language modeling
- Connects game theory to scaling laws literature (Kaplan, Chinchilla)
- Opens path to strategic reasoning without game-specific engineering

---

## 2. DIRECTORY STRUCTURE

```
sigma-go/
├── README.md
├── pyproject.toml              # Use uv or poetry
├── AGENTS.md                   # Research guidelines
│
├── src/
│   ├── __init__.py
│   │
│   ├── games/                  # Game implementations
│   │   ├── __init__.py
│   │   ├── base.py             # Abstract game interface
│   │   ├── kuhn.py             # Kuhn Poker (3 cards, simplest)
│   │   ├── leduc.py            # Leduc Poker (6 cards, main testbed)
│   │   └── liars_dice.py       # For generalization tests
│   │
│   ├── cfr/                    # Oracle solvers
│   │   ├── __init__.py
│   │   ├── vanilla_cfr.py      # Basic CFR implementation
│   │   ├── dcfr.py             # Discounted CFR (faster convergence)
│   │   ├── mccfr.py            # Monte Carlo CFR (sampling variant)
│   │   └── solver.py           # High-level solver interface
│   │
│   ├── data/                   # Dataset generation & loading
│   │   ├── __init__.py
│   │   ├── oracle_generator.py # Generate CFR-labeled data
│   │   ├── dataset.py          # PyTorch Dataset classes
│   │   └── schema.py           # Data format definitions
│   │
│   ├── models/                 # Neural network architectures
│   │   ├── __init__.py
│   │   ├── base.py             # Abstract model interface
│   │   ├── mlp.py              # MLP family (small/med/large)
│   │   ├── transformer.py      # Optional: transformer variant
│   │   └── configs.py          # Model size configurations
│   │
│   ├── training/               # Training infrastructure
│   │   ├── __init__.py
│   │   ├── trainer.py          # Main training loop
│   │   ├── losses.py           # CFV loss, regret loss, etc.
│   │   └── callbacks.py        # Logging, checkpointing
│   │
│   ├── evaluation/             # Metrics & evaluation
│   │   ├── __init__.py
│   │   ├── cfv_metrics.py      # CFV prediction accuracy
│   │   ├── exploitability.py   # Full exploitability calculation
│   │   ├── regret.py           # Regret-based metrics
│   │   └── search.py           # Online search evaluation
│   │
│   └── utils/                  # Utilities
│       ├── __init__.py
│       ├── infoset.py          # Information set encoding
│       ├── belief.py           # Belief state representations
│       └── visualization.py    # Plotting utilities
│
├── experiments/                # Experiment configs & scripts
│   ├── configs/
│   │   ├── scaling_N.yaml      # Vary model size
│   │   ├── scaling_D.yaml      # Vary dataset size
│   │   ├── scaling_C.yaml      # Vary search compute
│   │   └── threshold.yaml      # Search threshold experiments
│   │
│   ├── run_scaling.py          # Main experiment runner
│   ├── run_threshold.py        # Threshold analysis
│   └── analyze_results.py      # Generate plots & tables
│
├── notebooks/                  # Exploration & visualization
│   ├── 01_cfr_sanity.ipynb
│   ├── 02_data_exploration.ipynb
│   ├── 03_scaling_curves.ipynb
│   └── 04_threshold_analysis.ipynb
│
├── tests/                      # Unit tests
│   ├── test_games.py
│   ├── test_cfr.py
│   ├── test_models.py
│   └── test_metrics.py
│
├── data/                       # Generated datasets (gitignored)
├── checkpoints/                # Model checkpoints (gitignored)
└── results/                    # Experiment results
```

---

## 3. CORE ABSTRACTIONS

### 3.1 Game Interface

```python
# src/games/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import numpy as np

@dataclass
class InfoSet:
    """Information set representation"""
    player: int                      # Which player's perspective
    public_state: Tuple              # Publicly visible info
    private_info: Tuple              # This player's private info
    history: Tuple[str, ...]         # Action history
    
    def to_key(self) -> str:
        """Unique string key for this infoset"""
        ...

@dataclass  
class GameState:
    """Full game state (for simulation)"""
    ...

class Game(ABC):
    """Abstract base class for extensive-form games"""
    
    @property
    @abstractmethod
    def num_players(self) -> int: ...
    
    @property
    @abstractmethod
    def num_actions(self) -> int: ...
    
    @abstractmethod
    def initial_state(self) -> GameState: ...
    
    @abstractmethod
    def get_infoset(self, state: GameState, player: int) -> InfoSet: ...
    
    @abstractmethod
    def get_actions(self, state: GameState) -> List[int]: ...
    
    @abstractmethod
    def apply_action(self, state: GameState, action: int) -> GameState: ...
    
    @abstractmethod
    def is_terminal(self, state: GameState) -> bool: ...
    
    @abstractmethod
    def get_payoffs(self, state: GameState) -> np.ndarray: ...
    
    @abstractmethod
    def is_chance_node(self, state: GameState) -> bool: ...
    
    @abstractmethod
    def chance_outcomes(self, state: GameState) -> List[Tuple[GameState, float]]: ...
```

### 3.2 CFR Solver Interface

```python
# src/cfr/solver.py
from dataclasses import dataclass
from typing import Dict
import numpy as np

@dataclass
class CFROutput:
    """Output from CFR for a single infoset"""
    infoset_key: str
    player: int
    cfv: np.ndarray              # Counterfactual values per action [num_actions]
    strategy: np.ndarray         # Current strategy (action probs) [num_actions]
    regrets: np.ndarray          # Cumulative regrets [num_actions]
    reach_prob: float            # Probability of reaching this infoset

@dataclass
class SolverResult:
    """Full solver output"""
    game_name: str
    iterations: int
    exploitability: float
    infoset_data: Dict[str, CFROutput]  # key -> CFROutput

class CFRSolver(ABC):
    """Abstract CFR solver"""
    
    @abstractmethod
    def solve(
        self, 
        game: Game, 
        iterations: int,
        convergence_threshold: float = 1e-6
    ) -> SolverResult: ...
    
    @abstractmethod
    def get_exploitability(self, game: Game) -> float: ...
```

### 3.3 Dataset Schema

```python
# src/data/schema.py
from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class OracleDataPoint:
    """Single training example from oracle"""
    # Input features
    infoset_encoding: np.ndarray    # Encoded infoset [input_dim]
    
    # Oracle labels (targets)
    cfv: np.ndarray                 # Counterfactual values [num_actions]
    strategy: np.ndarray            # Nash strategy [num_actions]
    regrets: np.ndarray             # Immediate CF regrets [num_actions]
    
    # Metadata
    infoset_key: str
    player: int
    reach_prob: float
    
    # Optional: for weighted sampling
    weight: float = 1.0

@dataclass
class DatasetConfig:
    game: str                       # "kuhn", "leduc", "liars_dice"
    cfr_iterations: int             # Oracle quality
    num_samples: int                # Dataset size (D)
    sampling_method: str            # "uniform", "reach_weighted", "regret_weighted"
    seed: int
```

### 3.4 Model Interface

```python
# src/models/base.py
from abc import ABC, abstractmethod
import torch
import torch.nn as nn

class CFVPredictor(nn.Module, ABC):
    """Base class for CFV prediction models"""
    
    def __init__(self, input_dim: int, num_actions: int):
        super().__init__()
        self.input_dim = input_dim
        self.num_actions = num_actions
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Encoded infoset [batch, input_dim]
        Returns:
            cfv_pred: Predicted CFVs [batch, num_actions]
        """
        ...
    
    def predict_strategy(self, x: torch.Tensor) -> torch.Tensor:
        """Convert CFV predictions to strategy via regret matching"""
        cfv = self.forward(x)
        # Regret matching: strategy proportional to positive regrets
        regrets = cfv - cfv.mean(dim=-1, keepdim=True)
        positive_regrets = torch.clamp(regrets, min=0)
        total = positive_regrets.sum(dim=-1, keepdim=True)
        # Uniform if all regrets non-positive
        strategy = torch.where(
            total > 0,
            positive_regrets / total,
            torch.ones_like(positive_regrets) / self.num_actions
        )
        return strategy
    
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
```

### 3.5 Model Configurations (Scaling N)

```python
# src/models/configs.py

MODEL_CONFIGS = {
    # Tiny models for debugging
    "tiny": {
        "hidden_dims": [32],
        "approx_params": 1_000,
    },
    
    # Scaling ladder for experiments
    "xs": {
        "hidden_dims": [64, 32],
        "approx_params": 5_000,
    },
    "s": {
        "hidden_dims": [128, 64],
        "approx_params": 20_000,
    },
    "m": {
        "hidden_dims": [256, 128, 64],
        "approx_params": 100_000,
    },
    "l": {
        "hidden_dims": [512, 256, 128],
        "approx_params": 400_000,
    },
    "xl": {
        "hidden_dims": [1024, 512, 256, 128],
        "approx_params": 1_500_000,
    },
    "xxl": {
        "hidden_dims": [2048, 1024, 512, 256],
        "approx_params": 6_000_000,
    },
}

# Dataset sizes for scaling D
DATASET_SIZES = [100, 500, 1_000, 5_000, 10_000, 50_000, 100_000, 500_000]

# Search budgets for scaling C  
SEARCH_BUDGETS = [0, 10, 50, 100, 500, 1000, 5000]
```

---

## 4. GAME IMPLEMENTATIONS

### 4.1 Kuhn Poker (Warmup)
- 3 cards: J, Q, K
- 2 players, 1 card each
- Actions: Check, Bet (1 chip)
- ~12 information sets total
- **Use for:** Sanity checks, debugging

```
Rules:
1. Both players ante 1 chip
2. Each dealt 1 card from {J, Q, K}
3. P1 acts: Check or Bet
4. If P1 checks: P2 can Check (showdown) or Bet
5. If bet made: other player can Fold or Call
6. Higher card wins at showdown
```

### 4.2 Leduc Poker (Main Testbed)
- 6 cards: 2 Jacks, 2 Queens, 2 Kings
- 2 players, 1 private card each
- 2 betting rounds (preflop + flop with 1 community card)
- Raise sizes: 2 chips (round 1), 4 chips (round 2)
- Max 2 raises per round
- ~936 information sets
- **Use for:** Main scaling experiments

```
Rules:
1. Both players ante 1 chip
2. Each dealt 1 private card
3. Round 1 betting (raise = 2 chips)
4. Community card dealt
5. Round 2 betting (raise = 4 chips)
6. Showdown: pair beats high card, else high card wins
```

### 4.3 Liar's Dice (Generalization Test)
- Each player has N dice (hidden)
- Players bid on total count of a face value
- Can raise bid or call "liar"
- **Use for:** Testing if results generalize beyond poker

---

## 5. INFORMATION SET ENCODING

### Encoding Strategy for Leduc

```python
def encode_leduc_infoset(infoset: InfoSet) -> np.ndarray:
    """
    Encode Leduc poker infoset as fixed-size vector.
    
    Components:
    1. Private card: one-hot [6] (JJ, QQ, KK × 2 suits conceptually, but 3 ranks)
    2. Public card: one-hot [7] (6 cards + "not yet dealt")
    3. Betting history: 
       - Round 1 actions: [max_actions × num_action_types]
       - Round 2 actions: [max_actions × num_action_types]
    4. Pot odds / stack info: [4] normalized values
    
    Total: ~50-100 dimensional depending on encoding choices
    """
    encoding = []
    
    # Private card (one-hot over 3 ranks, ignoring suit)
    private_card = np.zeros(3)
    private_card[card_to_rank(infoset.private_info[0])] = 1
    encoding.append(private_card)
    
    # Public card (one-hot, with "none" option)
    public_card = np.zeros(4)  # J, Q, K, None
    if infoset.public_state.community_card is not None:
        public_card[card_to_rank(infoset.public_state.community_card)] = 1
    else:
        public_card[3] = 1  # None indicator
    encoding.append(public_card)
    
    # Action history encoding
    # Fixed-length: pad/truncate to max_history_len
    history_encoding = encode_action_history(infoset.history, max_len=12)
    encoding.append(history_encoding)
    
    # Pot and stack info (normalized)
    pot_info = np.array([
        infoset.public_state.pot / MAX_POT,
        infoset.public_state.player_stacks[0] / STARTING_STACK,
        infoset.public_state.player_stacks[1] / STARTING_STACK,
        infoset.public_state.round / 2.0,  # 0 or 0.5
    ])
    encoding.append(pot_info)
    
    return np.concatenate(encoding)
```

### Alternative: Learned Embeddings
Could also learn embeddings for cards and use attention over action history. Start simple (concatenated one-hots), add complexity if needed.

---

## 6. CFR IMPLEMENTATION NOTES

### Vanilla CFR Core Loop

```python
def cfr(game: Game, iterations: int) -> Dict[str, CFROutput]:
    """
    Vanilla CFR implementation.
    
    Key data structures:
    - regret_sum[infoset][action]: cumulative regrets
    - strategy_sum[infoset][action]: cumulative strategy (for averaging)
    """
    regret_sum = defaultdict(lambda: np.zeros(game.num_actions))
    strategy_sum = defaultdict(lambda: np.zeros(game.num_actions))
    
    for t in range(iterations):
        for player in range(game.num_players):
            cfr_recursive(
                game=game,
                state=game.initial_state(),
                player=player,
                reach_probs=np.ones(game.num_players),
                regret_sum=regret_sum,
                strategy_sum=strategy_sum,
            )
    
    # Compute average strategy and final CFVs
    results = {}
    for infoset_key, regrets in regret_sum.items():
        strategy = regret_matching(regrets)
        avg_strategy = normalize(strategy_sum[infoset_key])
        results[infoset_key] = CFROutput(
            infoset_key=infoset_key,
            cfv=compute_cfv(...),  # Recompute with final strategy
            strategy=avg_strategy,
            regrets=regrets,
            reach_prob=...,
        )
    return results

def regret_matching(regrets: np.ndarray) -> np.ndarray:
    """Convert regrets to strategy"""
    positive = np.maximum(regrets, 0)
    total = positive.sum()
    if total > 0:
        return positive / total
    return np.ones_like(regrets) / len(regrets)
```

### Discounted CFR (DCFR) - Recommended
Faster convergence. Discounts older iterations:
```python
# At iteration t, weight regrets and strategies by:
# - Positive regrets: weight = t^α / (t^α + 1), α=1.5
# - Negative regrets: weight = t^β / (t^β + 1), β=0.5
# - Strategy: weight = (t / (t+1))^γ, γ=2
```

### Exploitability Calculation

```python
def compute_exploitability(game: Game, strategy: Dict[str, np.ndarray]) -> float:
    """
    Exploitability = sum of best-response values for each player.
    
    For 2-player zero-sum: exploitability = br_value_p1 + br_value_p2
    At Nash equilibrium, this equals 0.
    """
    total = 0.0
    for player in range(game.num_players):
        br_value = compute_best_response_value(game, strategy, player)
        total += br_value
    return total / game.num_players  # Average exploitability
```

---

## 7. TRAINING PIPELINE

### Loss Functions

```python
# src/training/losses.py
import torch
import torch.nn.functional as F

def cfv_mse_loss(pred_cfv: torch.Tensor, target_cfv: torch.Tensor) -> torch.Tensor:
    """Primary loss: MSE on counterfactual values"""
    return F.mse_loss(pred_cfv, target_cfv)

def cfv_huber_loss(pred_cfv: torch.Tensor, target_cfv: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    """Huber loss: more robust to outliers"""
    return F.huber_loss(pred_cfv, target_cfv, delta=delta)

def strategy_kl_loss(pred_strategy: torch.Tensor, target_strategy: torch.Tensor) -> torch.Tensor:
    """KL divergence on strategies (optional auxiliary loss)"""
    # Add small epsilon for numerical stability
    pred_log = torch.log(pred_strategy + 1e-8)
    return F.kl_div(pred_log, target_strategy, reduction='batchmean')

def regret_loss(pred_cfv: torch.Tensor, target_regrets: torch.Tensor) -> torch.Tensor:
    """
    Loss on immediate counterfactual regrets.
    Regret = CFV(a) - sum_a' strategy(a') * CFV(a')
    """
    pred_regrets = pred_cfv - (pred_cfv.mean(dim=-1, keepdim=True))
    return F.mse_loss(pred_regrets, target_regrets)

def combined_loss(
    pred_cfv: torch.Tensor,
    target_cfv: torch.Tensor,
    target_strategy: torch.Tensor,
    alpha: float = 0.1,  # Weight for strategy loss
) -> torch.Tensor:
    """Combined CFV + strategy loss"""
    cfv_loss = cfv_mse_loss(pred_cfv, target_cfv)
    
    # Derive strategy from predicted CFV
    pred_strategy = regret_matching_batched(pred_cfv)
    strat_loss = strategy_kl_loss(pred_strategy, target_strategy)
    
    return cfv_loss + alpha * strat_loss
```

### Training Loop

```python
# src/training/trainer.py
from dataclasses import dataclass
from typing import Optional, Callable
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

@dataclass
class TrainingConfig:
    # Model
    model_size: str              # "xs", "s", "m", "l", "xl"
    
    # Data
    dataset_size: int            # D
    batch_size: int = 256
    
    # Optimization
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 100
    early_stopping_patience: int = 10
    
    # Loss
    loss_fn: str = "mse"         # "mse", "huber", "combined"
    
    # Logging
    eval_every: int = 5
    log_every: int = 100
    
    # Reproducibility
    seed: int = 42

class Trainer:
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
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.epochs
        )
        
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.history = {"train_loss": [], "val_loss": [], "val_cfv_mae": []}
    
    def train(self) -> dict:
        """Main training loop. Returns training history."""
        for epoch in range(self.config.epochs):
            train_loss = self._train_epoch()
            
            if epoch % self.config.eval_every == 0:
                val_metrics = self._evaluate()
                self.history["train_loss"].append(train_loss)
                self.history["val_loss"].append(val_metrics["loss"])
                self.history["val_cfv_mae"].append(val_metrics["cfv_mae"])
                
                # Early stopping
                if val_metrics["loss"] < self.best_val_loss:
                    self.best_val_loss = val_metrics["loss"]
                    self.patience_counter = 0
                    self._save_checkpoint("best")
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.config.early_stopping_patience:
                        print(f"Early stopping at epoch {epoch}")
                        break
            
            self.scheduler.step()
        
        return self.history
    
    def _train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        
        for batch in self.train_loader:
            x, target_cfv = batch["encoding"], batch["cfv"]
            
            self.optimizer.zero_grad()
            pred_cfv = self.model(x)
            loss = cfv_mse_loss(pred_cfv, target_cfv)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(self.train_loader)
    
    def _evaluate(self) -> dict:
        self.model.eval()
        total_loss = 0.0
        total_mae = 0.0
        
        with torch.no_grad():
            for batch in self.val_loader:
                x, target_cfv = batch["encoding"], batch["cfv"]
                pred_cfv = self.model(x)
                
                total_loss += cfv_mse_loss(pred_cfv, target_cfv).item()
                total_mae += torch.abs(pred_cfv - target_cfv).mean().item()
        
        n = len(self.val_loader)
        return {"loss": total_loss / n, "cfv_mae": total_mae / n}
```

---

## 8. EVALUATION & METRICS

### Primary Metrics

```python
# src/evaluation/cfv_metrics.py

def cfv_mse(pred: np.ndarray, target: np.ndarray) -> float:
    """Mean squared error on CFV predictions"""
    return np.mean((pred - target) ** 2)

def cfv_mae(pred: np.ndarray, target: np.ndarray) -> float:
    """Mean absolute error on CFV predictions"""
    return np.mean(np.abs(pred - target))

def strategy_accuracy(pred_strategy: np.ndarray, target_strategy: np.ndarray, threshold: float = 0.1) -> float:
    """Fraction of actions where |pred - target| < threshold"""
    return np.mean(np.abs(pred_strategy - target_strategy) < threshold)

def action_agreement(pred_strategy: np.ndarray, target_strategy: np.ndarray) -> float:
    """Fraction where argmax agrees"""
    return np.mean(pred_strategy.argmax(axis=-1) == target_strategy.argmax(axis=-1))
```

### Exploitability Evaluation

```python
# src/evaluation/exploitability.py

def evaluate_exploitability(
    model: CFVPredictor,
    game: Game,
    num_samples: int = 10000,
) -> float:
    """
    Compute exploitability of the strategy induced by the model.
    
    1. For each infoset, get model's predicted CFVs
    2. Convert to strategy via regret matching
    3. Compute best-response value against this strategy
    4. Return sum of BR values (exploitability)
    """
    # Build strategy profile from model
    strategy = {}
    for infoset in game.all_infosets():
        encoding = encode_infoset(infoset)
        with torch.no_grad():
            cfv_pred = model(torch.tensor(encoding).unsqueeze(0))
        strategy[infoset.to_key()] = regret_matching(cfv_pred.numpy()[0])
    
    # Compute exploitability
    return compute_exploitability(game, strategy)
```

### Online Search Evaluation

```python
# src/evaluation/search.py

def evaluate_with_search(
    model: CFVPredictor,
    game: Game,
    search_iterations: int,  # C parameter
) -> dict:
    """
    Evaluate model with online CFR refinement.
    
    At each infoset:
    1. Initialize strategy from model's prediction
    2. Run `search_iterations` of CFR from this point
    3. Use refined strategy for action
    
    Returns exploitability and other metrics.
    """
    def get_strategy_with_search(infoset: InfoSet) -> np.ndarray:
        # Get model's initial prediction
        encoding = encode_infoset(infoset)
        with torch.no_grad():
            cfv_pred = model(torch.tensor(encoding).unsqueeze(0))
        initial_strategy = regret_matching(cfv_pred.numpy()[0])
        
        if search_iterations == 0:
            return initial_strategy
        
        # Run limited CFR from this infoset
        refined_strategy = online_cfr_resolve(
            game=game,
            root_infoset=infoset,
            initial_strategy=initial_strategy,
            iterations=search_iterations,
        )
        return refined_strategy
    
    # Evaluate with search-augmented strategy
    exploitability = compute_exploitability_with_fn(game, get_strategy_with_search)
    
    return {
        "exploitability": exploitability,
        "search_iterations": search_iterations,
    }
```

---

## 9. EXPERIMENT CONFIGURATIONS

### Experiment 1: Scaling N (Model Size)

```yaml
# experiments/configs/scaling_N.yaml
experiment: scaling_N
game: leduc
oracle:
  cfr_iterations: 10000
  convergence_threshold: 1e-6
dataset:
  size: 50000  # Fixed D
  sampling: reach_weighted
  seed: 42
models:
  - xs    # ~5k params
  - s     # ~20k params
  - m     # ~100k params
  - l     # ~400k params
  - xl    # ~1.5M params
  - xxl   # ~6M params
training:
  epochs: 200
  batch_size: 256
  learning_rate: 1e-3
  early_stopping_patience: 20
evaluation:
  compute_exploitability: true
  exploitability_frequency: 10  # Every 10 epochs
seeds: [42, 123, 456]  # 3 seeds for error bars
```

### Experiment 2: Scaling D (Dataset Size)

```yaml
# experiments/configs/scaling_D.yaml
experiment: scaling_D
game: leduc
oracle:
  cfr_iterations: 10000
dataset:
  sizes: [100, 500, 1000, 5000, 10000, 50000, 100000, 500000]
  sampling: reach_weighted
models:
  - m  # Fixed model size
training:
  epochs: 200
  batch_size: 256
seeds: [42, 123, 456]
```

### Experiment 3: Search Threshold Analysis

```yaml
# experiments/configs/threshold.yaml
experiment: threshold
game: leduc
oracle:
  cfr_iterations: 10000
dataset:
  size: 50000
models:
  - xs   # Under-trained
  - s    # Under-trained
  - m    # Near threshold?
  - l    # Above threshold?
  - xl   # Well-trained
search_budgets: [0, 10, 50, 100, 500, 1000, 5000]
evaluation:
  metrics: [exploitability, cfv_mse]
seeds: [42, 123, 456]
```

---

## 10. EXPECTED RESULTS FORMAT

### Scaling Curves Data

```python
# results/scaling_N_leduc.json
{
    "experiment": "scaling_N",
    "game": "leduc",
    "dataset_size": 50000,
    "results": [
        {
            "model": "xs",
            "params": 5120,
            "seeds": {
                "42": {"cfv_mse": 0.0234, "exploitability": 0.156},
                "123": {"cfv_mse": 0.0241, "exploitability": 0.162},
                "456": {"cfv_mse": 0.0228, "exploitability": 0.151},
            },
            "mean_cfv_mse": 0.0234,
            "std_cfv_mse": 0.0006,
            "mean_exploitability": 0.156,
        },
        # ... more models
    ]
}
```

### Threshold Analysis Data

```python
# results/threshold_leduc.json
{
    "experiment": "threshold",
    "results": [
        {
            "model": "xs",
            "params": 5120,
            "search_results": [
                {"search_iters": 0, "exploitability": 0.156},
                {"search_iters": 10, "exploitability": 0.162},  # Worse!
                {"search_iters": 100, "exploitability": 0.158},  # Still worse
                # ...
            ]
        },
        {
            "model": "xl",
            "params": 1500000,
            "search_results": [
                {"search_iters": 0, "exploitability": 0.042},
                {"search_iters": 10, "exploitability": 0.038},  # Better
                {"search_iters": 100, "exploitability": 0.029},  # Much better
                # ...
            ]
        },
    ]
}
```

---

## 11. VISUALIZATION

### Key Plots to Generate

```python
# experiments/analyze_results.py

def plot_scaling_N(results: dict):
    """
    X-axis: log(params)
    Y-axis: log(CFV MSE)
    
    Expected: linear relationship (power law)
    Fit: L = A * N^(-alpha) + E
    Report: alpha coefficient
    """
    ...

def plot_scaling_D(results: dict):
    """
    X-axis: log(dataset_size)
    Y-axis: log(CFV MSE)
    
    Expected: linear relationship (power law)
    Fit: L = B * D^(-beta) + E
    Report: beta coefficient
    """
    ...

def plot_threshold_analysis(results: dict):
    """
    X-axis: search_iterations (log scale)
    Y-axis: exploitability
    
    Multiple lines: one per model size
    Expected: small models flat/worse with search, large models improve
    """
    ...

def plot_N_D_tradeoff(results: dict):
    """
    Contour plot or heatmap
    X: log(N), Y: log(D), Color: CFV MSE
    
    Identify Chinchilla-optimal frontier
    """
    ...
```

---

## 12. IMPLEMENTATION ORDER

### Phase 1: Foundation (Week 1)
1. [ ] Set up project structure, pyproject.toml
2. [ ] Implement Kuhn Poker game
3. [ ] Implement vanilla CFR solver
4. [ ] Verify CFR converges on Kuhn (known Nash)
5. [ ] Unit tests for game + CFR

### Phase 2: Leduc + Data (Week 2)
1. [ ] Implement Leduc Poker game
2. [ ] Implement DCFR (faster convergence)
3. [ ] Oracle data generation pipeline
4. [ ] Dataset class with encoding
5. [ ] Verify exploitability calculation

### Phase 3: Models + Training (Week 3)
1. [ ] MLP model family (configs for each size)
2. [ ] Training loop with logging
3. [ ] CFV loss + optional auxiliary losses
4. [ ] Evaluation metrics
5. [ ] Single training run end-to-end

### Phase 4: Scaling N Experiments (Week 4)
1. [ ] Run scaling_N experiment across model sizes
2. [ ] 3 seeds per configuration
3. [ ] Generate scaling curve plot
4. [ ] Fit power law, extract exponent

### Phase 5: Scaling D Experiments (Week 5)
1. [ ] Generate datasets of varying sizes
2. [ ] Run scaling_D experiment
3. [ ] Generate scaling curve plot
4. [ ] Fit power law, extract exponent

### Phase 6: Threshold Analysis (Week 6)
1. [ ] Implement online search evaluation
2. [ ] Run threshold experiment
3. [ ] Generate threshold plots
4. [ ] Identify N_critical if it exists

### Phase 7: Polish + Paper (Week 7-8)
1. [ ] Liar's Dice generalization test
2. [ ] Clean up visualizations
3. [ ] Write up results
4. [ ] Error analysis, ablations

---

## 13. DEPENDENCIES

```toml
# pyproject.toml
[project]
name = "sigma-go"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "numpy>=1.24",
    "torch>=2.0",
    "scipy>=1.10",
    "matplotlib>=3.7",
    "seaborn>=0.12",
    "pandas>=2.0",
    "tqdm>=4.65",
    "pyyaml>=6.0",
    "wandb>=0.15",  # Optional: experiment tracking
    "pytest>=7.0",
]

[project.optional-dependencies]
dev = [
    "jupyter>=1.0",
    "ipykernel>=6.0",
    "black>=23.0",
    "ruff>=0.0.270",
]
```

---

## 14. GOTCHAS & TIPS

### CFR Pitfalls
- **Regret matching with all-negative regrets**: Must default to uniform, not zeros
- **Floating point precision**: Use float64 for CFR accumulators
- **Traversal order**: Must traverse for EACH player separately per iteration
- **Chance sampling**: MCCFR needs careful handling of chance node probabilities

### Training Pitfalls
- **Overfitting to small D**: Use dropout, weight decay, early stopping
- **CFV scale**: Normalize CFVs to similar scale across games
- **Reach probability weighting**: Consider weighting loss by reach probability

### Evaluation Pitfalls
- **Exploitability is expensive**: Don't compute every epoch, sample or compute periodically
- **Strategy extraction**: Regret matching from predicted CFVs, not softmax
- **Search fairness**: When comparing search budgets, ensure same random seeds

### Scaling Law Fitting
- **Use log-log plots**: Power laws appear linear
- **Fit in log space**: log(L) = -α*log(N) + log(A) + log(1 + (E/A)*N^α)
- **Report confidence intervals**: Need multiple seeds
- **Check residuals**: Make sure power law is actually a good fit

---

## 15. SUCCESS CRITERIA

### Minimum Viable Result
- [ ] Clean scaling curve for N with R² > 0.9
- [ ] Clean scaling curve for D with R² > 0.9
- [ ] Observable difference in search benefit between small and large models

### Strong Result
- [ ] Power law exponents consistent across seeds (std < 0.1)
- [ ] Clear threshold N_critical where search transitions from harmful to helpful
- [ ] Results replicate on Liar's Dice

### Publication-Ready
- [ ] All of the above
- [ ] Chinchilla-style optimal N/D tradeoff characterized
- [ ] Clean narrative connecting to LLM scaling laws

---

## 16. REFERENCES TO CITE

1. **Deep CFR**: Brown et al., 2019 - "Deep Counterfactual Regret Minimization"
2. **ReBeL**: Brown et al., 2020 - "Combining Deep RL and Search for Imperfect-Information Games"  
3. **Scaling Laws (LLMs)**: Kaplan et al., 2020 - "Scaling Laws for Neural Language Models"
4. **Chinchilla**: Hoffmann et al., 2022 - "Training Compute-Optimal Large Language Models"
5. **Test-Time Compute**: Snell et al., 2024 - "Scaling LLM Test-Time Compute"
6. **DeepStack**: Moravčík et al., 2017 - "DeepStack: Expert-Level AI in Heads-Up No-Limit Poker"
7. **CFR Original**: Zinkevich et al., 2008 - "Regret Minimization in Games with Incomplete Information"
8. **DCFR**: Brown & Sandholm, 2019 - "Solving Imperfect-Information Games via Discounted Regret Minimization"

---

**END OF BUILD CONTEXT**

Feed this entire document to Claude Code. Start with Phase 1.
