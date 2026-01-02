# CHEATSHEET.md - Copy-Paste Patterns

## Regret Matching (CRITICAL - used everywhere)

```python
def regret_matching(regrets: np.ndarray) -> np.ndarray:
    """Convert cumulative regrets to strategy. MUST handle all-negative case."""
    positive = np.maximum(regrets, 0)
    total = positive.sum()
    if total > 0:
        return positive / total
    else:
        return np.ones_like(regrets) / len(regrets)  # Uniform fallback
```

## Kuhn Poker Constants

```python
KUHN_CARDS = ['J', 'Q', 'K']
KUHN_ACTIONS = ['p', 'b']  # pass/check, bet
KUHN_NUM_ACTIONS = 2

# Known Nash equilibrium for P1 with J:
# - Always pass first
# - If P2 bets, fold (pass)
# Alpha parameter α ∈ [0, 1/3] controls bluffing frequency
```

## Leduc Poker Constants

```python
LEDUC_CARDS = ['Js', 'Jh', 'Qs', 'Qh', 'Ks', 'Kh']  # 6 cards
LEDUC_RANKS = ['J', 'Q', 'K']
LEDUC_ACTIONS = ['f', 'c', 'r']  # fold, call/check, raise
LEDUC_NUM_ACTIONS = 3

LEDUC_ANTE = 1
LEDUC_RAISE_R1 = 2
LEDUC_RAISE_R2 = 4
LEDUC_MAX_RAISES = 2
```

## CFR Recursive Traversal Pattern

```python
def cfr_traverse(game, state, player, reach_probs, regret_sum, strategy_sum):
    """Core CFR recursive traversal."""
    if game.is_terminal(state):
        return game.get_payoffs(state)[player]
    
    if game.is_chance_node(state):
        value = 0.0
        for next_state, prob in game.chance_outcomes(state):
            value += prob * cfr_traverse(game, next_state, player, reach_probs, regret_sum, strategy_sum)
        return value
    
    current_player = game.current_player(state)
    infoset = game.get_infoset(state, current_player)
    key = infoset.to_key()
    actions = game.get_actions(state)
    
    # Get current strategy from regrets
    strategy = regret_matching(regret_sum[key])
    
    # Compute action values
    action_values = np.zeros(len(actions))
    for i, action in enumerate(actions):
        next_state = game.apply_action(state, action)
        new_reach = reach_probs.copy()
        new_reach[current_player] *= strategy[i]
        action_values[i] = cfr_traverse(game, next_state, player, new_reach, regret_sum, strategy_sum)
    
    # Node value
    node_value = np.dot(strategy, action_values)
    
    # Update regrets and strategy (only for traversing player)
    if current_player == player:
        opp_reach = np.prod([reach_probs[p] for p in range(game.num_players) if p != player])
        for i, action in enumerate(actions):
            regret_sum[key][i] += opp_reach * (action_values[i] - node_value)
        strategy_sum[key] += reach_probs[player] * strategy
    
    return node_value
```

## Exploitability Calculation Pattern

```python
def compute_exploitability(game, strategy):
    """
    Exploitability = sum of best-response values.
    At Nash: exploitability = 0.
    """
    total = 0.0
    for player in range(game.num_players):
        br_value = compute_best_response_value(game, strategy, player)
        total += br_value
    return total

def compute_best_response_value(game, strategy, player):
    """Value of best response for `player` against fixed `strategy`."""
    return _br_traverse(game, game.initial_state(), player, strategy, 1.0)

def _br_traverse(game, state, player, strategy, prob):
    if game.is_terminal(state):
        return prob * game.get_payoffs(state)[player]
    
    if game.is_chance_node(state):
        return sum(p * _br_traverse(game, s, player, strategy, prob) 
                   for s, p in game.chance_outcomes(state))
    
    current = game.current_player(state)
    infoset = game.get_infoset(state, current)
    actions = game.get_actions(state)
    
    if current == player:
        # Best response: take max
        return max(_br_traverse(game, game.apply_action(state, a), player, strategy, prob)
                   for a in actions)
    else:
        # Opponent plays fixed strategy
        strat = strategy[infoset.to_key()]
        return sum(strat[i] * _br_traverse(game, game.apply_action(state, a), player, strategy, prob)
                   for i, a in enumerate(actions))
```

## Infoset Encoding Pattern

```python
def encode_infoset(infoset, game_config):
    """Generic encoding pattern."""
    parts = []
    
    # 1. Private info (one-hot)
    private_onehot = np.zeros(game_config.num_private_states)
    private_onehot[infoset.private_index] = 1
    parts.append(private_onehot)
    
    # 2. Public info (one-hot or continuous)
    public_onehot = np.zeros(game_config.num_public_states)
    public_onehot[infoset.public_index] = 1
    parts.append(public_onehot)
    
    # 3. Action history (fixed-length encoding)
    history_enc = encode_history(infoset.history, max_len=game_config.max_history)
    parts.append(history_enc)
    
    return np.concatenate(parts).astype(np.float32)
```

## Model Architecture Pattern

```python
class CFVPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_actions, dropout=0.1):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(prev_dim, num_actions)
    
    def forward(self, x):
        """Returns CFV predictions [batch, num_actions]"""
        features = self.backbone(x)
        return self.head(features)
    
    def get_strategy(self, x):
        """Convert CFV to strategy via regret matching."""
        cfv = self.forward(x)
        return regret_matching_batch(cfv)

def regret_matching_batch(cfv):
    """Batched regret matching."""
    regrets = cfv - cfv.mean(dim=-1, keepdim=True)
    positive = torch.clamp(regrets, min=0)
    total = positive.sum(dim=-1, keepdim=True)
    uniform = torch.ones_like(positive) / positive.shape[-1]
    return torch.where(total > 0, positive / total, uniform)
```

## Training Loop Pattern

```python
def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        x = batch['encoding'].to(device)
        target_cfv = batch['cfv'].to(device)
        
        optimizer.zero_grad()
        pred_cfv = model(x)
        loss = F.mse_loss(pred_cfv, target_cfv)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
    return total_loss / len(loader)
```

## Scaling Law Fitting Pattern

```python
from scipy.optimize import curve_fit

def power_law(x, a, alpha, e):
    """L = a * x^(-alpha) + e"""
    return a * np.power(x, -alpha) + e

def fit_scaling_law(x_data, y_data):
    """Fit power law in log-log space."""
    # Initial guess
    p0 = [y_data[0], 0.5, y_data[-1] * 0.1]
    
    # Fit
    popt, pcov = curve_fit(power_law, x_data, y_data, p0=p0, maxfev=10000)
    
    # R² calculation
    residuals = y_data - power_law(x_data, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_data - np.mean(y_data))**2)
    r_squared = 1 - (ss_res / ss_tot)
    
    return {
        'a': popt[0],
        'alpha': popt[1],
        'e': popt[2],
        'r_squared': r_squared,
    }
```

## Plotting Pattern

```python
import matplotlib.pyplot as plt

def plot_scaling_curve(results, xlabel, title, save_path):
    """Standard scaling curve plot."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    x = np.array([r['x'] for r in results])
    y_mean = np.array([r['y_mean'] for r in results])
    y_std = np.array([r['y_std'] for r in results])
    
    # Log-log plot
    ax.loglog(x, y_mean, 'o-', markersize=8, linewidth=2)
    ax.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.3)
    
    # Fit line
    fit = fit_scaling_law(x, y_mean)
    x_fit = np.logspace(np.log10(x.min()), np.log10(x.max()), 100)
    y_fit = power_law(x_fit, fit['a'], fit['alpha'], fit['e'])
    ax.loglog(x_fit, y_fit, '--', color='red', 
              label=f"α={fit['alpha']:.3f}, R²={fit['r_squared']:.3f}")
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel('CFV MSE')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
```

## Experiment Runner Pattern

```python
def run_experiment(config_path):
    config = load_yaml(config_path)
    results = []
    
    for model_size in config['models']:
        for seed in config['seeds']:
            set_seed(seed)
            
            # Setup
            model = create_model(model_size, config)
            train_loader, val_loader = create_dataloaders(config)
            
            # Train
            trainer = Trainer(model, train_loader, val_loader, config)
            history = trainer.train()
            
            # Evaluate
            metrics = evaluate(model, config)
            
            results.append({
                'model_size': model_size,
                'params': model.num_parameters(),
                'seed': seed,
                **metrics,
            })
    
    # Aggregate
    aggregated = aggregate_results(results)
    save_results(aggregated, config['output_path'])
    return aggregated
```
