"""
Transformer model for CFV prediction.

Uses a Set Transformer / Encoder-style architecture:
- No causal masking (infoset components are unordered)
- Self-attention over feature groups
- Pooled representation → CFV prediction

This is the controlled upgrade from MLP for scaling comparison.
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import CFVPredictor


@dataclass
class TransformerConfig:
    """Configuration for Transformer model."""
    d_model: int = 64         # Model dimension
    n_heads: int = 4          # Number of attention heads
    n_layers: int = 2         # Number of transformer layers
    d_ff: int = 256           # Feed-forward hidden dimension
    dropout: float = 0.1      # Dropout rate
    max_seq_len: int = 16     # Max sequence length (feature groups)


# Model size configurations (matched to MLP param counts approximately)
# Tuned for input_dim=40, num_actions=3 (Leduc poker)
TRANSFORMER_CONFIGS = {
    # ~2k params (matches MLP tiny ~1.9k)
    "tiny": TransformerConfig(d_model=16, n_heads=2, n_layers=1, d_ff=32, dropout=0.0),
    
    # ~5k params (matches MLP xs ~4.8k)
    "xs": TransformerConfig(d_model=16, n_heads=2, n_layers=2, d_ff=64, dropout=0.0),
    
    # ~14k params (matches MLP s ~13.7k)
    "s": TransformerConfig(d_model=24, n_heads=4, n_layers=3, d_ff=72, dropout=0.1),
    
    # ~52k params (matches MLP m ~52k)
    "m": TransformerConfig(d_model=40, n_heads=4, n_layers=4, d_ff=128, dropout=0.1),
    
    # ~186k params (matches MLP l ~186k)
    "l": TransformerConfig(d_model=56, n_heads=4, n_layers=7, d_ff=192, dropout=0.1),
    
    # ~730k params (matches MLP xl ~731k)
    "xl": TransformerConfig(d_model=80, n_heads=8, n_layers=12, d_ff=280, dropout=0.1),
}


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention layer."""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_head)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
            mask: Optional attention mask [batch, seq_len] (True = masked)
        
        Returns:
            out: [batch, seq_len, d_model]
        """
        B, L, D = x.shape
        
        # Project to Q, K, V
        Q = self.q_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)  # [B, H, L, D_h]
        K = self.k_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        V = self.v_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        
        # Attention scores
        attn = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [B, H, L, L]
        
        if mask is not None:
            # Expand mask for broadcasting: [B, L] -> [B, 1, 1, L]
            mask = mask.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(mask, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, V)  # [B, H, L, D_h]
        out = out.transpose(1, 2).contiguous().view(B, L, D)  # [B, L, D]
        
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    """Single transformer encoder block with pre-norm."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-norm residual attention
        x = x + self.attn(self.ln1(x), mask)
        # Pre-norm residual FFN
        x = x + self.ff(self.ln2(x))
        return x


class SetTransformer(CFVPredictor):
    """
    Set Transformer for CFV prediction (scalar tokenization - baseline).
    
    Treats infoset encoding as a set of feature tokens.
    Known limitation: treats each scalar as a token, which is suboptimal.
    Use GroupedTransformer for proper poker tokenization.
    """
    
    def __init__(self, input_dim: int, num_actions: int, config: TransformerConfig, 
                 use_pos_embed: bool = True):
        super().__init__(input_dim, num_actions)
        self.config = config
        self.use_pos_embed = use_pos_embed
        
        self.input_proj = nn.Linear(1, config.d_model)
        
        if use_pos_embed:
            self.pos_embed = nn.Parameter(torch.randn(1, input_dim, config.d_model) * 0.02)
        else:
            self.register_buffer('pos_embed', torch.zeros(1, input_dim, config.d_model))
        
        self.layers = nn.ModuleList([
            TransformerBlock(config.d_model, config.n_heads, config.d_ff, config.dropout)
            for _ in range(config.n_layers)
        ])
        
        self.ln_final = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, num_actions)
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, D = x.shape
        x = x.unsqueeze(-1)
        x = self.input_proj(x)
        x = x + self.pos_embed[:, :D, :]
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        x = x.mean(dim=1)
        return self.head(x)
    
    @classmethod
    def from_config(cls, input_dim: int, num_actions: int, config: TransformerConfig,
                    use_pos_embed: bool = True) -> "SetTransformer":
        return cls(input_dim, num_actions, config, use_pos_embed)
    
    @classmethod
    def from_name(cls, name: str, input_dim: int, num_actions: int,
                  use_pos_embed: bool = True) -> "SetTransformer":
        if name not in TRANSFORMER_CONFIGS:
            raise ValueError(f"Unknown config: {name}. Available: {list(TRANSFORMER_CONFIGS.keys())}")
        return cls.from_config(input_dim, num_actions, TRANSFORMER_CONFIGS[name], use_pos_embed)


class GroupedTransformer(CFVPredictor):
    """
    Grouped Transformer for CFV prediction.
    
    Tokenizes poker infoset into MEANINGFUL groups:
    - Token 1: private card one-hot (num_ranks dims)
    - Token 2: public card one-hot (num_ranks+1 dims)  
    - Token 3: [pot, round] scalars (2 dims)
    - Tokens 4..N: action history (each action is 3-dim one-hot)
    
    This is the CORRECT architecture for poker CFV prediction.
    """
    
    def __init__(self, num_ranks: int, max_history: int, num_actions: int, 
                 config: TransformerConfig, use_pos_embed: bool = True,
                 allow_memorization: bool = False):
        # Calculate input_dim for base class
        input_dim = num_ranks + (num_ranks + 1) + 2 + (max_history * 3)
        super().__init__(input_dim, num_actions)
        
        self.num_ranks = num_ranks
        self.max_history = max_history
        self.config = config
        self.use_pos_embed = use_pos_embed
        self.allow_memorization = allow_memorization
        
        # Token dimensions
        self.private_dim = num_ranks
        self.public_dim = num_ranks + 1
        self.state_dim = 2  # pot + round
        self.action_dim = 3  # fold/call/raise one-hot
        
        # Number of tokens: private + public + state + history
        self.num_tokens = 3 + max_history
        
        # Per-token projections to d_model
        self.private_proj = nn.Linear(self.private_dim, config.d_model)
        self.public_proj = nn.Linear(self.public_dim, config.d_model)
        self.state_proj = nn.Linear(self.state_dim, config.d_model)
        self.action_proj = nn.Linear(self.action_dim, config.d_model)
        
        # Positional embeddings (optional)
        if use_pos_embed:
            self.pos_embed = nn.Parameter(torch.randn(1, self.num_tokens, config.d_model) * 0.02)
        else:
            self.register_buffer('pos_embed', torch.zeros(1, self.num_tokens, config.d_model))
        
        # Transformer layers
        dropout = 0.0 if allow_memorization else config.dropout
        self.layers = nn.ModuleList([
            TransformerBlock(config.d_model, config.n_heads, config.d_ff, dropout)
            for _ in range(config.n_layers)
        ])
        
        self.ln_final = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, num_actions)
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Encoded infoset [batch, input_dim]
               Layout: [private_card | public_card | pot | round | history...]
        
        Returns:
            cfv: Predicted CFVs [batch, num_actions]
        """
        B = x.shape[0]
        
        # Parse input into groups
        idx = 0
        private = x[:, idx:idx+self.private_dim]
        idx += self.private_dim
        
        public = x[:, idx:idx+self.public_dim]
        idx += self.public_dim
        
        state = x[:, idx:idx+self.state_dim]
        idx += self.state_dim
        
        # Project each group to d_model
        tokens = []
        tokens.append(self.private_proj(private))  # [B, d_model]
        tokens.append(self.public_proj(public))
        tokens.append(self.state_proj(state))
        
        # History tokens
        for i in range(self.max_history):
            action = x[:, idx:idx+self.action_dim]
            idx += self.action_dim
            tokens.append(self.action_proj(action))
        
        # Stack: [B, num_tokens, d_model]
        x = torch.stack(tokens, dim=1)
        
        # Add positional embeddings
        x = x + self.pos_embed
        
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x)
        
        # Final norm and pool
        x = self.ln_final(x)
        x = x.mean(dim=1)  # [B, d_model]
        
        return self.head(x)
    
    @classmethod
    def from_name(cls, name: str, num_ranks: int, max_history: int, num_actions: int,
                  use_pos_embed: bool = True, allow_memorization: bool = False) -> "GroupedTransformer":
        if name not in TRANSFORMER_CONFIGS:
            raise ValueError(f"Unknown config: {name}. Available: {list(TRANSFORMER_CONFIGS.keys())}")
        return cls(num_ranks, max_history, num_actions, TRANSFORMER_CONFIGS[name],
                   use_pos_embed, allow_memorization)


def print_transformer_sizes(input_dim: int = 40, num_actions: int = 3):
    """Print parameter counts for all transformer sizes."""
    print("\n" + "="*60)
    print("TRANSFORMER MODEL SIZES")
    print("="*60)
    print(f"{'Size':<8} {'d_model':<8} {'layers':<8} {'heads':<8} {'Params':<12}")
    print("-"*60)
    
    for name, config in TRANSFORMER_CONFIGS.items():
        model = SetTransformer(input_dim, num_actions, config)
        params = model.num_parameters()
        print(f"{name:<8} {config.d_model:<8} {config.n_layers:<8} {config.n_heads:<8} {params:>10,}")
    
    print("-"*60)


def compare_model_sizes(input_dim: int = 40, num_actions: int = 3):
    """Compare MLP and Transformer parameter counts."""
    from .mlp import MLP
    
    print("\n" + "="*70)
    print("MLP vs TRANSFORMER PARAMETER COMPARISON")
    print("="*70)
    print(f"{'Size':<8} {'MLP Params':<15} {'Transformer Params':<18} {'Ratio':<10}")
    print("-"*70)
    
    for name in ["tiny", "xs", "s", "m", "l", "xl"]:
        mlp = MLP.from_name(name, input_dim, num_actions)
        transformer = SetTransformer.from_name(name, input_dim, num_actions)
        
        mlp_params = mlp.num_parameters()
        transformer_params = transformer.num_parameters()
        ratio = transformer_params / mlp_params
        
        print(f"{name:<8} {mlp_params:>12,}    {transformer_params:>15,}    {ratio:>8.2f}x")
    
    print("-"*70)


if __name__ == "__main__":
    print_transformer_sizes()
    compare_model_sizes()

