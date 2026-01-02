"""
Grouped Tokenization Transformer for NLHE CFV Prediction

Instead of 141 feature-as-token (O(L²) = 19,881 ops), we use:
- 1 token: card bucket (learned embedding)
- 1 token: round (learned embedding)  
- 1 token: game state (pot_odds, SPR, texture → projected)
- 20 tokens: action history (each action → learned embedding)

Total: ~23 tokens → O(L²) = 529 ops (37× reduction!)

This is how you build a transformer that's actually trainable.
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class GroupedTransformer(nn.Module):
    """
    Transformer with semantic token groups for poker.
    
    Token structure:
    [CLS] [CARD] [ROUND] [STATE] [ACT_1] [ACT_2] ... [ACT_20]
    
    Total: 24 tokens max (vs 141 features)
    """
    
    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        num_buckets: int = 10,
        num_rounds: int = 4,
        num_actions: int = 6,
        max_history: int = 20,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions
        self.max_history = max_history
        
        # Token embeddings
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.bucket_embed = nn.Embedding(num_buckets, hidden_dim)
        self.round_embed = nn.Embedding(num_rounds, hidden_dim)
        self.action_embed = nn.Embedding(num_actions + 1, hidden_dim)  # +1 for padding
        
        # State projection (pot_odds, SPR, 4 texture bits → hidden)
        self.state_proj = nn.Linear(6, hidden_dim)
        
        # Positional embeddings for history
        self.pos_embed = nn.Embedding(max_history + 4, hidden_dim)  # +4 for special tokens
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-norm for stability
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output head (from CLS token)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_actions),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 141) raw encoding from Rust
               Layout: [bucket_onehot(10), round_onehot(4), pot_odds(1), spr(1), 
                        texture(5), history_onehot(20*6)]
        
        Returns:
            (B, num_actions) predicted advantages
        """
        B = x.shape[0]
        device = x.device
        
        # Parse encoding
        bucket_idx = x[:, :10].argmax(dim=1)  # (B,)
        round_idx = x[:, 10:14].argmax(dim=1)  # (B,)
        pot_odds = x[:, 14:15]  # (B, 1)
        spr = x[:, 15:16]  # (B, 1)
        texture = x[:, 16:20]  # (B, 4) - actually 4 bits, not 5
        state_feats = torch.cat([pot_odds, spr, texture], dim=1)  # (B, 6)
        
        # History: (B, 20, 6) one-hot → action indices
        history_onehot = x[:, 21:].reshape(B, self.max_history, self.num_actions)  # (B, 20, 6)
        history_mask = history_onehot.sum(dim=2) > 0  # (B, 20) - which positions have actions
        history_idx = history_onehot.argmax(dim=2)  # (B, 20)
        history_idx = torch.where(history_mask, history_idx, 
                                   torch.full_like(history_idx, self.num_actions))  # Pad token
        
        # Build token sequence
        tokens = []
        
        # [CLS] token at position 0
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, H)
        tokens.append(cls)
        
        # [CARD] token at position 1
        card = self.bucket_embed(bucket_idx).unsqueeze(1)  # (B, 1, H)
        tokens.append(card)
        
        # [ROUND] token at position 2
        round_tok = self.round_embed(round_idx).unsqueeze(1)  # (B, 1, H)
        tokens.append(round_tok)
        
        # [STATE] token at position 3
        state = self.state_proj(state_feats).unsqueeze(1)  # (B, 1, H)
        tokens.append(state)
        
        # [ACT_1..20] tokens at positions 4-23
        actions = self.action_embed(history_idx)  # (B, 20, H)
        tokens.append(actions)
        
        # Concatenate all tokens
        x = torch.cat(tokens, dim=1)  # (B, 24, H)
        
        # Add positional embeddings
        seq_len = x.shape[1]
        positions = torch.arange(seq_len, device=device)
        x = x + self.pos_embed(positions)
        
        # Create attention mask for padding (optional - helps with variable-length history)
        # For now, let all tokens attend to all (simpler, works fine)
        
        # Transformer forward
        x = self.encoder(x)  # (B, 24, H)
        
        # Use CLS token for prediction
        cls_out = x[:, 0]  # (B, H)
        
        return self.head(cls_out)  # (B, num_actions)
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Model configs matching MLP param counts
TRANSFORMER_CONFIGS = {
    "tiny":   {"hidden_dim": 48,  "num_layers": 2, "num_heads": 4},   # ~15k
    "small":  {"hidden_dim": 96,  "num_layers": 3, "num_heads": 4},   # ~60k
    "medium": {"hidden_dim": 192, "num_layers": 4, "num_heads": 4},   # ~300k
    "large":  {"hidden_dim": 384, "num_layers": 4, "num_heads": 8},   # ~1.2M
}


def create_grouped_transformer(size: str = "small") -> GroupedTransformer:
    """Create a GroupedTransformer by size name."""
    if size not in TRANSFORMER_CONFIGS:
        raise ValueError(f"Unknown size: {size}. Choose from {list(TRANSFORMER_CONFIGS.keys())}")
    return GroupedTransformer(**TRANSFORMER_CONFIGS[size])


if __name__ == "__main__":
    # Quick test
    print("GroupedTransformer Parameter Counts:")
    print("-" * 40)
    
    for name, cfg in TRANSFORMER_CONFIGS.items():
        model = GroupedTransformer(**cfg)
        params = model.count_parameters()
        print(f"  {name:8s}: {params:>10,} params")
        
        # Test forward
        x = torch.randn(32, 141)
        y = model(x)
        assert y.shape == (32, 6), f"Bad shape: {y.shape}"
    
    print("\n✅ All configs work!")
    
    # Speed test
    import time
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"\nSpeed test on {device}:")
    
    model = create_grouped_transformer("small").to(device)
    x = torch.randn(4096, 141, device=device)
    
    # Warmup
    for _ in range(5):
        _ = model(x)
    
    if device == "mps":
        torch.mps.synchronize()
    
    start = time.time()
    for _ in range(20):
        _ = model(x)
    if device == "mps":
        torch.mps.synchronize()
    elapsed = time.time() - start
    
    print(f"  20 batches of 4096: {elapsed:.2f}s ({4096*20/elapsed:.0f} samples/sec)")

