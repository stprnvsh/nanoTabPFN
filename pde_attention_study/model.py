"""Attention model for PDE learning."""
import torch
import torch.nn as nn
import math


class AttentionLayer(nn.Module):
    """Single attention layer with FFN."""
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.last_attention = None
    
    def forward(self, x):
        B, S, D = x.shape
        
        residual = x
        x = self.norm1(x)
        
        Q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        
        attn = (Q @ K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = torch.softmax(attn, dim=-1)
        self.last_attention = attn.detach()
        
        x = (attn @ V).transpose(1, 2).reshape(B, S, D)
        x = self.out_proj(x)
        x = residual + x
        
        x = x + self.ffn(self.norm2(x))
        return x


class PDETransformer(nn.Module):
    """
    Transformer for PDE data with configurable layers.
    Input: sequence of spatial points at time t
    Output: sequence of spatial points at time t+1
    """
    def __init__(self, d_model=64, n_heads=4, n_layers=1):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        
        self.input_proj = nn.Linear(1, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, 256, d_model) * 0.02)
        
        self.layers = nn.ModuleList([
            AttentionLayer(d_model, n_heads) for _ in range(n_layers)
        ])
        
        self.output_proj = nn.Linear(d_model, 1)
    
    def forward(self, x, return_attention=False):
        B, S = x.shape
        
        x = x.unsqueeze(-1)
        x = self.input_proj(x)
        x = x + self.pos_embed[:, :S, :]
        
        for layer in self.layers:
            x = layer(x)
        
        out = self.output_proj(x).squeeze(-1)
        
        if return_attention:
            # Return attention from all layers: (n_layers, B, n_heads, S, S)
            all_attn = torch.stack([l.last_attention for l in self.layers])
            return out, all_attn
        return out


# Alias for backward compatibility
SingleLayerAttention = lambda d_model=64, n_heads=4: PDETransformer(d_model, n_heads, n_layers=1)


if __name__ == "__main__":
    # Test 1-layer
    model1 = PDETransformer(n_layers=1)
    x = torch.randn(4, 64)
    out, attn = model1(x, return_attention=True)
    print(f"1-layer: Input {x.shape}, Output {out.shape}, Attention {attn.shape}")
    
    # Test 2-layer
    model2 = PDETransformer(n_layers=2)
    out, attn = model2(x, return_attention=True)
    print(f"2-layer: Input {x.shape}, Output {out.shape}, Attention {attn.shape}")

