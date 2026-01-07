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


class FourierLayer(nn.Module):
    """Single Fourier layer: FFT -> multiply -> IFFT."""
    def __init__(self, d_model, modes=16):
        super().__init__()
        self.d_model = d_model
        self.modes = modes
        
        # Learnable weights in frequency domain
        self.weights = nn.Parameter(torch.randn(d_model, modes, dtype=torch.cfloat) * 0.02)
        
        # Local convolution for residual
        self.conv = nn.Conv1d(d_model, d_model, 3, padding=1)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        B, S, D = x.shape
        residual = x
        
        x = self.norm(x)
        
        # FFT along spatial dimension: (B, S, D) -> (B, S//2+1, D) complex
        x_ft = torch.fft.rfft(x, dim=1, norm='ortho')
        n_modes = min(self.modes, x_ft.shape[1])
        
        # Multiply in frequency domain: (B, n_modes, D) * (D, n_modes).T -> (B, n_modes, D)
        out_ft = torch.zeros_like(x_ft)
        out_ft[:, :n_modes, :] = x_ft[:, :n_modes, :] * self.weights[:, :n_modes].T.unsqueeze(0)
        
        # IFFT: (B, S//2+1, D) -> (B, S, D)
        x = torch.fft.irfft(out_ft, n=S, dim=1, norm='ortho')
        
        # Local convolution residual
        x_conv = self.conv(x.transpose(1, 2)).transpose(1, 2)
        
        return residual + x + x_conv


class FNO1D(nn.Module):
    """
    Fourier Neural Operator for 1D PDEs.
    Input: sequence of spatial points at time t
    Output: sequence of spatial points at time t+1
    """
    def __init__(self, d_model=64, modes=16, n_layers=1):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        
        self.input_proj = nn.Linear(1, d_model)
        
        self.layers = nn.ModuleList([
            FourierLayer(d_model, modes) for _ in range(n_layers)
        ])
        
        self.output_proj = nn.Linear(d_model, 1)
    
    def forward(self, x, return_attention=False):
        B, S = x.shape
        
        x = x.unsqueeze(-1)
        x = self.input_proj(x)
        
        for layer in self.layers:
            x = layer(x)
        
        out = self.output_proj(x).squeeze(-1)
        
        if return_attention:
            # FNO doesn't have attention, return None for compatibility
            dummy_attn = torch.zeros(self.n_layers, B, 1, S, S)
            return out, dummy_attn
        return out


class GINLayer(nn.Module):
    """
    Geometric Information Network layer (optimized).
    
    Attends to learned manifold points instead of input tokens.
    """
    def __init__(self, d_model, num_manifold_points=16, n_heads=4):
        super().__init__()
        self.d_model = d_model
        self.num_points = num_manifold_points
        self.n_heads = n_heads
        
        # Learned manifold points and values
        self.manifold = nn.Parameter(torch.randn(num_manifold_points, d_model) * 0.02)
        self.values = nn.Parameter(torch.randn(num_manifold_points, d_model) * 0.02)
        
        # Simple projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        
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
        
        # Simple attention: Q from input, K from manifold
        Q = self.q_proj(x)  # (B, S, D)
        K = self.k_proj(self.manifold)  # (K, D)
        
        # Attention scores: (B, S, D) @ (D, K) -> (B, S, K)
        attn = (Q @ K.T) / math.sqrt(D)
        attn = torch.softmax(attn, dim=-1)
        
        # Store for visualization (add fake head dim)
        self.last_attention = attn.unsqueeze(1).detach()  # (B, 1, S, K)
        
        # Gather values: (B, S, K) @ (K, D) -> (B, S, D)
        out = attn @ self.values
        
        x = residual + out
        x = x + self.ffn(self.norm2(x))
        return x


class GIN1D(nn.Module):
    """
    Geometric Information Network for 1D PDEs.
    
    Attends to learned manifold points rather than input tokens.
    """
    def __init__(self, d_model=64, num_manifold_points=16, n_heads=1, n_layers=1):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.num_manifold_points = num_manifold_points
        
        self.input_proj = nn.Linear(1, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, 256, d_model) * 0.02)
        
        self.layers = nn.ModuleList([
            GINLayer(d_model, num_manifold_points) for _ in range(n_layers)
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
            # Return attention to manifold: (n_layers, B, 1, S, K)
            all_attn = torch.stack([l.last_attention for l in self.layers])
            return out, all_attn
        return out


if __name__ == "__main__":
    x = torch.randn(4, 64)
    
    # Test Transformer
    model1 = PDETransformer(n_layers=1)
    out, attn = model1(x, return_attention=True)
    print(f"Transformer: Input {x.shape}, Output {out.shape}, Attn {attn.shape}")
    
    # Test FNO
    model2 = FNO1D(n_layers=1)
    out, _ = model2(x, return_attention=True)
    print(f"FNO: Input {x.shape}, Output {out.shape}")
    
    # Test GIN
    model3 = GIN1D(n_layers=1, num_manifold_points=16)
    out, attn = model3(x, return_attention=True)
    print(f"GIN: Input {x.shape}, Output {out.shape}, Attn {attn.shape} (S x K manifold)")

