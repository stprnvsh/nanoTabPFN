"""
Optimized model with Flash Attention for O(n) memory instead of O(n²).
Uses torch.nn.functional.scaled_dot_product_attention which auto-selects:
- Flash Attention (if available)
- Memory-efficient attention (xformers-style)
- Math fallback
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class NanoTabPFNModelOptimized(nn.Module):
    def __init__(self, embedding_size: int, num_attention_heads: int, mlp_hidden_size: int, num_layers: int, num_outputs: int):
        super().__init__()
        self.feature_encoder = FeatureEncoder(embedding_size)
        self.target_encoder = TargetEncoder(embedding_size)
        self.transformer_blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.transformer_blocks.append(TransformerEncoderLayerOptimized(embedding_size, num_attention_heads, mlp_hidden_size))
        self.decoder = Decoder(embedding_size, mlp_hidden_size, num_outputs)

    def forward(self, src: tuple[torch.Tensor, torch.Tensor], train_test_split_index: int) -> torch.Tensor:
        x_src, y_src = src
        if len(y_src.shape) < len(x_src.shape):
            y_src = y_src.unsqueeze(-1)
        x_src = self.feature_encoder(x_src, train_test_split_index)
        num_rows = x_src.shape[1]
        y_src = self.target_encoder(y_src, num_rows)
        src = torch.cat([x_src, y_src], 2)
        for block in self.transformer_blocks:
            src = block(src, train_test_split_index=train_test_split_index)
        output = src[:, train_test_split_index:, -1, :]
        output = self.decoder(output)
        return output


class FeatureEncoder(nn.Module):
    def __init__(self, embedding_size: int):
        super().__init__()
        self.linear_layer = nn.Linear(1, embedding_size)

    def forward(self, x: torch.Tensor, train_test_split_index: int) -> torch.Tensor:
        x = x.unsqueeze(-1)
        mean = torch.mean(x[:, :train_test_split_index], dim=1, keepdims=True)
        std = torch.std(x[:, :train_test_split_index], dim=1, keepdims=True) + 1e-20
        x = (x - mean) / std
        x = torch.clip(x, min=-100, max=100)
        return self.linear_layer(x)


class TargetEncoder(nn.Module):
    def __init__(self, embedding_size: int):
        super().__init__()
        self.linear_layer = nn.Linear(1, embedding_size)

    def forward(self, y_train: torch.Tensor, num_rows: int) -> torch.Tensor:
        mean = torch.mean(y_train, dim=1, keepdim=True)
        padding = mean.repeat(1, num_rows - y_train.shape[1], 1)
        y = torch.cat([y_train, padding], dim=1)
        y = y.unsqueeze(-1)
        return self.linear_layer(y)


class FlashMultiheadAttention(nn.Module):
    """
    Drop-in replacement for nn.MultiheadAttention using scaled_dot_product_attention.
    Uses Flash Attention when available (PyTorch 2.0+, Ampere+ GPU).
    """
    def __init__(self, embed_dim: int, num_heads: int, batch_first: bool = True, device=None, dtype=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(embed_dim, embed_dim, device=device, dtype=dtype)
        self.k_proj = nn.Linear(embed_dim, embed_dim, device=device, dtype=dtype)
        self.v_proj = nn.Linear(embed_dim, embed_dim, device=device, dtype=dtype)
        self.out_proj = nn.Linear(embed_dim, embed_dim, device=device, dtype=dtype)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                attn_mask: torch.Tensor = None, is_causal: bool = False) -> tuple[torch.Tensor, None]:
        batch_size, seq_len, _ = query.shape
        
        # Project Q, K, V
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # Reshape for multi-head attention: (batch, seq, heads, head_dim) -> (batch, heads, seq, head_dim)
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Flash Attention via scaled_dot_product_attention
        # Automatically uses flash attention if available, else memory-efficient, else math
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            is_causal=is_causal,
            scale=1.0 / (self.head_dim ** 0.5)
        )
        
        # Reshape back: (batch, heads, seq, head_dim) -> (batch, seq, embed_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        
        return self.out_proj(attn_output), None


class TransformerEncoderLayerOptimized(nn.Module):
    """
    Transformer layer using Flash Attention for O(n) memory.
    """
    def __init__(self, embedding_size: int, nhead: int, mlp_hidden_size: int,
                 layer_norm_eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.self_attention_between_datapoints = FlashMultiheadAttention(embedding_size, nhead, device=device, dtype=dtype)
        self.self_attention_between_features = FlashMultiheadAttention(embedding_size, nhead, device=device, dtype=dtype)
        
        self.linear1 = nn.Linear(embedding_size, mlp_hidden_size, device=device, dtype=dtype)
        self.linear2 = nn.Linear(mlp_hidden_size, embedding_size, device=device, dtype=dtype)
        
        self.norm1 = nn.LayerNorm(embedding_size, eps=layer_norm_eps, device=device, dtype=dtype)
        self.norm2 = nn.LayerNorm(embedding_size, eps=layer_norm_eps, device=device, dtype=dtype)
        self.norm3 = nn.LayerNorm(embedding_size, eps=layer_norm_eps, device=device, dtype=dtype)

    def forward(self, src: torch.Tensor, train_test_split_index: int) -> torch.Tensor:
        batch_size, rows_size, col_size, embedding_size = src.shape
        
        # Attention between features
        src = src.reshape(batch_size * rows_size, col_size, embedding_size)
        src = self.self_attention_between_features(src, src, src)[0] + src
        src = src.reshape(batch_size, rows_size, col_size, embedding_size)
        src = self.norm1(src)
        
        # Attention between datapoints
        src = src.transpose(1, 2)
        src = src.reshape(batch_size * col_size, rows_size, embedding_size)
        
        # Training data attends to itself
        src_left = self.self_attention_between_datapoints(
            src[:, :train_test_split_index],
            src[:, :train_test_split_index],
            src[:, :train_test_split_index]
        )[0]
        
        # Test data attends to training data (cross-attention)
        src_right = self.self_attention_between_datapoints(
            src[:, train_test_split_index:],
            src[:, :train_test_split_index],
            src[:, :train_test_split_index]
        )[0]
        
        src = torch.cat([src_left, src_right], dim=1) + src
        src = src.reshape(batch_size, col_size, rows_size, embedding_size)
        src = src.transpose(2, 1)
        src = self.norm2(src)
        
        # MLP
        src = self.linear2(F.gelu(self.linear1(src))) + src
        src = self.norm3(src)
        return src


class Decoder(nn.Module):
    def __init__(self, embedding_size: int, mlp_hidden_size: int, num_outputs: int):
        super().__init__()
        self.linear1 = nn.Linear(embedding_size, mlp_hidden_size)
        self.linear2 = nn.Linear(mlp_hidden_size, num_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(F.gelu(self.linear1(x)))


class NanoTabPFNClassifier:
    """scikit-learn like interface"""
    def __init__(self, model, device: torch.device):
        self.model = model.to(device)
        self.device = device

    def fit(self, X_train: np.array, y_train: np.array):
        self.X_train = X_train
        self.y_train = y_train
        self.num_classes = max(set(y_train)) + 1

    def predict_proba(self, X_test: np.array) -> np.array:
        x = np.concatenate((self.X_train, X_test))
        y = self.y_train
        with torch.no_grad():
            x = torch.from_numpy(x).unsqueeze(0).to(torch.float).to(self.device)
            y = torch.from_numpy(y).unsqueeze(0).to(torch.float).to(self.device)
            out = self.model((x, y), train_test_split_index=len(self.X_train)).squeeze(0)
            out = out[:, :self.num_classes]
            probabilities = F.softmax(out, dim=1)
            return probabilities.to("cpu").numpy()

    def predict(self, X_test: np.array) -> np.array:
        return self.predict_proba(X_test).argmax(axis=1)

