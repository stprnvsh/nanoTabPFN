"""
Optimized model with Flash Attention for O(n) memory instead of O(n²).
Uses torch.nn.functional.scaled_dot_product_attention which auto-selects:
- Flash Attention (if available)
- Memory-efficient attention (xformers-style)
- Math fallback

Also supports gradient checkpointing to further reduce memory.
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint


class NanoTabPFNModelOptimized(nn.Module):
    def __init__(self, embedding_size: int, num_attention_heads: int, mlp_hidden_size: int, 
                 num_layers: int, num_outputs: int, use_checkpointing: bool = False,
                 num_kv_heads: int = None):
        super().__init__()
        self.feature_encoder = FeatureEncoder(embedding_size)
        self.target_encoder = TargetEncoder(embedding_size)
        self.transformer_blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.transformer_blocks.append(
                TransformerEncoderLayerOptimized(
                    embedding_size, num_attention_heads, mlp_hidden_size,
                    num_kv_heads=num_kv_heads
                )
            )
        self.decoder = Decoder(embedding_size, mlp_hidden_size, num_outputs)
        self.use_checkpointing = use_checkpointing

    def forward(self, src: tuple[torch.Tensor, torch.Tensor], train_test_split_index: int,
                save_peak_mem_factor: int = None) -> torch.Tensor:
        """
        Args:
            save_peak_mem_factor: If set, chunks along batch dimension to reduce peak memory.
                                  Only works during inference (no gradients).
        """
        x_src, y_src = src
        if len(y_src.shape) < len(x_src.shape):
            y_src = y_src.unsqueeze(-1)
        x_src = self.feature_encoder(x_src, train_test_split_index)
        num_rows = x_src.shape[1]
        y_src = self.target_encoder(y_src, num_rows)
        src = torch.cat([x_src, y_src], 2)
        for block in self.transformer_blocks:
            if self.use_checkpointing and self.training:
                src = checkpoint(block, src, train_test_split_index, use_reentrant=False)
            else:
                src = block(src, train_test_split_index=train_test_split_index, 
                           save_peak_mem_factor=save_peak_mem_factor)
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
    Supports GQA (Grouped Query Attention) and batch chunking for memory efficiency.
    """
    def __init__(self, embed_dim: int, num_heads: int, num_kv_heads: int = None, 
                 batch_first: bool = True, device=None, dtype=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        assert num_heads % self.num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"
        self.num_queries_per_kv = num_heads // self.num_kv_heads
        
        # Q projection: full number of heads
        self.q_proj = nn.Linear(embed_dim, embed_dim, device=device, dtype=dtype)
        # K, V projections: fewer heads (GQA)
        kv_dim = self.num_kv_heads * self.head_dim
        self.k_proj = nn.Linear(embed_dim, kv_dim, device=device, dtype=dtype)
        self.v_proj = nn.Linear(embed_dim, kv_dim, device=device, dtype=dtype)
        self.out_proj = nn.Linear(embed_dim, embed_dim, device=device, dtype=dtype)
        
        # Check if PyTorch supports GQA
        self.use_gqa = self._check_gqa_support()
    
    def _check_gqa_support(self) -> bool:
        """Check if PyTorch supports enable_gqa parameter."""
        if not torch.cuda.is_available():
            return False
        torch_version = torch.__version__.split(".")
        torch_major, torch_minor = int(torch_version[0]), int(torch_version[1])
        if torch_major > 2 or (torch_major == 2 and torch_minor >= 5):
            device = torch.cuda.current_device()
            compute_capability = torch.cuda.get_device_capability(device)
            return compute_capability[0] >= 8  # Ampere+
        return False
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                attn_mask: torch.Tensor = None, is_causal: bool = False,
                save_peak_mem_factor: int = None) -> tuple[torch.Tensor, None]:
        """
        Args:
            save_peak_mem_factor: If set, chunks along batch dimension to reduce peak memory.
                                  Only works during inference (no gradients).
        """
        batch_size, seq_len, _ = query.shape
        
        # Project Q, K, V
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H_q, L, D)
        k = k.view(batch_size, -1, self.num_kv_heads, self.head_dim).transpose(1, 2)    # (B, H_kv, L, D)
        v = v.view(batch_size, -1, self.num_kv_heads, self.head_dim).transpose(1, 2)    # (B, H_kv, L, D)
        
        # Batch chunking: process in smaller chunks along batch dimension
        if save_peak_mem_factor is not None and not self.training and not q.requires_grad:
            assert save_peak_mem_factor > 1, "save_peak_mem_factor must be > 1"
            chunk_size = (batch_size + save_peak_mem_factor - 1) // save_peak_mem_factor
            attn_outputs = []
            
            for i in range(0, batch_size, chunk_size):
                end_idx = min(i + chunk_size, batch_size)
                q_chunk = q[i:end_idx]
                k_chunk = k[i:end_idx]
                v_chunk = v[i:end_idx]
                mask_chunk = attn_mask[i:end_idx] if attn_mask is not None else None
                
                attn_chunk = self._compute_attention(q_chunk, k_chunk, v_chunk, mask_chunk, is_causal)
                attn_outputs.append(attn_chunk)
            
            attn_output = torch.cat(attn_outputs, dim=0)
        else:
            attn_output = self._compute_attention(q, k, v, attn_mask, is_causal)
        
        # Reshape back: (batch, heads, seq, head_dim) -> (batch, seq, embed_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        
        return self.out_proj(attn_output), None
    
    def _compute_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                           attn_mask: torch.Tensor = None, is_causal: bool = False) -> torch.Tensor:
        """Core attention computation with GQA support."""
        extra_inputs = {}
        if self.use_gqa and self.num_kv_heads < self.num_heads:
            # PyTorch 2.5+ GQA support
            extra_inputs["enable_gqa"] = True
        else:
            # Manual broadcasting: expand K,V to match Q heads
            if self.num_kv_heads < self.num_heads:
                k = k.repeat_interleave(self.num_queries_per_kv, dim=1)
                v = v.repeat_interleave(self.num_queries_per_kv, dim=1)
        
        # Flash Attention via scaled_dot_product_attention
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            is_causal=is_causal,
            scale=1.0 / (self.head_dim ** 0.5),
            **extra_inputs
        )
        
        return attn_output


class TransformerEncoderLayerOptimized(nn.Module):
    """
    Transformer layer using Flash Attention for O(n) memory.
    Supports GQA and batch chunking.
    """
    def __init__(self, embedding_size: int, nhead: int, mlp_hidden_size: int,
                 layer_norm_eps: float = 1e-5, num_kv_heads: int = None, device=None, dtype=None):
        super().__init__()
        self.self_attention_between_datapoints = FlashMultiheadAttention(
            embedding_size, nhead, num_kv_heads=num_kv_heads, device=device, dtype=dtype
        )
        self.self_attention_between_features = FlashMultiheadAttention(
            embedding_size, nhead, num_kv_heads=num_kv_heads, device=device, dtype=dtype
        )
        
        self.linear1 = nn.Linear(embedding_size, mlp_hidden_size, device=device, dtype=dtype)
        self.linear2 = nn.Linear(mlp_hidden_size, embedding_size, device=device, dtype=dtype)
        
        self.norm1 = nn.LayerNorm(embedding_size, eps=layer_norm_eps, device=device, dtype=dtype)
        self.norm2 = nn.LayerNorm(embedding_size, eps=layer_norm_eps, device=device, dtype=dtype)
        self.norm3 = nn.LayerNorm(embedding_size, eps=layer_norm_eps, device=device, dtype=dtype)

    def forward(self, src: torch.Tensor, train_test_split_index: int,
                save_peak_mem_factor: int = None) -> torch.Tensor:
        batch_size, rows_size, col_size, embedding_size = src.shape
        
        # Attention between features
        src = src.reshape(batch_size * rows_size, col_size, embedding_size)
        src = self.self_attention_between_features(
            src, src, src, save_peak_mem_factor=save_peak_mem_factor
        )[0] + src
        src = src.reshape(batch_size, rows_size, col_size, embedding_size)
        src = self.norm1(src)
        
        # Attention between datapoints
        src = src.transpose(1, 2)
        src = src.reshape(batch_size * col_size, rows_size, embedding_size)
        
        # Training data attends to itself
        src_left = self.self_attention_between_datapoints(
            src[:, :train_test_split_index],
            src[:, :train_test_split_index],
            src[:, :train_test_split_index],
            save_peak_mem_factor=save_peak_mem_factor
        )[0]
        
        # Test data attends to training data (cross-attention)
        src_right = self.self_attention_between_datapoints(
            src[:, train_test_split_index:],
            src[:, :train_test_split_index],
            src[:, :train_test_split_index],
            save_peak_mem_factor=save_peak_mem_factor
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

