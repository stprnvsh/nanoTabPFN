# Mosaic: Multi-Axis Attention Sharding

## Technical Specification v2.0

**Goal**: A thin coordination layer for models with **multiple attention axes**, built on top of existing attention sharding libraries (MagiAttention, ring-flash-attn).

**Target Use Case**: Models like nanoTabPFN with 4D tensors `(batch, rows, features, embed)` where attention operates over different axes with different sharding strategies.

**Key Insight**: Existing libraries (MagiAttention, ring-flash-attn, Ulysses) solve single-axis attention sharding well. The gap is **coordinating multiple attention axes** in the same model.

---

## 0. Existing Landscape (Leverage, Don't Reinvent)

| Library | Sharding | Use For |
|---------|----------|---------|
| [ring-flash-attn](https://github.com/zhuzilin/ring-flash-attention) | Row (ring) | Simple sequence parallelism |
| [MagiAttention](https://github.com/SandAI-org/MagiAttention) | 2D tiles | Advanced sharding with masks |
| DeepSpeed-Ulysses | Row (all-to-all) | Integration with DeepSpeed |
| PyTorch DTensor | General sharding | Tensor redistribution |

**Our value-add**: Coordinate these for multi-axis models.

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         User API                                 │
│  model = mosaic.parallelize(model, axes={1: "ring", 2: "local"})│
├─────────────────────────────────────────────────────────────────┤
│                    Mosaic Coordination Layer                     │
│  AxisRouter | ShardSpec per axis | Axis permutation             │
├─────────────────────────────────────────────────────────────────┤
│                   Existing Attention Backends                    │
│  ring-flash-attn | MagiAttention | local FlashAttention         │
├─────────────────────────────────────────────────────────────────┤
│                     PyTorch DTensor                              │
│  DeviceMesh | Shard/Replicate specs | redistribute              │
├─────────────────────────────────────────────────────────────────┤
│                    torch.distributed + NCCL                      │
└─────────────────────────────────────────────────────────────────┘
```

**Key difference from v1**: We're a coordination layer, not reimplementing ring attention.

---

## 2. Core Abstractions (Simplified)

### 2.1 MosaicContext

Thin wrapper around PyTorch's DeviceMesh.

```python
class MosaicContext:
    """Global context for multi-axis parallelism"""
    
    _instance = None
    
    def __init__(self, sp_size: int = 1):
        dist.init_process_group("nccl")
        
        # Use PyTorch's built-in DeviceMesh
        from torch.distributed.device_mesh import init_device_mesh
        self.mesh = init_device_mesh("cuda", (sp_size,), mesh_dim_names=("sp",))
        
        self.sp_size = sp_size
        self.sp_rank = dist.get_rank() % sp_size
        self.device = torch.device(f"cuda:{dist.get_rank()}")
    
    @classmethod
    def init(cls, **kwargs) -> "MosaicContext":
        cls._instance = cls(**kwargs)
        return cls._instance
    
    @classmethod
    def get(cls) -> "MosaicContext":
        return cls._instance

# Usage
ctx = mosaic.init(sp_size=4)
```

### 2.2 AxisSpec

Simple dataclass mapping axes to backends.

```python
@dataclass
class AxisSpec:
    """Which attention axis uses which backend"""
    axis: int              # Tensor axis to attend over
    backend: str           # "local" | "ring" | "magi"
    
    # For ring: ring-flash-attn
    # For magi: MagiAttention (2D tiles)
    # For local: standard F.scaled_dot_product_attention
```

**Note**: We do NOT reimplement ShardedTensor or DeviceMesh - use PyTorch DTensor if needed.

---

## 3. Communication (Delegated to Backends)

**We do NOT reimplement communication primitives.**

| Pattern | Handled By |
|---------|------------|
| Ring send/recv | ring-flash-attn |
| 2D tile communication | MagiAttention |
| All-to-all | DeepSpeed-Ulysses (if integrated) |
| AllGather/ReduceScatter | torch.distributed |

Mosaic's job is to **route to the right backend**, not implement communication.

---

## 4. Layer Implementation

### MultiAxisAttention (The Core Layer)

Routes to appropriate backend based on axis and strategy.

```python
class MultiAxisAttention(nn.Module):
    """
    Attention over arbitrary axis with backend selection.
    
    The key insight: different axes may need different sharding strategies.
    - Small axes (features, ~5-10) → local attention
    - Large axes (rows, ~150k) → ring attention
    """
    
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 attention_axis: int,
                 backend: str = "local"):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.attention_axis = attention_axis
        self.backend = backend
        
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Select backend
        if backend == "ring":
            from ring_flash_attn import ring_flash_attn_func
            self._attn_fn = self._ring_attention
        elif backend == "magi":
            # Optional MagiAttention integration
            self._attn_fn = self._magi_attention
        else:
            self._attn_fn = self._local_attention
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: N-D tensor, attention computed over attention_axis
        """
        # Step 1: Move attention axis to seq position (-2)
        x, inverse_perm = self._permute_to_seq(x)
        
        # Step 2: Flatten batch dims → (batch, seq, embed)
        batch_shape = x.shape[:-2]
        x = x.reshape(-1, x.shape[-2], x.shape[-1])
        
        # Step 3: Project Q, K, V
        B, S, _ = x.shape
        qkv = self.qkv_proj(x).reshape(B, S, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)  # (B, H, S, D)
        
        # Step 4: Attention via backend
        out = self._attn_fn(q, k, v)
        
        # Step 5: Reshape and project
        out = out.permute(0, 2, 1, 3).reshape(B, S, self.embed_dim)
        out = self.out_proj(out)
        
        # Step 6: Restore original shape
        out = out.reshape(*batch_shape, -1, self.embed_dim)
        out = out.permute(inverse_perm)
        
        return out
    
    def _permute_to_seq(self, x):
        """Move attention_axis to position -2, return inverse permutation"""
        ndim = x.ndim
        ax = self.attention_axis if self.attention_axis >= 0 else ndim + self.attention_axis
        
        if ax == ndim - 2:
            return x, list(range(ndim))
        
        # Build permutation: move ax to -2
        perm = [i for i in range(ndim) if i != ax]
        perm.insert(-1, ax)
        
        # Inverse permutation
        inv = [0] * ndim
        for i, p in enumerate(perm):
            inv[p] = i
        
        return x.permute(perm), inv
    
    def _local_attention(self, q, k, v):
        """Standard FlashAttention (no communication)"""
        return F.scaled_dot_product_attention(q, k, v)
    
    def _ring_attention(self, q, k, v):
        """Ring attention via ring-flash-attn"""
        from ring_flash_attn import ring_flash_attn_func
        ctx = MosaicContext.get()
        return ring_flash_attn_func(q, k, v, group=ctx.mesh.get_group("sp"))
    
    def _magi_attention(self, q, k, v):
        """2D tile attention via MagiAttention (if available)"""
        # Integration point for MagiAttention
        raise NotImplementedError("MagiAttention integration pending")
```

### No ParallelLinear / ParallelNorm

For sequence parallelism, standard `nn.Linear` and `nn.LayerNorm` work fine.

The sequence dimension is sharded, but:
- Linear operates on embed dim (not sharded) → no communication needed
- LayerNorm normalizes over embed dim (not sharded) → no communication needed

Only attention needs special handling because it operates **across the sharded axis**.

---

## 5. No Auto-Parallelization (Keep It Simple)

Auto-parallelization is out of scope for v1. Users explicitly specify:
- Which axis each attention operates on
- Which backend to use (local/ring/magi)

This keeps the codebase small (<1000 lines) and predictable.

---

## 6. Implementation Phases (Revised - Leveraging Existing Libraries)

### Phase 1: Foundation (Week 1)

**Goal**: Thin wrapper around existing tools.

| Component | Lines | Priority |
|-----------|-------|----------|
| `MosaicContext` (wrap DTensor DeviceMesh) | 80 | P0 |
| `AxisSpec` (which axis → which backend) | 60 | P0 |
| Backend detection (ring-flash-attn, MagiAttention) | 40 | P0 |
| Unit tests | 100 | P0 |

**Deliverable**: Can specify axis sharding strategies, detect available backends.

### Phase 2: MultiAxisAttention (Week 2-3)

**Goal**: Core layer that routes to appropriate backend per axis.

| Component | Lines | Priority |
|-----------|-------|----------|
| `MultiAxisAttention` | 200 | P0 |
| Axis permutation (move axis to seq position) | 80 | P0 |
| ring-flash-attn integration | 50 | P0 |
| MagiAttention integration (optional) | 50 | P1 |
| Multi-GPU tests | 150 | P0 |

**Deliverable**: `MultiAxisAttention` working with ring on nanoTabPFN.

### Phase 3: nanoTabPFN Example & Polish (Week 4)

**Goal**: Working end-to-end example.

| Component | Lines | Priority |
|-----------|-------|----------|
| nanoTabPFN parallel model | 100 | P0 |
| Sharded data loader | 80 | P0 |
| Benchmarks (1/2/4/8 GPU scaling) | 100 | P1 |
| Documentation | 150 | P1 |

**Deliverable**: nanoTabPFN runs on 4 GPUs with 150k rows.

**Total: ~1200 lines, 4 weeks** (down from 3000 lines, 10 weeks)

---

## 7. Directory Structure (Simplified)

```
mosaic/
├── __init__.py              # Public API
├── context.py               # MosaicContext (wraps DTensor DeviceMesh)
├── spec.py                  # AxisSpec (axis → sharding strategy)
├── attention.py             # MultiAxisAttention
├── backends/
│   ├── __init__.py
│   ├── ring.py              # ring-flash-attn wrapper
│   ├── magi.py              # MagiAttention wrapper (optional)
│   └── local.py             # Standard FlashAttention
├── examples/
│   └── nanotabpfn.py        # Full working example
└── tests/
    ├── test_attention.py
    └── test_nanotabpfn.py
```

**~800 lines total** (excluding tests)

---

## 8. API Design

### User-Facing API (Simplified)

```python
import mosaic

# Initialize (wraps DTensor DeviceMesh)
ctx = mosaic.init(sp_size=4)  # 4-way sequence parallel

# Build model with multi-axis attention
class NanoTabPFNParallel(nn.Module):
    def __init__(self):
        # Attention over features (axis 2) - small dim, keep local
        self.feature_attn = mosaic.MultiAxisAttention(
            embed_dim=96,
            num_heads=4,
            attention_axis=2,
            backend="local"
        )
        
        # Attention over rows (axis 1) - large dim, shard with ring
        self.row_attn = mosaic.MultiAxisAttention(
            embed_dim=96,
            num_heads=4,
            attention_axis=1,
            backend="ring"  # uses ring-flash-attn under the hood
        )
        
        self.mlp = nn.Sequential(
            nn.Linear(96, 384),
            nn.GELU(),
            nn.Linear(384, 96)
        )
    
    def forward(self, x):
        # x: (batch, rows_local, features, embed)
        # rows_local = total_rows / sp_size (sharded across GPUs)
        x = self.feature_attn(x) + x   # local, no communication
        x = self.row_attn(x) + x       # ring attention across GPUs
        x = self.mlp(x) + x
        return x

# Train - standard PyTorch
model = NanoTabPFNParallel().to(ctx.device)
for batch in sharded_dataloader:
    loss = model(batch).mean()
    loss.backward()
    optimizer.step()
```

### Key Simplification

No need for:
- `ColumnParallelLinear` / `RowParallelLinear` (use standard nn.Linear)
- `SequenceParallelLayerNorm` (use standard nn.LayerNorm)
- Manual communication primitives

Just specify `attention_axis` and `backend` - Mosaic handles the rest.

### Data Loading with Sharding

```python
class ShardedDataLoader:
    """Dataloader that shards along specified axis"""
    
    def __init__(self, 
                 data: torch.Tensor,
                 shard_axis: int,
                 batch_size: int):
        ctx = map.ParallelContext.get()
        self.sp_rank = ctx.mesh.rank("sp")
        self.sp_size = ctx.mesh.size("sp")
        
        # Each GPU gets 1/sp_size of shard_axis
        self.shard_size = data.shape[shard_axis] // self.sp_size
        
        # Slice data for this rank
        slices = [slice(None)] * data.ndim
        slices[shard_axis] = slice(
            self.sp_rank * self.shard_size,
            (self.sp_rank + 1) * self.shard_size
        )
        self.data = data[tuple(slices)]
```

---

## 9. Dependencies

### Required

| Package | Version | Purpose |
|---------|---------|---------|
| PyTorch | ≥2.0 | Base framework, DTensor |
| ring-flash-attn | ≥0.1 | Ring attention backend |
| flash-attn | ≥2.0 | FlashAttention for local attention |

### Optional (for advanced features)

| Package | Version | Purpose |
|---------|---------|---------|
| MagiAttention | latest | 2D tile sharding backend |
| triton | ≥2.0 | Custom kernels if needed |

### We Do NOT Reimplement

- Ring communication (use ring-flash-attn)
- 2D tile sharding (use MagiAttention)
- DeviceMesh (use PyTorch DTensor)
- AllGather/ReduceScatter (use torch.distributed)

---

## 10. Performance Targets

| Metric | Target |
|--------|--------|
| Mosaic overhead vs direct ring-flash-attn | <5% (just axis permutation) |
| Linear scaling to 4 GPUs | >85% efficiency |
| nanoTabPFN 150k rows, 4 GPUs | Fits in memory, trains |
| Memory per GPU reduction | ~Linear with GPU count |

**Note**: Performance is largely determined by the backend (ring-flash-attn, MagiAttention).
Mosaic's overhead is just tensor reshaping and axis permutation.

---

## 11. Testing Strategy

### Unit Tests (Single GPU)

- `test_attention.py`: Axis permutation correctness
- `test_backends.py`: Backend detection and fallback

### Integration Tests (Multi-GPU)

- `test_ring_backend.py`: ring-flash-attn integration works
- `test_nanotabpfn.py`: Full model forward/backward, gradient correctness

### Benchmarks

- `bench_scaling.py`: 1/2/4 GPU throughput comparison

---

## 12. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| ring-flash-attn API changes | Low | Medium | Pin version, wrap in abstraction |
| Axis permutation bugs | Medium | Medium | Extensive unit tests with random shapes |
| Backend not available | Medium | Low | Graceful fallback to local attention |
| MagiAttention integration complexity | Medium | Low | Make it optional, focus on ring first |

---

## 13. Success Criteria

1. **nanoTabPFN runs on 4 GPUs** with 150k rows per sample (currently OOMs on 1 GPU)
2. **<100 lines of user code change** to parallelize existing model
3. **<1000 lines of library code** (excluding tests)
4. **Working example** with benchmarks

---

## 14. Timeline Summary (Revised)

| Week | Milestone |
|------|-----------|
| 1 | Foundation: Context, AxisSpec, backend detection |
| 2-3 | MultiAxisAttention with ring-flash-attn |
| 4 | nanoTabPFN example, benchmarks, docs |

**Total: ~4 weeks to production-ready** (leveraging existing libraries)

---

## Appendix A: Comparison with Existing Frameworks

| Feature | Mosaic | MagiAttention | ring-flash-attn | DeepSpeed | NeMo |
|---------|--------|---------------|-----------------|-----------|------|
| Multi-axis attention | ✅ | ❌ | ❌ | ❌ | ❌ |
| Row sharding | ✅ (via backend) | ✅ | ✅ | ✅ | ✅ |
| 2D tile sharding | ✅ (via MagiAttn) | ✅ | ❌ | ❌ | ❌ |
| Arbitrary tensor dims | ✅ | ❌ | ❌ | ❌ | ❌ |
| PyTorch native | ✅ | ✅ | ✅ | ❌ | ❌ |
| Lightweight (<2k LOC) | ✅ | ❌ | ✅ | ❌ | ❌ |
| Custom architectures | ✅ | Partial | Partial | Partial | ❌ |

**Mosaic's unique value**: Coordinates multiple attention axes, each with its own sharding strategy.

---

## Appendix B: Why Not Just Use MagiAttention?

MagiAttention is excellent for single-attention-pattern models. But for nanoTabPFN:

```python
# nanoTabPFN has TWO attention patterns:
x = feature_attention(x)  # Attend over axis 2 (features) - small, keep local
x = row_attention(x)      # Attend over axis 1 (rows) - huge, need sharding
```

MagiAttention doesn't handle:
1. **Axis switching**: Moving between attention over different axes
2. **Mixed strategies**: Local for one axis, ring for another
3. **4D tensor reshaping**: (batch, rows, features, embed) ↔ (batch*X, seq, embed)

Mosaic provides this coordination layer on top of MagiAttention/ring-flash-attn.

