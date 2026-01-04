# From 77% CPU Idle to GPU-Bound: A Practical Guide to Eliminating Data Transfer Bottlenecks in PyTorch

When I first profiled our transformer training pipeline, I expected the GPU to be the bottleneck. Instead, I found the CPU spending **77% of its time waiting on data transfers**. The GPU was starving for data while the CPU struggled to feed it.

This is the story of how we identified and eliminated that bottleneck, achieving a **1.5x speedup** with data loading optimizations alone—and what we learned about the limits of these techniques.

## The Setup

We're training [nanoTabPFN](https://github.com/stprnvsh/nanoTabPFN), a transformer model for tabular data. The training data lives in HDF5 files: 30,000 samples, each with 5,000 rows and 5 features. Hardware: NVIDIA A100-SXM4-80GB.

The original training loop was straightforward:

```python
with h5py.File(filename, "r") as f:
    for step in range(num_steps):
        x = torch.from_numpy(f["X"][ptr:end])
        y = torch.from_numpy(f["y"][ptr:end])
        yield dict(x=x.to(device), y=y.to(device))
```

Simple. Correct. And, as it turns out, devastatingly slow.

## Step 1: Profile First, Optimize Later

Before touching any code, we instrumented the training loop with PyTorch's built-in profiler:

```python
from torch.profiler import profile, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
) as prof:
    train(model, prior)

print(prof.key_averages().table(sort_by="cpu_time_total"))
```

The results were shocking:

| Operation | CPU Time | % of Total |
|-----------|----------|------------|
| `cudaMemcpyAsync` | 44,084ms | **76.78%** |
| `cudaMalloc` | 7,081ms | 12.33% |
| `cudaLaunchKernel` | 645ms | 1.12% |
| `aten::bmm` | 180ms | 0.31% |

The GPU was doing matrix multiplications in milliseconds while the CPU spent **44 seconds** copying data. The training loop was CPU-bound on data transfers.

## Understanding the Problem

The `.to(device)` call in PyTorch is synchronous by default. Here's what happens:

1. **h5py reads from disk** → CPU memory (pageable)
2. **PyTorch allocates** → CPU staging buffer
3. **cudaMemcpy** → GPU memory (blocks until complete)
4. **GPU computes** → while CPU waits

The GPU sits idle during steps 1-3. With 5,000-row samples at float32, each batch transfer is ~120MB. That's 12GB of transfers over 100 steps, all sequential.

## Optimization 1: Pinned Memory + Non-blocking Transfers

The first fix: use page-locked (pinned) memory and async transfers.

```python
# Before: synchronous, pageable memory
x = torch.from_numpy(x_np).to(device)

# After: pinned memory, async transfer
x = torch.from_numpy(x_np).pin_memory().to(device, non_blocking=True)
```

**Why this works:**
- Pinned memory is DMA-accessible—the GPU can read it directly without CPU intervention
- `non_blocking=True` returns immediately; the transfer happens in the background
- Combined with proper synchronization, this overlaps transfer with compute

**Impact:** Reduced `cudaMemcpyAsync` CPU time from 44s to ~4s.

## Optimization 2: CUDA Streams for True Overlap

Non-blocking transfers alone aren't enough. By default, operations on the same stream are serialized. We need a separate stream for data transfer:

```python
class PriorDumpDataLoader:
    def __init__(self, ...):
        self.transfer_stream = torch.cuda.Stream()
    
    def __iter__(self):
        # Pre-fill buffer
        vram_buffer = [self._load_to_vram(f) for _ in range(prefetch)]
        
        for step in range(num_steps):
            batch = vram_buffer.pop(0)  # Already in VRAM
            
            # Prefetch next batch on separate stream
            with torch.cuda.stream(self.transfer_stream):
                next_batch = self._load_to_vram(f)
            vram_buffer.append(next_batch)
            
            # Sync before yielding
            torch.cuda.current_stream().wait_stream(self.transfer_stream)
            yield batch
```

This is **double buffering**: while the GPU processes batch N, the CPU+DMA engine load batch N+1. The GPU never waits for data.

## Optimization 3: GPU Direct Storage (GDS)

The ultimate optimization: bypass the CPU entirely. NVIDIA's GPUDirect Storage allows reading directly from disk to GPU memory.

```python
import kvikio
import cupy as cp

# Allocate GPU buffer
x_gpu = cp.empty((batch_size, seq_len, features), dtype=cp.float32)

# Direct read: NVMe → GPU (no CPU copy)
with kvikio.CuFile("data.bin", "r") as f:
    f.pread(x_gpu, file_offset=offset)

# Zero-copy to PyTorch
x = torch.as_tensor(x_gpu, device="cuda")
```

**Catch:** GDS requires raw binary files. HDF5 has headers and structure that require CPU parsing. We added automatic conversion:

```python
def convert_h5_to_raw(h5_filename):
    with h5py.File(h5_filename, "r") as f:
        X = f["X"][:].astype(np.float32)
        y = f["y"][:].astype(np.float32)
    X.tofile(f"{base}_X.bin")
    y.tofile(f"{base}_y.bin")
```

First run converts; subsequent runs use cached `.bin` files.

## Results

After all optimizations:

| Metric | Baseline | Optimized | Speedup |
|--------|----------|-----------|---------|
| Total time (100 steps) | 68.75s | 45.30s | **1.52x** |
| `cudaMemcpyAsync` CPU | 44,084ms | 268ms | **164x** |
| Steps/sec | 1.5 | 2.2 | **1.47x** |

Memory transfer overhead dropped from 77% to <1% of CPU time.

## The New Bottleneck

With data loading solved, the profile looks completely different:

| Operation | CPU Time | % of Total |
|-----------|----------|------------|
| `Command Buffer Full` | 23,450ms | **46.91%** |
| `cudaLaunchKernel` | 10,733ms | 21.47% |
| `cudaMalloc` | 5,607ms | 11.22% |

The GPU is now **saturated**. "Command Buffer Full" means the GPU can't keep up with kernel submissions. This is exactly what we want—the GPU is the bottleneck, not data loading.

The remaining bottleneck is attention computation (`aten::bmm` at 45% CUDA time). With 5,000-row sequences, attention's O(n²) memory scaling dominates. Flash Attention would be the next optimization.

## What We Learned

**1. Profile before optimizing.** Our intuition was wrong. The GPU wasn't the bottleneck; data transfer was.

**2. Async is not automatic.** `non_blocking=True` does nothing without proper stream management and synchronization.

**3. Pinned memory matters.** The difference between pageable and pinned memory is 10x+ for large transfers.

**4. GDS has constraints.** True zero-copy requires raw binary files, NVMe with GDS support, and proper alignment. HDF5/Parquet/etc. still need CPU parsing.

**5. Optimizations compound.** Pinned memory + streams + prefetching together achieved more than any single technique.

**6. Know when to stop.** Once you're GPU-bound, data loading optimizations won't help. The next step is model architecture changes (Flash Attention, gradient checkpointing).

## Code

All code is available at [github.com/stprnvsh/nanoTabPFN](https://github.com/stprnvsh/nanoTabPFN).

```bash
# Baseline
python train.py --profile --steps=100 --batch-size=6

# Optimized with GDS
python train_optimized.py --gds-bin --batch-size=4 --steps=200

# With Flash Attention
python train_optimized.py --flash --gds-bin --batch-size=8 --steps=200
```

## Summary

| Technique | What it does | When to use |
|-----------|--------------|-------------|
| `pin_memory()` | Page-locked CPU memory | Always for GPU training |
| `non_blocking=True` | Async H2D transfer | With CUDA streams |
| CUDA Streams | Parallel transfer/compute | Large batch sizes |
| Double buffering | Prefetch next batch | I/O-bound workloads |
| GDS (kvikio) | Disk → GPU direct | Large sequential reads |
| Flash Attention | O(n) attention memory | Long sequences |

The GPU is expensive. Don't let it wait for data.

---

*Thanks to the PyTorch and RAPIDS teams for excellent profiling and GDS tooling.*

