"""
Optimized training with GPU utilization improvements:
1. torch.compile() - fuses kernels, reduces launch overhead
2. TF32 - faster matmuls on Ampere+
3. Larger batch size - better GPU occupancy
4. Non-blocking transfers - overlap H2D with compute
5. Pinned memory - faster CPU->GPU transfers
6. CUDA streams - parallel data transfer and compute
7. kvikio/GDS - GPU Direct Storage (disk -> VRAM, bypass CPU)
"""
import random
import time
from collections import defaultdict
import queue
import threading
import h5py
import numpy as np
import schedulefree
import torch
from model import NanoTabPFNClassifier, NanoTabPFNModel
from model_optimized import NanoTabPFNModelOptimized, NanoTabPFNClassifier as NanoTabPFNClassifierOptimized
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader


# =============================================================================
# MODEL PROFILING UTILITIES
# =============================================================================

class LayerProfiler:
    """Profiles individual layers with timing and memory tracking."""
    
    def __init__(self):
        self.timings = defaultdict(list)
        self.memory = defaultdict(list)
        self.shapes = {}
        self.hooks = []
        self.enabled = False
    
    def _get_mem_mb(self):
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1e6
        return 0
    
    def attach(self, model):
        """Attach timing hooks to all layers."""
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                self.hooks.append(module.register_forward_pre_hook(
                    self._make_pre_hook(name)))
                self.hooks.append(module.register_forward_hook(
                    self._make_post_hook(name)))
        return self
    
    def _make_pre_hook(self, name):
        def hook(module, input):
            if not self.enabled:
                return
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self._start_time = time.perf_counter()
            self._start_mem = self._get_mem_mb()
            # Record input shape
            if input and hasattr(input[0], 'shape'):
                self.shapes[name] = tuple(input[0].shape)
        return hook
    
    def _make_post_hook(self, name):
        def hook(module, input, output):
            if not self.enabled:
                return
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = (time.perf_counter() - self._start_time) * 1000  # ms
            mem_delta = self._get_mem_mb() - self._start_mem
            self.timings[name].append(elapsed)
            self.memory[name].append(mem_delta)
        return hook
    
    def enable(self):
        self.enabled = True
    
    def disable(self):
        self.enabled = False
    
    def reset(self):
        self.timings.clear()
        self.memory.clear()
        self.shapes.clear()
    
    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()
    
    def summary(self, top_k=20):
        """Print summary sorted by total time."""
        rows = []
        for name in self.timings:
            times = self.timings[name]
            mems = self.memory[name]
            shape = self.shapes.get(name, ())
            rows.append({
                'name': name,
                'calls': len(times),
                'total_ms': sum(times),
                'avg_ms': sum(times) / len(times) if times else 0,
                'mem_mb': sum(mems) / len(mems) if mems else 0,
                'shape': shape
            })
        
        rows.sort(key=lambda x: x['total_ms'], reverse=True)
        
        print(f"\n{'='*90}")
        print(f"{'LAYER PROFILING SUMMARY':^90}")
        print(f"{'='*90}")
        print(f"{'Layer':<45} {'Calls':>6} {'Total(ms)':>10} {'Avg(ms)':>9} {'Mem(MB)':>9}")
        print(f"{'-'*90}")
        
        total_time = sum(r['total_ms'] for r in rows)
        for r in rows[:top_k]:
            pct = 100 * r['total_ms'] / total_time if total_time > 0 else 0
            print(f"{r['name'][:44]:<45} {r['calls']:>6} {r['total_ms']:>9.2f} {r['avg_ms']:>9.3f} {r['mem_mb']:>+8.1f}")
        
        print(f"{'-'*90}")
        print(f"{'TOTAL':<45} {'':<6} {total_time:>9.2f}ms")
        return rows


class ModelProfiler:
    """High-level profiler for the full training step."""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.step_times = defaultdict(list)
        self.memory_snapshots = []
    
    def _sync(self):
        if self.device == 'cuda':
            torch.cuda.synchronize()
    
    def _mem_stats(self):
        if self.device != 'cuda':
            return {}
        return {
            'allocated_mb': torch.cuda.memory_allocated() / 1e6,
            'reserved_mb': torch.cuda.memory_reserved() / 1e6,
            'max_allocated_mb': torch.cuda.max_memory_allocated() / 1e6,
        }
    
    def profile_step(self, model, data, targets, train_test_split_index, criterion, optimizer):
        """Profile a single training step with detailed breakdown."""
        times = {}
        mems = {}
        
        # Reset peak memory
        if self.device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        
        mems['start'] = self._mem_stats()
        
        # Forward pass
        self._sync()
        t0 = time.perf_counter()
        output = model(data, train_test_split_index=train_test_split_index)
        output = output.view(-1, output.shape[-1])
        self._sync()
        times['forward'] = (time.perf_counter() - t0) * 1000
        mems['after_forward'] = self._mem_stats()
        
        # Loss computation
        t0 = time.perf_counter()
        loss = criterion(output, targets)
        self._sync()
        times['loss'] = (time.perf_counter() - t0) * 1000
        mems['after_loss'] = self._mem_stats()
        
        # Backward pass
        t0 = time.perf_counter()
        loss.backward()
        self._sync()
        times['backward'] = (time.perf_counter() - t0) * 1000
        mems['after_backward'] = self._mem_stats()
        
        # Gradient clipping
        t0 = time.perf_counter()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        self._sync()
        times['grad_clip'] = (time.perf_counter() - t0) * 1000
        
        # Optimizer step
        t0 = time.perf_counter()
        optimizer.step()
        self._sync()
        times['optimizer'] = (time.perf_counter() - t0) * 1000
        
        # Zero grad
        t0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        self._sync()
        times['zero_grad'] = (time.perf_counter() - t0) * 1000
        mems['end'] = self._mem_stats()
        
        # Record
        for k, v in times.items():
            self.step_times[k].append(v)
        self.memory_snapshots.append(mems)
        
        return loss, times, mems
    
    def summary(self):
        """Print profiling summary."""
        print(f"\n{'='*70}")
        print(f"{'TRAINING STEP BREAKDOWN':^70}")
        print(f"{'='*70}")
        print(f"{'Phase':<20} {'Total(ms)':>12} {'Avg(ms)':>12} {'%':>8}")
        print(f"{'-'*70}")
        
        total = sum(sum(v) for v in self.step_times.values())
        for phase in ['forward', 'loss', 'backward', 'grad_clip', 'optimizer', 'zero_grad']:
            times = self.step_times.get(phase, [])
            if times:
                t = sum(times)
                pct = 100 * t / total if total > 0 else 0
                print(f"{phase:<20} {t:>12.2f} {t/len(times):>12.3f} {pct:>7.1f}%")
        
        print(f"{'-'*70}")
        n_steps = len(self.step_times.get('forward', [1]))
        print(f"{'TOTAL':<20} {total:>12.2f} {total/n_steps:>12.3f}ms/step")
        
        # Memory summary
        if self.memory_snapshots:
            last = self.memory_snapshots[-1]
            print(f"\n{'MEMORY (final step)':<70}")
            print(f"{'-'*70}")
            print(f"  After forward:  {last.get('after_forward', {}).get('allocated_mb', 0):>8.1f} MB allocated")
            print(f"  After backward: {last.get('after_backward', {}).get('allocated_mb', 0):>8.1f} MB allocated")
            print(f"  Peak:           {last.get('after_backward', {}).get('max_allocated_mb', 0):>8.1f} MB")


def profile_attention_ops(model, x, y, train_test_split_index, device='cuda'):
    """Profile attention operations specifically."""
    from torch.profiler import profile, ProfilerActivity, record_function
    
    print(f"\n{'='*80}")
    print(f"{'ATTENTION OPERATION BREAKDOWN':^80}")
    print(f"{'='*80}")
    
    data = (x, y[:, :train_test_split_index])
    
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA] if device == 'cuda' else [ProfilerActivity.CPU],
        record_shapes=True,
        with_stack=False,
    ) as prof:
        with record_function("model_forward"):
            _ = model(data, train_test_split_index=train_test_split_index)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # Filter and display attention-related ops
    print(f"\n{'Operation':<40} {'CUDA(ms)':>10} {'CPU(ms)':>10} {'Calls':>8} {'Shape'}")
    print(f"{'-'*80}")
    
    attn_ops = ['bmm', 'matmul', 'softmax', 'scaled_dot', 'attention', 'linear', 'layer_norm', 'gelu']
    
    for evt in prof.key_averages(group_by_input_shape=True):
        if any(op in evt.key.lower() for op in attn_ops):
            cuda_t = getattr(evt, 'cuda_time_total', 0) / 1000  # to ms
            cpu_t = evt.cpu_time_total / 1000
            shapes = str(evt.input_shapes)[:30] if evt.input_shapes else ''
            print(f"{evt.key[:39]:<40} {cuda_t:>10.3f} {cpu_t:>10.3f} {evt.count:>8} {shapes}")
    
    # Compute breakdown
    print(f"\n{'COMPUTE CATEGORY SUMMARY':^80}")
    print(f"{'-'*80}")
    
    categories = {
        'Attention (bmm/matmul)': ['bmm', 'matmul', 'scaled_dot'],
        'Softmax': ['softmax'],
        'Linear layers': ['linear', 'addmm'],
        'Normalization': ['layer_norm', 'native_layer_norm'],
        'Activations': ['gelu', 'relu'],
        'Memory ops': ['copy_', 'to', 'contiguous', 'reshape', 'view'],
    }
    
    cat_times = defaultdict(float)
    for evt in prof.key_averages():
        cuda_t = getattr(evt, 'cuda_time_total', 0) / 1000
        for cat, keywords in categories.items():
            if any(kw in evt.key.lower() for kw in keywords):
                cat_times[cat] += cuda_t
                break
    
    total = sum(cat_times.values())
    for cat, t in sorted(cat_times.items(), key=lambda x: -x[1]):
        pct = 100 * t / total if total > 0 else 0
        print(f"  {cat:<30} {t:>10.2f}ms  ({pct:>5.1f}%)")
    
    return prof


def set_randomness_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_randomness_seed(0)

def get_default_device():
    device = "cpu"
    if torch.backends.mps.is_available(): device = "mps"
    if torch.cuda.is_available(): device = "cuda"
    return device

# Enable TF32 for Ampere+ GPUs (2x faster matmuls)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True  # auto-tune convolutions

datasets = []
datasets.append(train_test_split(*load_breast_cancer(return_X_y=True), test_size=0.5, random_state=0))

def eval(classifier):
    scores = {"roc_auc": 0, "acc": 0, "balanced_acc": 0}
    for X_train, X_test, y_train, y_test in datasets:
        classifier.fit(X_train, y_train)
        prob = classifier.predict_proba(X_test)
        pred = prob.argmax(axis=1)
        if prob.shape[1] == 2:
            prob = prob[:, :1]
        scores["roc_auc"] += float(roc_auc_score(y_test, prob, multi_class="ovr"))
        scores["acc"] += float(accuracy_score(y_test, pred))
        scores["balanced_acc"] += float(balanced_accuracy_score(y_test, pred))
    return {k: v / len(datasets) for k, v in scores.items()}


def train(model: NanoTabPFNModel, prior: DataLoader,
          lr: float = 1e-4, device: torch.device = None, steps_per_eval=10, eval_func=None):
    if not device:
        device = get_default_device()
    model.to(device)
    
    optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=lr, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss()

    model.train()
    optimizer.train()

    train_time = 0
    eval_history = []
    
    
    try:
        for step, full_data in enumerate(prior):
            step_start_time = time.time()
            train_test_split_index = full_data["train_test_split_index"]
            
            # Data already in VRAM from prefetching dataloader
            x = full_data["x"]
            y_full = full_data["y"]
            
            data = (x, y_full[:, :train_test_split_index])
            targets = y_full[:, train_test_split_index:].reshape(-1).long()

            output = model(data, train_test_split_index=train_test_split_index)
            output = output.view(-1, output.shape[-1])

            loss = criterion(output, targets)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)  # Faster than zero_grad()
            
            step_train_duration = time.time() - step_start_time
            train_time += step_train_duration

            if step % steps_per_eval == steps_per_eval - 1:
                total_loss = loss.item()
                
                if eval_func is not None:
                    model.eval()
                    optimizer.eval()
                    classifier = NanoTabPFNClassifier(model, device)
                    scores = eval_func(classifier)
                    eval_history.append((train_time, scores))
                    score_str = " | ".join([f"{k} {v:7.4f}" for k, v in scores.items()])
                    print(f"time {train_time:7.1f}s | loss {total_loss:7.4f} | {score_str}")
                    model.train()
                    optimizer.train()
                else:
                    print(f"time {train_time:7.1f}s | loss {total_loss:7.4f}")
    except KeyboardInterrupt:
        pass

    return model, eval_history


class PriorDumpDataLoader(DataLoader):
    """
    Double-buffered dataloader with VRAM prefetching.
    
    While GPU computes on buffer A, we:
    1. Read next batch from disk to pinned CPU memory
    2. Transfer to VRAM buffer B via separate CUDA stream
    
    This overlaps: disk I/O, H2D transfer, and GPU compute.
    """
    def __init__(self, filename, num_steps=None, batch_size=32, device=None, num_prefetch=2):
        self.filename = filename
        self.batch_size = batch_size
        self.device = device if device else get_default_device()
        self.pointer = 0
        self.num_prefetch = num_prefetch
        
        with h5py.File(self.filename, "r") as f:
            self.max_num_classes = f["max_num_classes"][0]
            self.total_samples = f["X"].shape[0]
        
        # If num_steps not provided, use full dataset
        if num_steps is None:
            self.num_steps = self.total_samples // batch_size
        else:
            self.num_steps = num_steps
        
        print(f"DataLoader: {self.total_samples:,} samples, {self.num_steps} steps, batch_size={batch_size}")
        
        # Create dedicated transfer stream
        if self.device == "cuda":
            self.transfer_stream = torch.cuda.Stream()
        
    def __iter__(self):
        with h5py.File(self.filename, "r") as f:
            # No prefetch: simple synchronous loading
            if self.num_prefetch == 0:
                for step in range(self.num_steps):
                    yield self._load_to_vram(f)
                return
            
            # Pre-fill VRAM buffer with first N batches
            vram_buffer = []
            for _ in range(min(self.num_prefetch, self.num_steps)):
                batch = self._load_to_vram(f)
                vram_buffer.append(batch)
            
            for step in range(self.num_steps):
                # Pop batch from buffer (already in VRAM)
                batch = vram_buffer.pop(0)
                
                # Prefetch next batch to VRAM while GPU computes
                remaining = self.num_steps - step - 1
                if remaining > 0 and self.device == "cuda":
                    with torch.cuda.stream(self.transfer_stream):
                        next_batch = self._load_to_vram(f)
                    vram_buffer.append(next_batch)
                elif remaining > 0:
                    vram_buffer.append(self._load_to_vram(f))
                
                # Sync transfer stream before yielding
                if self.device == "cuda":
                    torch.cuda.current_stream().wait_stream(self.transfer_stream)
                
                yield batch

    def _load_to_vram(self, f):
        """Load batch: disk -> pinned CPU -> VRAM"""
        end = self.pointer + self.batch_size
        num_features = f["num_features"][self.pointer:end].max()
        num_datapoints_batch = f["num_datapoints"][self.pointer:end]
        max_seq_in_batch = int(num_datapoints_batch.max())
        
        # Disk -> numpy
        x_np = f["X"][self.pointer:end, :max_seq_in_batch, :num_features]
        y_np = f["y"][self.pointer:end, :max_seq_in_batch]
        train_test_split_index = f["single_eval_pos"][self.pointer:end][0].item()
        
        # numpy -> pinned CPU -> VRAM (non-blocking)
        if self.device == "cuda":
            x = torch.from_numpy(x_np).pin_memory().to(self.device, non_blocking=True)
            y = torch.from_numpy(y_np).pin_memory().to(self.device, non_blocking=True)
        else:
            x = torch.from_numpy(x_np).to(self.device)
            y = torch.from_numpy(y_np).to(self.device)
        
        self.pointer += self.batch_size
        if self.pointer >= f["X"].shape[0]:
            print("Finished iteration over all stored datasets!")
            self.pointer = 0
        
        return dict(x=x, y=y, train_test_split_index=train_test_split_index)

    def __len__(self):
        return self.num_steps


class ThreadedPriorDumpDataLoader(DataLoader):
    """
    Threaded version: background thread loads data while main thread trains.
    
    Benefits:
    - True overlap: h5py disk reads happen in background thread
    - Better for slow I/O: disk reads don't block training loop
    - Bounded memory: queue size prevents unbounded growth
    """
    def __init__(self, filename, num_steps=None, batch_size=32, device=None, num_prefetch=2, max_queue_size=8):
        self.filename = filename
        self.batch_size = batch_size
        self.device = device if device else get_default_device()
        self.pointer = 0
        self.num_prefetch = num_prefetch
        self.max_queue_size = max_queue_size
        
        with h5py.File(self.filename, "r") as f:
            self.max_num_classes = f["max_num_classes"][0]
            self.total_samples = f["X"].shape[0]
        
        if num_steps is None:
            self.num_steps = self.total_samples // batch_size
        else:
            self.num_steps = num_steps
        
        print(f"ThreadedDataLoader: {self.total_samples:,} samples, {self.num_steps} steps, batch_size={batch_size}, queue_size={max_queue_size}")
        
        if self.device == "cuda":
            self.transfer_stream = torch.cuda.Stream()
    
    def __iter__(self):
        # Queue for batches (producer thread → main thread)
        batch_queue = queue.Queue(maxsize=self.max_queue_size)
        exception_queue = queue.Queue()
        stop_event = threading.Event()
        
        def producer():
            """Background thread: loads batches and puts them in queue."""
            try:
                with h5py.File(self.filename, "r") as f:
                    pointer = 0
                    for step in range(self.num_steps):
                        if stop_event.is_set():
                            break
                        
                        # Load batch (CPU work: h5py read, numpy ops)
                        batch = self._load_batch_cpu(f, pointer)
                        pointer += self.batch_size
                        if pointer >= f["X"].shape[0]:
                            pointer = 0
                        
                        # Transfer to GPU (async, on transfer stream)
                        if self.device == "cuda":
                            with torch.cuda.stream(self.transfer_stream):
                                batch = self._cpu_to_gpu(batch)
                            batch_queue.put(("batch", batch))
                        else:
                            batch = self._cpu_to_gpu(batch)
                            batch_queue.put(("batch", batch))
                    
                    batch_queue.put(("done", None))
            except Exception as e:
                exception_queue.put(e)
                stop_event.set()
        
        # Start producer thread
        producer_thread = threading.Thread(target=producer, daemon=True)
        producer_thread.start()
        
        # Main thread: consume from queue
        try:
            while True:
                msg_type, batch = batch_queue.get()
                if msg_type == "done":
                    break
                
                # Sync transfer stream before yielding
                if self.device == "cuda":
                    torch.cuda.current_stream().wait_stream(self.transfer_stream)
                
                yield batch
        finally:
            stop_event.set()
            # Check for exceptions
            try:
                exc = exception_queue.get_nowait()
                raise exc
            except queue.Empty:
                pass
    
    def _load_batch_cpu(self, f, pointer):
        """Load batch on CPU (runs in background thread)."""
        end = pointer + self.batch_size
        num_features = f["num_features"][pointer:end].max()
        num_datapoints_batch = f["num_datapoints"][pointer:end]
        max_seq_in_batch = int(num_datapoints_batch.max())
        
        x_np = f["X"][pointer:end, :max_seq_in_batch, :num_features]
        y_np = f["y"][pointer:end, :max_seq_in_batch]
        train_test_split_index = f["single_eval_pos"][pointer:end][0].item()
        
        return {
            "x_np": x_np,
            "y_np": y_np,
            "train_test_split_index": train_test_split_index
        }
    
    def _cpu_to_gpu(self, batch):
        """Transfer to GPU (can run on transfer stream)."""
        x = torch.from_numpy(batch["x_np"]).pin_memory()
        y = torch.from_numpy(batch["y_np"]).pin_memory()
        
        if self.device == "cuda":
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
        else:
            x = x.to(self.device)
            y = y.to(self.device)
        
        return dict(
            x=x,
            y=y,
            train_test_split_index=batch["train_test_split_index"]
        )
    
    def __len__(self):
        return self.num_steps


def convert_h5_to_raw(h5_filename, use_bin=False):
    """Convert HDF5 to raw files. Returns paths and shape info."""
    import os
    base = h5_filename.replace('.h5', '')
    
    if use_bin:
        x_path = f"{base}_X.bin"
        y_path = f"{base}_y.bin"
    else:
        x_path = f"{base}_X.npy"
        y_path = f"{base}_y.npy"
    meta_path = f"{base}_meta.npy"
    
    # Load metadata first to get shapes
    with h5py.File(h5_filename, "r") as f:
        shape_x = f["X"].shape
        shape_y = f["y"].shape
        meta = {
            'max_num_classes': f["max_num_classes"][0],
            'num_datapoints': f["num_datapoints"][:],
            'num_features': f["num_features"][:],
            'single_eval_pos': f["single_eval_pos"][:],
            'shape_x': shape_x,
            'shape_y': shape_y
        }
    
    if os.path.exists(x_path) and os.path.exists(y_path) and os.path.exists(meta_path):
        print(f"Raw files exist: {x_path}")
        return x_path, y_path, meta_path, meta
    
    print(f"Converting {h5_filename} to raw {'bin' if use_bin else 'npy'} files...")
    with h5py.File(h5_filename, "r") as f:
        X = f["X"][:].astype(np.float32)
        y = f["y"][:].astype(np.float32)
    
    if use_bin:
        X.tofile(x_path)
        y.tofile(y_path)
    else:
        np.save(x_path, X)
        np.save(y_path, y)
    
    np.save(meta_path, meta, allow_pickle=True)
    print(f"Saved: {x_path} ({X.nbytes / 1e9:.2f} GB)")
    return x_path, y_path, meta_path, meta


class GDSDataLoader(DataLoader):
    """
    GPU Direct Storage dataloader.
    --gds-bin: true GDS with raw .bin files (disk -> GPU, no CPU)
    --gds: uses .npy with mmap + cupy
    """
    def __init__(self, filename, num_steps=None, batch_size=32, device=None, use_bin=False):
        self.batch_size = batch_size
        self.device = device if device else get_default_device()
        self.pointer = 0
        self.use_bin = use_bin
        
        # Convert to raw format
        self.x_path, self.y_path, meta_path, meta = convert_h5_to_raw(filename, use_bin=use_bin)
        
        self.max_num_classes = meta['max_num_classes']
        self.single_eval_pos = meta['single_eval_pos']
        self.total_samples, self.max_seq, self.num_features = meta['shape_x']
        
        if num_steps is None:
            self.num_steps = self.total_samples // batch_size
        else:
            self.num_steps = num_steps
        
        # Check for kvikio
        try:
            import kvikio
            import cupy as cp
            self.has_kvikio = True
            self.kvikio = kvikio
            self.cp = cp
        except ImportError:
            self.has_kvikio = False
        
        if use_bin and self.has_kvikio:
            print(f"GDSDataLoader: TRUE GDS with .bin (disk -> GPU)")
            self.mode = "true_gds"
        elif self.has_kvikio:
            self.X_mmap = np.load(self.x_path, mmap_mode='r')
            self.y_mmap = np.load(self.y_path, mmap_mode='r')
            print(f"GDSDataLoader: mmap + cupy")
            self.mode = "mmap_cupy"
        else:
            if use_bin:
                # .bin files need to be memory-mapped with np.memmap
                self.X_mmap = np.memmap(self.x_path, dtype=np.float32, mode='r').reshape(self.total_samples, self.max_seq, self.num_features)
                self.y_mmap = np.memmap(self.y_path, dtype=np.float32, mode='r').reshape(self.total_samples, self.max_seq)
                print(f"GDSDataLoader: .bin mmap + torch (no kvikio)")
            else:
                self.X_mmap = np.load(self.x_path, mmap_mode='r')
                self.y_mmap = np.load(self.y_path, mmap_mode='r')
                print(f"GDSDataLoader: .npy mmap + torch")
            self.mode = "mmap_torch"
        
        print(f"GDSDataLoader: {self.total_samples:,} samples, {self.num_steps} steps, batch_size={batch_size}")
    
    def __iter__(self):
        if self.mode == "true_gds":
            yield from self._iter_true_gds()
        elif self.mode == "mmap_cupy":
            yield from self._iter_mmap_cupy()
        else:
            yield from self._iter_mmap_torch()
    
    def _iter_true_gds(self):
        """TRUE GDS: .bin file -> GPU via kvikio, bypassing CPU entirely"""
        kvikio = self.kvikio
        cp = self.cp
        
        bytes_per_x_sample = self.max_seq * self.num_features * 4
        bytes_per_y_sample = self.max_seq * 4
        
        x_file = kvikio.CuFile(self.x_path, "r")
        y_file = kvikio.CuFile(self.y_path, "r")
        
        try:
            for step in range(self.num_steps):
                # Allocate GPU buffers
                x_gpu = cp.empty((self.batch_size, self.max_seq, self.num_features), dtype=cp.float32)
                y_gpu = cp.empty((self.batch_size, self.max_seq), dtype=cp.float32)
                
                # Calculate file offsets
                x_offset = self.pointer * bytes_per_x_sample
                y_offset = self.pointer * bytes_per_y_sample
                
                # Direct read: disk -> GPU (no CPU involved)
                x_file.pread(x_gpu, file_offset=x_offset)
                y_file.pread(y_gpu, file_offset=y_offset)
                
                x = torch.as_tensor(x_gpu, device=self.device)
                y = torch.as_tensor(y_gpu, device=self.device)
                
                train_test_split_index = self.single_eval_pos[self.pointer]
                
                self.pointer += self.batch_size
                if self.pointer >= self.total_samples:
                    self.pointer = 0
                
                yield dict(x=x, y=y, train_test_split_index=int(train_test_split_index))
        finally:
            x_file.close()
            y_file.close()
    
    def _iter_mmap_cupy(self):
        """mmap -> cupy -> torch (still fast, but touches CPU)"""
        cp = self.cp
        
        for step in range(self.num_steps):
            end = self.pointer + self.batch_size
            
            x_gpu = cp.asarray(self.X_mmap[self.pointer:end])
            y_gpu = cp.asarray(self.y_mmap[self.pointer:end])
            
            x = torch.as_tensor(x_gpu, device=self.device)
            y = torch.as_tensor(y_gpu, device=self.device)
            
            train_test_split_index = self.single_eval_pos[self.pointer]
            
            self.pointer += self.batch_size
            if self.pointer >= self.total_samples:
                self.pointer = 0
            
            yield dict(x=x, y=y, train_test_split_index=int(train_test_split_index))
    
    def _iter_mmap_torch(self):
        """Fallback: mmap + torch"""
        for step in range(self.num_steps):
            end = self.pointer + self.batch_size
            
            x = torch.from_numpy(self.X_mmap[self.pointer:end].copy()).to(self.device)
            y = torch.from_numpy(self.y_mmap[self.pointer:end].copy()).to(self.device)
            train_test_split_index = self.single_eval_pos[self.pointer]
            
            self.pointer += self.batch_size
            if self.pointer >= self.total_samples:
                self.pointer = 0
            
            yield dict(x=x, y=y, train_test_split_index=int(train_test_split_index))
    
    def __len__(self):
        return self.num_steps


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", action="store_true", help="Profile with PyTorch profiler (ops-level)")
    parser.add_argument("--profile-model", action="store_true", help="Detailed model profiling (layer-by-layer)")
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile")
    parser.add_argument("--batch-size", type=int, default=64, help="Larger batch = better GPU util")
    parser.add_argument("--steps", type=int, default=100, help="Training steps for timing")
    parser.add_argument("--prefetch", type=int, default=2, help="Batches to prefetch in VRAM")
    parser.add_argument("--data", type=str, default="30k_5000x5_2.h5", help="HDF5 data file")
    parser.add_argument("--full", action="store_true", help="Train on full dataset (ignore --steps)")
    parser.add_argument("--gds", action="store_true", help="Use GDS with .npy (mmap + cupy)")
    parser.add_argument("--gds-bin", action="store_true", help="TRUE GDS with .bin (disk -> GPU, no CPU)")
    parser.add_argument("--flash", action="store_true", help="Use Flash Attention model (O(n) memory)")
    parser.add_argument("--checkpoint", action="store_true", help="Enable gradient checkpointing (saves memory)")
    parser.add_argument("--threaded", action="store_true", help="Use background thread for data loading (better I/O overlap)")
    parser.add_argument("--queue-size", type=int, default=8, help="Max queue size for threaded loader")
    parser.add_argument("--num-kv-heads", type=int, default=None, help="Number of K,V heads for GQA (default: same as num_heads)")
    parser.add_argument("--save-peak-mem-factor", type=int, default=None, help="Chunk batch dimension by this factor to reduce peak memory (inference only)")
    args = parser.parse_args()

    device = get_default_device()
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"TF32 enabled: {torch.backends.cuda.matmul.allow_tf32}")
    print(f"torch.compile: {not args.no_compile}")
    print(f"Batch size: {args.batch_size}")
    print(f"VRAM prefetch: {args.prefetch} batches")
    print(f"Data file: {args.data}")
    print(f"GDS mode: {'bin (true GDS)' if args.gds_bin else 'npy' if args.gds else 'off'}")
    print(f"Threaded loading: {args.threaded}")
    if args.threaded:
        print(f"Queue size: {args.queue_size}")
    print(f"Flash Attention: {args.flash}")
    if args.flash:
        if args.num_kv_heads:
            print(f"GQA: {args.num_attention_heads} Q heads, {args.num_kv_heads} K,V heads")
        if args.save_peak_mem_factor:
            print(f"Batch chunking: factor={args.save_peak_mem_factor}")
    print(f"Gradient checkpointing: {args.checkpoint}")
    
    # Determine GDS mode
    use_gds = args.gds or args.gds_bin
    use_bin = args.gds_bin
    
    # Determine num_steps
    num_steps = None if args.full else args.steps
    
    # Helper function to create dataloader
    def create_dataloader(data_file, num_steps, batch_size, device):
        """Create appropriate dataloader based on flags."""
        if use_gds:
            return GDSDataLoader(data_file, num_steps=num_steps, batch_size=batch_size, device=device, use_bin=use_bin)
        elif args.threaded:
            return ThreadedPriorDumpDataLoader(
                data_file, 
                num_steps=num_steps, 
                batch_size=batch_size, 
                device=device, 
                num_prefetch=args.prefetch,
                max_queue_size=args.queue_size
            )
        else:
            return PriorDumpDataLoader(
                data_file, 
                num_steps=num_steps, 
                batch_size=batch_size, 
                device=device, 
                num_prefetch=args.prefetch
            )
    
    # Select model
    if args.flash:
        model = NanoTabPFNModelOptimized(
            embedding_size=96,
            num_attention_heads=4,
            mlp_hidden_size=192,
            num_layers=3,
            num_outputs=2,
            use_checkpointing=args.checkpoint,
            num_kv_heads=args.num_kv_heads
        )
    else:
        model = NanoTabPFNModel(
            embedding_size=96,
            num_attention_heads=4,
            mlp_hidden_size=192,
            num_layers=3,
            num_outputs=2
        )
    
    # Compile once here, not inside train()
    if not args.no_compile and device == "cuda":
        print("Compiling model...")
        model = torch.compile(model)

    if args.profile_model:
        # =====================================================================
        # DETAILED MODEL PROFILING MODE
        # =====================================================================
        print("\n" + "="*70)
        print(" DETAILED MODEL PROFILING MODE ".center(70, "="))
        print("="*70)
        
        # Warmup
        print("\nWarming up...")
        warmup_prior = create_dataloader(args.data, num_steps=3, batch_size=args.batch_size, device=device)
        train(model, warmup_prior, lr=4e-3, steps_per_eval=100)
        if device == "cuda":
            torch.cuda.synchronize()
        
        # Setup profilers
        layer_profiler = LayerProfiler()
        # Only attach to non-compiled model (compiled model's hooks won't work well)
        if args.no_compile:
            layer_profiler.attach(model)
        
        model_profiler = ModelProfiler(device=device)
        
        # Load data
        prior = create_dataloader(args.data, num_steps=args.steps, batch_size=args.batch_size, device=device)
        
        print(f"\nProfiling {args.steps} training steps...")
        print(f"Input shape: (batch={args.batch_size}, rows=variable, cols=variable, embed=96)")
        
        model.to(device)
        optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=4e-3, weight_decay=0.0)
        criterion = nn.CrossEntropyLoss()
        model.train()
        optimizer.train()
        
        # Enable layer profiling
        if args.no_compile:
            layer_profiler.enable()
        
        # Profile each step
        for step, full_data in enumerate(prior):
            train_test_split_index = full_data["train_test_split_index"]
            x = full_data["x"]
            y_full = full_data["y"]
            
            data = (x, y_full[:, :train_test_split_index])
            targets = y_full[:, train_test_split_index:].reshape(-1).long()
            
            # First step: also profile attention ops
            if step == 0:
                print(f"\nStep 0 shapes: x={tuple(x.shape)}, y={tuple(y_full.shape)}, split={train_test_split_index}")
                profile_attention_ops(model, x, y_full, train_test_split_index, device)
            
            loss, times, mems = model_profiler.profile_step(
                model, data, targets, train_test_split_index, criterion, optimizer
            )
            
            if step % 10 == 0:
                mem_alloc = mems['after_backward'].get('allocated_mb', 0) if mems['after_backward'] else 0
                print(f"  Step {step:3d}: fwd={times['forward']:6.1f}ms  bwd={times['backward']:6.1f}ms  mem={mem_alloc:6.0f}MB  loss={loss.item():.4f}")
        
        # Print summaries
        model_profiler.summary()
        
        if args.no_compile:
            layer_profiler.disable()
            layer_profiler.summary(top_k=25)
            layer_profiler.remove_hooks()
        else:
            print("\n[Note: Layer-level profiling disabled with torch.compile. Use --no-compile for layer breakdown.]")
        
        # Memory analysis
        if device == "cuda":
            print(f"\n{'='*70}")
            print(f"{'CUDA MEMORY ANALYSIS':^70}")
            print(f"{'='*70}")
            print(torch.cuda.memory_summary(abbreviated=True))
    
    elif args.profile:
        from torch.profiler import profile, ProfilerActivity
        
        # Warmup for torch.compile (captures compilation time separately)
        print("Warming up (includes torch.compile time if enabled)...")
        warmup_start = time.time()
        warmup_prior = create_dataloader(args.data, num_steps=5, batch_size=args.batch_size, device=device)
        train(model, warmup_prior, lr=4e-3, steps_per_eval=100)
        if device == "cuda":
            torch.cuda.synchronize()
        warmup_time = time.time() - warmup_start
        print(f"Warmup time: {warmup_time:.2f}s")
        
        print(f"\nProfiling {args.steps} steps...")
        prior = create_dataloader(args.data, num_steps=num_steps, batch_size=args.batch_size, device=device)
        profile_start = time.time()
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
        ) as prof:
            train(model, prior, lr=4e-3, steps_per_eval=args.steps + 1)
        if device == "cuda":
            torch.cuda.synchronize()
        profile_time = time.time() - profile_start
        
        print(f"\n=== TIMING ===")
        print(f"Warmup (compile): {warmup_time:.2f}s")
        print(f"Training {args.steps} steps: {profile_time:.2f}s")
        print(f"Steps/sec: {args.steps / profile_time:.1f}")
        print(f"ms/step: {1000 * profile_time / args.steps:.2f}")
        
        print("\n=== TOP CUDA OPERATIONS ===")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
        print("\n=== TOP CPU OPERATIONS ===")
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=15))
        print("\n=== MEMORY TRANSFERS ===")
        for evt in prof.key_averages():
            if any(x in evt.key for x in ["aten::to", "aten::copy_", "cudaMemcpy"]):
                cuda_t = getattr(evt, 'self_cuda_time_total', 0) or 0
                print(f"{evt.key:50} CPU:{evt.cpu_time_total/1e3:8.2f}ms  CUDA:{cuda_t/1e3:8.2f}ms  Calls:{evt.count}")
        if device == "cuda":
            print("\n=== CUDA MEMORY SUMMARY ===")
            print(torch.cuda.memory_summary())
    else:
        # Timed training run
        print(f"\nTraining {args.steps} steps...")
        prior = create_dataloader(args.data, num_steps=num_steps, batch_size=args.batch_size, device=device)
        
        start = time.time()
        model, history = train(model, prior, lr=4e-3, steps_per_eval=25)
        if device == "cuda":
            torch.cuda.synchronize()
        total_time = time.time() - start
        
        print(f"\n=== TIMING ===")
        print(f"Total time: {total_time:.2f}s")
        print(f"Steps/sec: {args.steps / total_time:.1f}")
        print(f"ms/step: {1000 * total_time / args.steps:.2f}")
        
        print("\nFinal evaluation:")
        print(eval(NanoTabPFNClassifier(model, device)))

