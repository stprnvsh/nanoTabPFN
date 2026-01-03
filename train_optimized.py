"""
Optimized training with GPU utilization improvements:
1. torch.compile() - fuses kernels, reduces launch overhead
2. TF32 - faster matmuls on Ampere+
3. Larger batch size - better GPU occupancy
4. Non-blocking transfers - overlap H2D with compute
5. Pinned memory - faster CPU->GPU transfers
6. CUDA streams - parallel data transfer and compute
"""
import random
import time
import h5py
import numpy as np
import schedulefree
import torch
from model import NanoTabPFNClassifier, NanoTabPFNModel
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader


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
          lr: float = 1e-4, device: torch.device = None, steps_per_eval=10, eval_func=None,
          use_compile=True):
    if not device:
        device = get_default_device()
    model.to(device)
    
    # Compile model to fuse kernels (reduces 18k launches to ~hundreds)
    if use_compile and device == "cuda":
        model = torch.compile(model, mode="reduce-overhead")
    
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


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile")
    parser.add_argument("--batch-size", type=int, default=64, help="Larger batch = better GPU util")
    parser.add_argument("--steps", type=int, default=100, help="Training steps for timing")
    parser.add_argument("--prefetch", type=int, default=2, help="Batches to prefetch in VRAM")
    parser.add_argument("--data", type=str, default="300k_150x5_2.h5", help="HDF5 data file")
    parser.add_argument("--full", action="store_true", help="Train on full dataset (ignore --steps)")
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
    
    # Determine num_steps
    num_steps = None if args.full else args.steps
    
    model = NanoTabPFNModel(
        embedding_size=96,
        num_attention_heads=4,
        mlp_hidden_size=192,
        num_layers=3,
        num_outputs=2
    )

    if args.profile:
        from torch.profiler import profile, ProfilerActivity
        
        # Warmup for torch.compile (captures compilation time separately)
        print("Warming up (includes torch.compile time if enabled)...")
        warmup_start = time.time()
        warmup_prior = PriorDumpDataLoader(args.data, num_steps=5, batch_size=args.batch_size, device=device, num_prefetch=args.prefetch)
        train(model, warmup_prior, lr=4e-3, steps_per_eval=100, use_compile=not args.no_compile)
        if device == "cuda":
            torch.cuda.synchronize()
        warmup_time = time.time() - warmup_start
        print(f"Warmup time: {warmup_time:.2f}s")
        
        print(f"\nProfiling {args.steps} steps...")
        prior = PriorDumpDataLoader(args.data, num_steps=num_steps, batch_size=args.batch_size, device=device, num_prefetch=args.prefetch)
        profile_start = time.time()
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
        ) as prof:
            train(model, prior, lr=4e-3, steps_per_eval=args.steps + 1, use_compile=False)  # Already compiled
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
        prior = PriorDumpDataLoader(args.data, num_steps=num_steps, batch_size=args.batch_size, device=device, num_prefetch=args.prefetch)
        
        start = time.time()
        model, history = train(model, prior, lr=4e-3, steps_per_eval=25, use_compile=not args.no_compile)
        if device == "cuda":
            torch.cuda.synchronize()
        total_time = time.time() - start
        
        print(f"\n=== TIMING ===")
        print(f"Total time: {total_time:.2f}s")
        print(f"Steps/sec: {args.steps / total_time:.1f}")
        print(f"ms/step: {1000 * total_time / args.steps:.2f}")
        
        print("\nFinal evaluation:")
        print(eval(NanoTabPFNClassifier(model, device)))

