"""
Training with Flash Attention for O(n) memory instead of O(n²).
Uses model_optimized.py with scaled_dot_product_attention.

Key changes from train_optimized.py:
- Uses NanoTabPFNModelOptimized with Flash Attention
- Can handle longer sequences without OOM
"""
import random
import time
import h5py
import numpy as np
import schedulefree
import torch
from model_optimized import NanoTabPFNModelOptimized, NanoTabPFNClassifier
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

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

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


def train(model, prior: DataLoader, lr: float = 1e-4, device: torch.device = None, steps_per_eval=10, eval_func=None):
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
            optimizer.zero_grad(set_to_none=True)
            
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


class GDSDataLoader(DataLoader):
    """GPU Direct Storage dataloader using kvikio."""
    def __init__(self, filename, num_steps=None, batch_size=32, device=None):
        self.batch_size = batch_size
        self.device = device if device else get_default_device()
        self.pointer = 0
        
        with h5py.File(filename, "r") as f:
            self.max_num_classes = f["max_num_classes"][0]
            self.total_samples = f["X"].shape[0]
            self.max_seq = f["X"].shape[1]
            self.num_features = f["X"].shape[2]
            self.num_datapoints = f["num_datapoints"][:]
            self.num_features_arr = f["num_features"][:]
            self.single_eval_pos = f["single_eval_pos"][:]
        
        if num_steps is None:
            self.num_steps = self.total_samples // batch_size
        else:
            self.num_steps = num_steps
        
        try:
            import cupy as cp
            self.use_gds = True
            self.cp = cp
            print(f"GDSDataLoader: Using cupy")
        except ImportError:
            self.use_gds = False
            print(f"GDSDataLoader: cupy not available, using h5py")
        
        self.filename = filename
        print(f"GDSDataLoader: {self.total_samples:,} samples, {self.num_steps} steps, batch_size={batch_size}")
    
    def __iter__(self):
        if self.use_gds:
            yield from self._iter_gds()
        else:
            yield from self._iter_h5py()
    
    def _iter_gds(self):
        cp = self.cp
        with h5py.File(self.filename, "r") as f:
            X_data = f["X"]
            y_data = f["y"]
            
            for step in range(self.num_steps):
                end = self.pointer + self.batch_size
                
                x_np = X_data[self.pointer:end]
                y_np = y_data[self.pointer:end]
                
                x_gpu = cp.asarray(x_np)
                y_gpu = cp.asarray(y_np)
                
                x = torch.as_tensor(x_gpu, device=self.device)
                y = torch.as_tensor(y_gpu, device=self.device)
                
                train_test_split_index = self.single_eval_pos[self.pointer]
                
                self.pointer += self.batch_size
                if self.pointer >= self.total_samples:
                    self.pointer = 0
                
                yield dict(x=x, y=y, train_test_split_index=int(train_test_split_index))
    
    def _iter_h5py(self):
        with h5py.File(self.filename, "r") as f:
            for step in range(self.num_steps):
                end = self.pointer + self.batch_size
                
                x = torch.from_numpy(f["X"][self.pointer:end]).to(self.device)
                y = torch.from_numpy(f["y"][self.pointer:end]).to(self.device)
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
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (can be larger with Flash Attention)")
    parser.add_argument("--steps", type=int, default=100, help="Training steps")
    parser.add_argument("--data", type=str, default="30k_5000x5_2.h5", help="HDF5 data file")
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile")
    args = parser.parse_args()

    device = get_default_device()
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        # Check Flash Attention availability
        print(f"Flash Attention available: {torch.backends.cuda.flash_sdp_enabled()}")
        print(f"Memory-efficient attention available: {torch.backends.cuda.mem_efficient_sdp_enabled()}")
    print(f"Batch size: {args.batch_size}")
    print(f"Data file: {args.data}")
    
    model = NanoTabPFNModelOptimized(
        embedding_size=96,
        num_attention_heads=4,
        mlp_hidden_size=192,
        num_layers=3,
        num_outputs=2
    )
    
    if not args.no_compile and device == "cuda":
        print("Compiling model...")
        model = torch.compile(model)
    
    prior = GDSDataLoader(args.data, num_steps=args.steps, batch_size=args.batch_size, device=device)
    
    start = time.time()
    model, history = train(model, prior, lr=4e-3, steps_per_eval=25)
    if device == "cuda":
        torch.cuda.synchronize()
    total_time = time.time() - start
    
    print(f"\n=== TIMING ===")
    print(f"Total time: {total_time:.2f}s")
    print(f"Steps/sec: {args.steps / total_time:.1f}")
    print(f"ms/step: {1000 * total_time / args.steps:.2f}")
    
    if device == "cuda":
        print(f"\n=== MEMORY ===")
        print(f"Peak memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
        print(f"Reserved memory: {torch.cuda.max_memory_reserved() / 1e9:.2f} GB")
    
    print("\nFinal evaluation:")
    print(eval(NanoTabPFNClassifier(model, device)))

