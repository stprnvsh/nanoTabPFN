"""
Compare NanoTabPFN (Transformer) vs NanoTabPFN-GIN on tabular classification.
Uses the same training infrastructure and NanoTabPFNClassifier interface.
"""
import random
import time
import h5py
import numpy as np
import schedulefree
import torch
from torch import nn
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from model import NanoTabPFNModel, NanoTabPFNGIN, NanoTabPFNClassifier


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# Evaluation datasets
def get_datasets():
    datasets = []
    datasets.append(("breast_cancer", *train_test_split(*load_breast_cancer(return_X_y=True), test_size=0.5, random_state=0)))
    datasets.append(("iris", *train_test_split(*load_iris(return_X_y=True), test_size=0.5, random_state=0)))
    datasets.append(("wine", *train_test_split(*load_wine(return_X_y=True), test_size=0.5, random_state=0)))
    return datasets


def evaluate(classifier, datasets):
    """Evaluate classifier on multiple datasets."""
    scores = {"roc_auc": 0, "acc": 0, "balanced_acc": 0}
    
    for name, X_train, X_test, y_train, y_test in datasets:
        classifier.fit(X_train, y_train)
        prob = classifier.predict_proba(X_test)
        pred = prob.argmax(axis=1)
        
        if prob.shape[1] == 2:
            prob = prob[:, 1]
        
        try:
            scores["roc_auc"] += float(roc_auc_score(y_test, prob, multi_class="ovr"))
        except:
            scores["roc_auc"] += 0.5
        scores["acc"] += float(accuracy_score(y_test, pred))
        scores["balanced_acc"] += float(balanced_accuracy_score(y_test, pred))
    
    return {k: v / len(datasets) for k, v in scores.items()}


class PriorDumpDataLoader:
    """Simple dataloader for HDF5 prior data."""
    def __init__(self, filename, num_steps, batch_size, device):
        self.filename = filename
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.device = device
        self.pointer = 0
        
        with h5py.File(self.filename, "r") as f:
            self.max_num_classes = f["max_num_classes"][0]

    def __iter__(self):
        with h5py.File(self.filename, "r") as f:
            for _ in range(self.num_steps):
                end = self.pointer + self.batch_size
                num_features = f["num_features"][self.pointer:end].max()
                num_datapoints = f["num_datapoints"][self.pointer:end]
                max_seq = int(num_datapoints.max())
                
                x = torch.from_numpy(f["X"][self.pointer:end, :max_seq, :num_features]).to(self.device)
                y = torch.from_numpy(f["y"][self.pointer:end, :max_seq]).to(self.device)
                split_idx = f["single_eval_pos"][self.pointer:end][0].item()
                
                self.pointer += self.batch_size
                if self.pointer >= f["X"].shape[0]:
                    self.pointer = 0
                
                yield dict(x=x, y=y, train_test_split_index=split_idx)

    def __len__(self):
        return self.num_steps


def train_model(model, prior, lr=4e-3, device="cuda"):
    """Train a model and return training history."""
    model.to(device)
    optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=lr, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    optimizer.train()
    
    losses = []
    start_time = time.time()
    
    for step, batch in enumerate(prior):
        split_idx = batch["train_test_split_index"]
        x = batch["x"]
        y_full = batch["y"]
        
        data = (x, y_full[:, :split_idx])
        targets = y_full[:, split_idx:].reshape(-1).long()
        
        output = model(data, train_test_split_index=split_idx)
        output = output.view(-1, output.shape[-1])
        
        loss = criterion(output, targets)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        
        losses.append(loss.item())
        
        if (step + 1) % 25 == 0:
            print(f"  Step {step+1}: loss={np.mean(losses[-25:]):.4f}")
    
    train_time = time.time() - start_time
    return losses, train_time


def main():
    set_seed(42)
    device = get_device()
    print(f"Device: {device}")
    
    # Config
    embedding_size = 96
    mlp_hidden_size = 192
    num_layers = 3
    num_outputs = 2
    num_manifold_points = 16  # For GIN
    
    batch_size = 64
    num_steps = 100
    data_file = "300k_150x5_2.h5"  # Adjust path if needed
    
    print(f"\n{'='*60}")
    print(f"{'NANOTABPFN vs GIN COMPARISON':^60}")
    print(f"{'='*60}")
    
    # Create models
    models = {
        "Transformer": NanoTabPFNModel(
            embedding_size=embedding_size,
            num_attention_heads=4,
            mlp_hidden_size=mlp_hidden_size,
            num_layers=num_layers,
            num_outputs=num_outputs
        ),
        "GIN": NanoTabPFNGIN(
            embedding_size=embedding_size,
            num_manifold_points=num_manifold_points,
            mlp_hidden_size=mlp_hidden_size,
            num_layers=num_layers,
            num_outputs=num_outputs
        ),
    }
    
    # Print model info
    print(f"\nModel comparison:")
    for name, model in models.items():
        print(f"  {name}: {count_params(model):,} parameters")
    
    # Get evaluation datasets
    datasets = get_datasets()
    print(f"\nEvaluation datasets: {[d[0] for d in datasets]}")
    
    # Train and evaluate each model
    results = {}
    
    for name, model in models.items():
        print(f"\n{'='*60}")
        print(f"Training {name}")
        print(f"{'='*60}")
        
        # Reset model
        set_seed(42)
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
        # Train
        prior = PriorDumpDataLoader(data_file, num_steps=num_steps, batch_size=batch_size, device=device)
        losses, train_time = train_model(model, prior, device=device)
        
        # Evaluate
        model.eval()
        classifier = NanoTabPFNClassifier(model, device)
        scores = evaluate(classifier, datasets)
        
        results[name] = {
            "params": count_params(model),
            "final_loss": np.mean(losses[-10:]),
            "train_time": train_time,
            **scores
        }
        
        print(f"\n{name} Results:")
        print(f"  Final loss: {results[name]['final_loss']:.4f}")
        print(f"  Accuracy: {scores['acc']:.4f}")
        print(f"  Balanced Acc: {scores['balanced_acc']:.4f}")
        print(f"  ROC AUC: {scores['roc_auc']:.4f}")
        print(f"  Train time: {train_time:.1f}s")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"{'SUMMARY':^60}")
    print(f"{'='*60}")
    print(f"{'Model':<15} {'Params':>10} {'Loss':>10} {'Acc':>10} {'ROC AUC':>10} {'Time':>10}")
    print(f"{'-'*65}")
    for name, r in results.items():
        print(f"{name:<15} {r['params']:>10,} {r['final_loss']:>10.4f} {r['acc']:>10.4f} {r['roc_auc']:>10.4f} {r['train_time']:>9.1f}s")
    
    # Winner
    best = max(results.items(), key=lambda x: x[1]['acc'])
    print(f"\nBest model by accuracy: {best[0]} ({best[1]['acc']:.4f})")
    
    return results


if __name__ == "__main__":
    results = main()

