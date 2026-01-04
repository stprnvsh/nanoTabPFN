"""Train model and visualize what attention learns."""
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from pde_data import generate_heat_equation_1d, generate_wave_equation_1d, generate_ood_heat_data
from model import PDETransformer

def train(pde_type='heat', n_epochs=100, lr=1e-3, n_layers=1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Generate data
    print(f"Generating {pde_type} equation data...")
    if pde_type == 'heat':
        _, solutions = generate_heat_equation_1d(n_samples=1000, n_points=64, n_timesteps=32)
    else:
        _, solutions = generate_wave_equation_1d(n_samples=1000, n_points=64, n_timesteps=32)
    
    # Train/test split
    train_sol = solutions[:800].to(device)
    test_sol = solutions[800:].to(device)
    
    # Model
    print(f"Model: {n_layers} layer(s), 4 heads, 64 dim")
    model = PDETransformer(d_model=64, n_heads=4, n_layers=n_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # Training loop
    losses = []
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        
        # Train on consecutive time steps
        for t in range(train_sol.shape[1] - 1):
            x = train_sol[:, t, :]  # (B, n_points)
            y = train_sol[:, t+1, :]
            
            pred = model(x)
            loss = criterion(pred, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / (train_sol.shape[1] - 1)
        losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.6f}")
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        test_loss = 0
        for t in range(test_sol.shape[1] - 1):
            x = test_sol[:, t, :]
            y = test_sol[:, t+1, :]
            pred = model(x)
            test_loss += criterion(pred, y).item()
        print(f"Test Loss: {test_loss / (test_sol.shape[1] - 1):.6f}")
    
    return model, losses, test_sol

def visualize_attention(model, test_data, pde_type='heat'):
    """Visualize what the attention has learned."""
    model.eval()
    device = next(model.parameters()).device
    n_layers = model.n_layers
    
    # Pick a sample
    sample = test_data[0]
    x = sample[0:1, :].to(device)
    
    with torch.no_grad():
        pred, attn = model(x, return_attention=True)
    
    # attn shape: (n_layers, B, n_heads, S, S)
    attn = attn[:, 0].cpu().numpy()  # (n_layers, n_heads, S, S)
    n_heads = attn.shape[1]
    
    # Create figure based on number of layers
    fig, axes = plt.subplots(n_layers, 4, figsize=(16, 4*n_layers))
    if n_layers == 1:
        axes = axes.reshape(1, -1)
    
    for layer in range(n_layers):
        layer_attn = attn[layer]  # (n_heads, S, S)
        
        # Plot each head
        for h in range(min(n_heads, 3)):
            ax = axes[layer, h]
            im = ax.imshow(layer_attn[h], cmap='viridis', aspect='auto')
            ax.set_title(f'L{layer+1} Head {h+1}')
            ax.set_xlabel('Key')
            ax.set_ylabel('Query')
            plt.colorbar(im, ax=ax)
        
        # Average attention for this layer
        ax = axes[layer, 3]
        center = x.shape[1] // 2
        for h in range(n_heads):
            ax.plot(layer_attn[h, center, :], label=f'H{h+1}', alpha=0.7)
        ax.axvline(x=center, color='red', linestyle='--')
        ax.set_title(f'L{layer+1} from center')
        ax.legend(fontsize=8)
    
    plt.suptitle(f'{pde_type.capitalize()} Equation - {n_layers} Layer(s)')
    plt.tight_layout()
    plt.savefig(f'attention_patterns_{pde_type}.png', dpi=150)
    plt.show()
    
    # Return last layer attention for compatibility
    return attn[-1]

def analyze_attention_locality(attn, pde_type):
    """Analyze if attention learns local (finite difference-like) patterns."""
    n_heads, seq_len, _ = attn.shape
    
    print(f"\n=== Attention Analysis for {pde_type} equation ===")
    
    for h in range(n_heads):
        head_attn = attn[h]
        
        # Measure locality: how much weight is on neighbors
        local_weight = 0
        for i in range(1, seq_len - 1):
            neighbors = [i-1, i, i+1]
            local_weight += sum(head_attn[i, j] for j in neighbors)
        local_weight /= (seq_len - 2)
        
        # Find dominant pattern
        avg_row = head_attn[seq_len//4:3*seq_len//4, :].mean(axis=0)
        peak_offset = np.argmax(avg_row) - seq_len // 2
        
        print(f"Head {h+1}:")
        print(f"  Local weight (3 neighbors): {local_weight:.3f}")
        print(f"  Peak attention offset: {peak_offset}")
        
        # Check if it resembles finite difference stencil
        center = seq_len // 2
        stencil = head_attn[center, center-2:center+3]
        print(f"  Stencil pattern at center: {stencil}")

def rollout_prediction(model, initial, n_steps, device):
    """Rollout model predictions autoregressively."""
    model.eval()
    predictions = [initial.cpu().numpy()]
    x = initial.unsqueeze(0).to(device)
    
    with torch.no_grad():
        for _ in range(n_steps):
            x = model(x)
            predictions.append(x[0].cpu().numpy())
            x = x.unsqueeze(0) if x.dim() == 1 else x
    
    return np.array(predictions)

def visualize_rollout(model, test_data, pde_type):
    """Visualize ground truth vs predicted rollout."""
    device = next(model.parameters()).device
    sample = test_data[5]  # (n_timesteps, n_points)
    
    n_steps = sample.shape[0] - 1
    pred = rollout_prediction(model, sample[0], n_steps, device)
    gt = sample.cpu().numpy()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Ground truth
    axes[0].imshow(gt, aspect='auto', cmap='RdBu_r')
    axes[0].set_title('Ground Truth')
    axes[0].set_xlabel('Spatial position')
    axes[0].set_ylabel('Time step')
    
    # Prediction
    axes[1].imshow(pred, aspect='auto', cmap='RdBu_r')
    axes[1].set_title('Model Prediction (Rollout)')
    axes[1].set_xlabel('Spatial position')
    axes[1].set_ylabel('Time step')
    
    # Error
    error = np.abs(gt - pred)
    axes[2].imshow(error, aspect='auto', cmap='Reds')
    axes[2].set_title('Absolute Error')
    axes[2].set_xlabel('Spatial position')
    axes[2].set_ylabel('Time step')
    
    plt.suptitle(f'{pde_type.capitalize()} Equation - Rollout Comparison')
    plt.tight_layout()
    plt.savefig(f'rollout_{pde_type}.png', dpi=150)
    plt.show()


def visualize_attention_vs_distance(model, test_data):
    """Plot attention weight as function of spatial distance."""
    model.eval()
    device = next(model.parameters()).device
    
    # Collect attention from multiple samples (use last layer)
    all_attns = []
    for i in range(min(50, len(test_data))):
        x = test_data[i, 0:1, :].to(device)
        with torch.no_grad():
            _, attn = model(x, return_attention=True)
        # attn: (n_layers, B, n_heads, S, S) -> take last layer, first batch
        all_attns.append(attn[-1, 0].cpu().numpy())
    
    all_attns = np.stack(all_attns)  # (n_samples, n_heads, seq, seq)
    n_heads = all_attns.shape[1]
    seq_len = all_attns.shape[2]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Attention vs distance for each head
    ax = axes[0]
    for h in range(n_heads):
        distances = []
        weights = []
        for d in range(-seq_len//2, seq_len//2):
            w = []
            for i in range(seq_len):
                j = i + d
                if 0 <= j < seq_len:
                    w.append(all_attns[:, h, i, j].mean())
            if w:
                distances.append(d)
                weights.append(np.mean(w))
        ax.plot(distances, weights, label=f'Head {h+1}', linewidth=2)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Relative Position (Key - Query)')
    ax.set_ylabel('Average Attention Weight')
    ax.set_title('Attention vs Spatial Distance')
    ax.legend()
    
    # Combined view - heat map of distance-based attention
    ax = axes[1]
    avg_attn = all_attns.mean(axis=(0, 1))  # (seq, seq)
    dist_attn = np.zeros((seq_len, seq_len))
    for i in range(seq_len):
        for j in range(seq_len):
            dist_attn[i, j] = abs(i - j)
    ax.scatter(dist_attn.flatten(), avg_attn.flatten(), alpha=0.1, s=5)
    ax.set_xlabel('Spatial Distance |i - j|')
    ax.set_ylabel('Attention Weight')
    ax.set_title('Distance vs Attention (all pairs)')
    
    plt.tight_layout()
    plt.savefig('attention_vs_distance.png', dpi=150)
    plt.show()

def compare_to_finite_difference(model, n_points=64):
    """Compare learned operator to finite difference stencil."""
    device = next(model.parameters()).device
    model.eval()
    
    # Create delta function inputs at different positions
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    
    positions = [n_points//4, n_points//2, 3*n_points//4]
    
    for idx, pos in enumerate(positions):
        # Delta function at position
        delta = torch.zeros(1, n_points)
        delta[0, pos] = 1.0
        delta = delta.to(device)
        
        with torch.no_grad():
            response = model(delta)[0].cpu().numpy()
        
        ax = axes[0, idx]
        ax.plot(response, 'b-', linewidth=2, label='Model response')
        ax.axvline(x=pos, color='r', linestyle='--', alpha=0.5, label='Input delta')
        ax.set_title(f'Response to delta at pos {pos}')
        ax.set_xlabel('Spatial position')
        ax.legend()
        
        # Zoom to local response
        ax = axes[1, idx]
        left = max(0, pos - 10)
        right = min(n_points, pos + 10)
        ax.bar(range(left, right), response[left:right], alpha=0.7)
        ax.axvline(x=pos, color='r', linestyle='--')
        ax.set_title(f'Local stencil (zoom)')
        ax.set_xlabel('Position')
    
    plt.suptitle('Learned Operator: Response to Delta Functions\n(Approximates Finite Difference Stencil)')
    plt.tight_layout()
    plt.savefig('learned_stencil.png', dpi=150)
    plt.show()

def test_ood_generalization(model, pde_type='heat'):
    """Test on out-of-distribution initial conditions."""
    if pde_type != 'heat':
        print("OOD test only implemented for heat equation")
        return
    
    device = next(model.parameters()).device
    model.eval()
    
    ic_types = ['step', 'gaussian', 'triangle', 'multi_bump']
    
    fig, axes = plt.subplots(len(ic_types), 4, figsize=(16, 12))
    
    criterion = nn.MSELoss()
    
    for row, ic_type in enumerate(ic_types):
        _, ood_data = generate_ood_heat_data(ic_type=ic_type, n_samples=20)
        ood_data = ood_data.to(device)
        
        # Test loss
        with torch.no_grad():
            test_loss = 0
            for t in range(ood_data.shape[1] - 1):
                pred = model(ood_data[:, t, :])
                test_loss += criterion(pred, ood_data[:, t+1, :]).item()
            avg_loss = test_loss / (ood_data.shape[1] - 1)
        
        # Visualize one sample
        sample = ood_data[0].cpu().numpy()
        
        # Initial condition
        axes[row, 0].plot(sample[0], 'b-', linewidth=2)
        axes[row, 0].set_title(f'{ic_type}: Initial')
        axes[row, 0].set_ylabel(ic_type.upper())
        
        # Ground truth evolution
        axes[row, 1].imshow(sample, aspect='auto', cmap='RdBu_r')
        axes[row, 1].set_title('Ground Truth')
        
        # Prediction rollout
        pred = rollout_prediction(model, ood_data[0, 0], ood_data.shape[1]-1, device)
        axes[row, 2].imshow(pred, aspect='auto', cmap='RdBu_r')
        axes[row, 2].set_title('Model Prediction')
        
        # Error
        error = np.abs(sample - pred)
        axes[row, 3].imshow(error, aspect='auto', cmap='Reds')
        axes[row, 3].set_title(f'Error (MSE: {avg_loss:.4f})')
    
    plt.suptitle('Out-of-Distribution Generalization Test\n(Trained on sine waves, tested on different ICs)', fontsize=14)
    plt.tight_layout()
    plt.savefig('ood_generalization.png', dpi=150)
    plt.show()

def visualize_what_attention_learns_summary(model, test_data, pde_type):
    """Summary visualization of attention learning."""
    model.eval()
    device = next(model.parameters()).device
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Get attention from center of domain (use last layer)
    x = test_data[0, 0:1, :].to(device)
    with torch.no_grad():
        _, attn = model(x, return_attention=True)
    # attn: (n_layers, B, n_heads, S, S) -> last layer, first batch
    attn = attn[-1, 0].cpu().numpy()
    n_heads = attn.shape[0]
    seq_len = attn.shape[1]
    
    # 1. Average attention pattern (should show locality)
    ax = axes[0, 0]
    avg_attn = attn.mean(axis=0)
    im = ax.imshow(avg_attn, cmap='viridis', aspect='auto')
    ax.set_title('Avg Attention Pattern')
    ax.set_xlabel('Key (spatial)')
    ax.set_ylabel('Query (spatial)')
    plt.colorbar(im, ax=ax)
    
    # 2. Diagonal dominance - attention concentrates near diagonal for local ops
    ax = axes[0, 1]
    for h in range(n_heads):
        diag_vals = np.diag(attn[h])
        ax.plot(diag_vals, label=f'Head {h+1}', alpha=0.7)
    ax.set_title('Diagonal Attention (self-attention strength)')
    ax.set_xlabel('Position')
    ax.set_ylabel('Self-attention weight')
    ax.legend()
    
    # 3. Attention entropy - low entropy = focused, high = diffuse
    ax = axes[1, 0]
    for h in range(n_heads):
        entropy = -np.sum(attn[h] * np.log(attn[h] + 1e-10), axis=1)
        ax.plot(entropy, label=f'Head {h+1}', alpha=0.7)
    max_entropy = np.log(seq_len)
    ax.axhline(y=max_entropy, color='r', linestyle='--', label='Max entropy')
    ax.set_title('Attention Entropy (lower = more focused)')
    ax.set_xlabel('Query position')
    ax.set_ylabel('Entropy')
    ax.legend()
    
    # 4. Effective receptive field
    ax = axes[1, 1]
    center = seq_len // 2
    for h in range(n_heads):
        attn_row = attn[h, center, :]
        ax.plot(attn_row, label=f'Head {h+1}', linewidth=2)
    ax.axvline(x=center, color='r', linestyle='--', alpha=0.5)
    ax.set_title(f'Receptive Field from Center (pos {center})')
    ax.set_xlabel('Key position')
    ax.set_ylabel('Attention weight')
    ax.legend()
    
    plt.suptitle(f'{pde_type.capitalize()} Equation - Attention Analysis Summary', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'attention_summary_{pde_type}.png', dpi=150)
    plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pde', type=str, default='heat', choices=['heat', 'wave'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--layers', type=int, default=1, help='Number of attention layers')
    args = parser.parse_args()
    
    # Train
    model, losses, test_data = train(pde_type=args.pde, n_epochs=args.epochs, n_layers=args.layers)
    
    # Save model
    torch.save(model.state_dict(), f'model_{args.pde}.pt')
    
    # Visualizations
    print("\n=== Attention Patterns ===")
    attn = visualize_attention(model, test_data, args.pde)
    analyze_attention_locality(attn, args.pde)
    
    print("\n=== Attention Summary ===")
    visualize_what_attention_learns_summary(model, test_data, args.pde)
    
    print("\n=== Attention vs Distance ===")
    visualize_attention_vs_distance(model, test_data)
    
    print("\n=== Learned Stencil (Response to Delta) ===")
    compare_to_finite_difference(model, n_points=64)
    
    print("\n=== OOD Generalization Test ===")
    test_ood_generalization(model, args.pde)
    
    print("\n=== Rollout Comparison ===")
    visualize_rollout(model, test_data, args.pde)
    
    # Plot training loss
    plt.figure(figsize=(8, 4))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training Loss - {args.pde.capitalize()} Equation')
    plt.savefig(f'training_loss_{args.pde}.png', dpi=150)
    plt.show()

