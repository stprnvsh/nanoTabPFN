# PDE Attention Study

Understanding what a single-layer transformer learns on PDE data.

## Structure

```
pde_attention_study/
├── pde_data.py           # PDE data generation (heat, wave, OOD ICs)
├── model.py              # Single-layer attention model
├── train_and_visualize.py # Training and visualization
└── README.md
```

## Usage

```bash
cd pde_attention_study
python train_and_visualize.py --pde heat --epochs 100
```

## Visualizations Generated

| File | What it shows |
|------|---------------|
| `attention_patterns_*.png` | Per-head attention heatmaps |
| `attention_summary_*.png` | Diagonal dominance, entropy, receptive field |
| `attention_vs_distance.png` | Attention weight vs spatial distance |
| `learned_stencil.png` | Response to delta functions (shows learned FD stencil) |
| `ood_generalization.png` | Test on step/gaussian/triangle/multi-bump ICs |
| `rollout_*.png` | Ground truth vs predicted time evolution |

## OOD Testing

Model is trained on **sine wave** initial conditions, then tested on:
- Step functions
- Gaussian bumps  
- Triangle waves
- Multi-bump patterns

This reveals if attention learns the **underlying PDE operator** vs just memorizing training ICs.

## Key Questions

1. Does attention learn **local** patterns (finite-difference-like)?
2. What is the **effective receptive field**?
3. Does it **generalize** to unseen initial conditions?

