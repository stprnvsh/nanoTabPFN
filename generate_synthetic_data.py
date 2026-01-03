"""
Generate synthetic HDF5 data similar to 300k_150x5_2.h5 but larger.
Usage: python generate_synthetic_data.py --scale 10
"""
import argparse
import h5py
import numpy as np

def generate_synthetic_h5(output_path: str, num_samples: int, max_seq: int = 150, num_features: int = 5, num_classes: int = 2):
    """
    Generate synthetic tabular data mimicking the original prior dump.
    
    Args:
        output_path: Output .h5 file path
        num_samples: Number of synthetic datasets
        max_seq: Maximum sequence length (rows per dataset)
        num_features: Number of features
        num_classes: Number of classes
    """
    print(f"Generating {num_samples:,} samples -> {output_path}")
    print(f"Shape: X=({num_samples}, {max_seq}, {num_features}), y=({num_samples}, {max_seq})")
    
    # Estimate file size
    size_gb = (num_samples * max_seq * num_features * 4 + num_samples * max_seq * 4) / 1e9
    print(f"Estimated size: {size_gb:.2f} GB")
    
    with h5py.File(output_path, 'w') as f:
        # Create datasets with chunking for efficient access
        chunk_size = min(1000, num_samples)
        
        X = f.create_dataset('X', shape=(num_samples, max_seq, num_features), 
                            dtype='float32', chunks=(chunk_size, max_seq, num_features))
        y = f.create_dataset('y', shape=(num_samples, max_seq), 
                            dtype='float32', chunks=(chunk_size, max_seq))
        
        num_datapoints = f.create_dataset('num_datapoints', shape=(num_samples,), dtype='int32')
        num_features_arr = f.create_dataset('num_features', shape=(num_samples,), dtype='int32')
        single_eval_pos = f.create_dataset('single_eval_pos', shape=(num_samples,), dtype='int32')
        
        f.create_dataset('max_num_classes', data=[num_classes], dtype='int64')
        f.create_dataset('original_batch_size', data=[32], dtype='int64')
        f.create_dataset('problem_type', data='classification')
        
        # Generate in batches to manage memory
        batch_size = 10000
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            n = end - start
            
            # Random features: standard normal N(0,1) like original
            X_batch = np.random.randn(n, max_seq, num_features).astype(np.float32)
            
            # Random binary/multiclass labels
            y_batch = np.random.randint(0, num_classes, size=(n, max_seq)).astype(np.float32)
            
            # Train/test split: 10%-90% of sequence (matches original 15-134 for 150 rows)
            min_split = max(1, int(0.1 * max_seq))
            max_split = int(0.9 * max_seq)
            split_pos = np.random.randint(min_split, max_split, size=n).astype(np.int32)
            
            # Varying number of features (1 to num_features) like original
            nf = np.random.randint(1, num_features + 1, size=n).astype(np.int32)
            
            # All samples use full sequence length
            ndp = np.full(n, max_seq, dtype=np.int32)
            
            X[start:end] = X_batch
            y[start:end] = y_batch
            num_datapoints[start:end] = ndp
            num_features_arr[start:end] = nf
            single_eval_pos[start:end] = split_pos
            
            pct = 100 * end / num_samples
            print(f"  {end:,}/{num_samples:,} ({pct:.0f}%)", end='\r')
        
        print(f"\nDone! Saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=int, default=10, help="Scale factor for num_samples (300k base)")
    parser.add_argument("--samples", type=int, default=None, help="Override: exact number of samples")
    parser.add_argument("--rows", type=int, default=150, help="Rows per sample (default 150)")
    parser.add_argument("--features", type=int, default=5, help="Features per sample (default 5)")
    parser.add_argument("--classes", type=int, default=2, help="Number of classes (default 2)")
    parser.add_argument("--output", type=str, default=None, help="Output path")
    args = parser.parse_args()
    
    num_samples = args.samples if args.samples else 300_000 * args.scale
    output = args.output or f"{num_samples // 1000}k_{args.rows}x{args.features}_{args.classes}.h5"
    
    generate_synthetic_h5(output, num_samples, max_seq=args.rows, num_features=args.features, num_classes=args.classes)

