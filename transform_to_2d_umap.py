#!/usr/bin/env python3
"""
Parameterized script to transform N-dimensional embeddings to 2D using UMAP
Works for any lens type (good_faith, excitement_directedness, etc.)
"""

# CRITICAL: Set ALL threading environment variables BEFORE any imports
import os
import sys
import argparse

# Disable TensorFlow completely for this script
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_DISABLE_MKL'] = '1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Force single-threading for all numerical libraries
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['NUMBA_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

# Prevent TensorFlow from being imported accidentally
class TensorFlowBlocker:
    def find_spec(self, name, path, target=None):
        if name.startswith('tensorflow'):
            raise ImportError(f"TensorFlow import blocked: {name}")
        return None

# Temporarily block TF imports
tf_blocker = TensorFlowBlocker()
sys.meta_path.insert(0, tf_blocker)

try:
    import numpy as np
    import pandas as pd
    import time
    import gc
    from umap import UMAP
    import joblib
    from sklearn.utils import resample
    from pathlib import Path

    print("All packages imported successfully without TensorFlow conflicts")

finally:
    # Remove the blocker after imports
    if tf_blocker in sys.meta_path:
        sys.meta_path.remove(tf_blocker)


def transform_to_2d_umap(lens_type='good_faith', n_dimensions=None):
    """
    Transform N-dimensional embeddings to 2D using UMAP

    Args:
        lens_type: The lens type (e.g., 'good_faith', 'excitement_directedness')
        n_dimensions: Number of input dimensions. If None, will infer from file.
    """

    # Determine number of dimensions if not provided
    if n_dimensions is None:
        # Try common dimension counts
        for dims in [2, 3]:
            potential_path = Path(f'~/Desktop/memedrive_experiments/output_data/{lens_type}_embeddings_{dims}d.npz').expanduser()
            if potential_path.exists():
                n_dimensions = dims
                break

        if n_dimensions is None:
            raise FileNotFoundError(f"Could not find embeddings file for {lens_type}")

    # File paths
    data_path = Path(f'~/Desktop/memedrive_experiments/output_data/{lens_type}_embeddings_{n_dimensions}d.npz').expanduser()
    model_output_dir = Path('~/Desktop/memedrive_experiments/trained_models').expanduser()
    model_output_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_output_dir / f'{lens_type}_umap_model.pkl'
    embeddings_output_path = Path(f'~/Desktop/memedrive_experiments/output_data/{lens_type}_embeddings_2d.npz').expanduser()

    print(f'Processing lens type: {lens_type}')
    print(f'Input dimensions: {n_dimensions}')

    # Load the embeddings
    print(f"Loading {n_dimensions}D embeddings from: {data_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {data_path}")

    data = np.load(data_path)
    embeddings = data['embeddings']
    print(f"Original embeddings shape: {embeddings.shape}")

    t1 = time.time()

    # Randomly sample embeddings for training UMAP (use fewer for higher dimensions)
    max_samples = 500000 if n_dimensions <= 3 else 200000
    n_samples = min(max_samples, len(embeddings))
    train_indices = resample(range(len(embeddings)), n_samples=n_samples, random_state=42)
    train_embeddings = embeddings[train_indices]
    print(f"Training UMAP on {len(train_embeddings)} samples")

    # Initialize and train UMAP (adjust parameters based on input dimensions)
    n_neighbors = min(30, max(10, n_samples // 10000))  # Adaptive neighbors
    min_dist = 0.05 if n_dimensions <= 3 else 0.1
    n_epochs = 500 if n_dimensions <= 3 else 200

    umap_model = UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric='cosine',
        n_epochs=n_epochs,
        random_state=42,
        verbose=True,
        low_memory=True
    )

    # Fit the model
    print(f"Training UMAP with {n_neighbors} neighbors, {min_dist} min_dist, {n_epochs} epochs...")
    umap_model.fit(train_embeddings)

    # Save the trained model
    joblib.dump(umap_model, model_path)
    print(f"Model saved to {model_path}")

    # Apply to all embeddings in batches to avoid memory issues
    batch_size = 50000 if n_dimensions <= 3 else 25000
    n_batches = (len(embeddings) + batch_size - 1) // batch_size
    print(f"Transforming all {len(embeddings)} embeddings in {n_batches} batches...")

    all_umap_embeddings = []
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(embeddings))
        batch = embeddings[start_idx:end_idx]
        batch_umap = umap_model.transform(batch)
        all_umap_embeddings.append(batch_umap)
        print(f"Processed batch {i+1}/{n_batches}")

        # Clean up memory
        if i % 10 == 0:
            gc.collect()

    # Concatenate all batches
    umap_embeddings = np.vstack(all_umap_embeddings)
    print(f"Final UMAP embeddings shape: {umap_embeddings.shape}")

    # Save the transformed embeddings
    np.savez_compressed(embeddings_output_path, embeddings=umap_embeddings)
    print(f"{lens_type} UMAP embeddings saved to {embeddings_output_path}")
    print(f'Completed in {time.time() - t1:.2f} seconds')

    return umap_embeddings


def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description='Transform N-dimensional embeddings to 2D using UMAP')
    parser.add_argument('--lens_type', type=str, default='good_faith',
                       help='Lens type to use (e.g., good_faith, excitement_directedness)')
    parser.add_argument('--n_dimensions', type=int, default=None,
                       help='Number of input dimensions (auto-detect if not specified)')

    args = parser.parse_args()

    print(f"Transforming {args.lens_type} embeddings to 2D...")
    embeddings = transform_to_2d_umap(args.lens_type, args.n_dimensions)
    print(f"\n✅ Successfully created {args.lens_type}_embeddings_2d.npz")


if __name__ == "__main__":
    main()