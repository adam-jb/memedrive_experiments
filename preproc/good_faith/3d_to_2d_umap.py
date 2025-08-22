# CRITICAL: Set ALL threading environment variables BEFORE any imports
# All this is in service of dealing with a conflict between numba (which UMAP uses) and tensorflow (which is initializing its threading system on import)
import os
import sys

# Disable TensorFlow completely for this script
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'           # Force CPU only
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'            # Suppress all TF logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'           # Disable oneDNN
os.environ['TF_DISABLE_MKL'] = '1'                  # Disable MKL-DNN
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'    # Prevent GPU memory allocation

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

    print("All packages imported successfully without TensorFlow conflicts")

finally:
    # Remove the blocker after imports
    if tf_blocker in sys.meta_path:
        sys.meta_path.remove(tf_blocker)

print('Starting')

# Load the 3D embeddings
data_path = os.path.expanduser('~/Desktop/memedrive_experiments/output_data/good_faith_embeddings_3d.npz')
data = np.load(data_path)
embeddings = data['good_faith_embeddings_all']
print(f"Original embeddings shape: {embeddings.shape}")

t1 = time.time()

# Randomly sample 500k embeddings for training UMAP
n_samples = min(500000, len(embeddings))
train_indices = resample(range(len(embeddings)), n_samples=n_samples, random_state=42)
train_embeddings = embeddings[train_indices]
print(f"Training on {len(train_embeddings)} samples")

# Initialize and train UMAP
umap_model = UMAP(
    n_components=2,
    n_neighbors=30,
    min_dist=0.05,
    metric='cosine',
    n_epochs=500,  # regular UMAP can handle more epochs efficiently
    random_state=42,
    verbose=True,
    low_memory=True  # helps with large datasets
)

# Fit the model
print("Training UMAP...")
umap_model.fit(train_embeddings)

# Save the trained model
os.makedirs(os.path.expanduser('~/Desktop/memedrive_experiments/trained_models'), exist_ok=True)
model_path = os.path.expanduser('~/Desktop/memedrive_experiments/trained_models/good_faith_umap_model.pkl')
joblib.dump(umap_model, model_path)
print(f"Model saved to {model_path}")

# Apply to all embeddings in batches to avoid memory issues
batch_size = 50000
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

# Concatenate all batches
umap_embeddings = np.vstack(all_umap_embeddings)
print(f"Final UMAP embeddings shape: {umap_embeddings.shape}")

# Save the transformed embeddings
output_path = os.path.expanduser('~/Desktop/memedrive_experiments/output_data/good_faith_umap_embeddings_2d.npz')
np.savez_compressed(output_path, good_faith_umap_embeddings=umap_embeddings)
print(f"Good faith UMAP embeddings saved to {output_path}")
print(f'Completed in {time.time() - t1} seconds')
