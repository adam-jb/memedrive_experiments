# pip install umap-learn
import numpy as np
import pandas as pd
import os
from umap import UMAP
import joblib
from sklearn.utils import resample

# Load the embeddings
data = np.load(os.path.expanduser('~/Desktop/memedrive_experiments/output_data/community_archive_embeddings.npz'))
embeddings = data['embeddings']
print(f"Original embeddings shape: {embeddings.shape}")

# Randomly sample 500k embeddings for training
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
model_path = os.path.expanduser('~/Desktop/memedrive_experiments/trained_models/umap_model.pkl')
joblib.dump(umap_model, model_path)
print(f"Model saved to {model_path}")

# Apply to all 5.5m embeddings in batches to avoid memory issues
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
output_path = os.path.expanduser('~/Desktop/memedrive_experiments/output_data/umap_embeddings_2d.npz')
np.savez_compressed(output_path, umap_embeddings=umap_embeddings)
print(f"UMAP embeddings saved to {output_path}")
