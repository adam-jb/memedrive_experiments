import pandas as pd
import numpy as np
import os
from pathlib import Path
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler
import time

# Paths
csv_path = Path('~/Desktop/memedrive_experiments/output_data/tweet_good_faith_ratings.csv').expanduser()
embeddings_path = Path('~/Desktop/memedrive_experiments/output_data/community_archive_embeddings.npz').expanduser()
output_dir = Path('~/Desktop/memedrive_experiments/output_data').expanduser()

print("Loading data...")
t1 = time.time()

# Load good faith ratings CSV
df = pd.read_csv(csv_path)
print(f"Loaded {len(df)} tweets with good faith ratings")

aa = df['original_index'].values
print(type(aa))
print(df.columns)

# Load all embeddings
all_embeddings = np.load(embeddings_path)['embeddings']
print(f'Embeddings loaded after {time.time() - t1} seconds')

# Get embeddings for our 25k sample using the 'index' column
sample_embeddings = all_embeddings[df['index'].values]
faith_labels = df[['sincerity', 'charity', 'constructiveness']].values

# Check for and remove NaN values
print(f"NaN values in faith labels: {np.isnan(faith_labels).sum()}")
nan_mask = np.isnan(faith_labels).any(axis=1)
if nan_mask.sum() > 0:
    print(f"Removing {nan_mask.sum()} rows with NaN values")
    sample_embeddings = sample_embeddings[~nan_mask]
    faith_labels = faith_labels[~nan_mask]
    print(f"Remaining samples: {len(sample_embeddings)}")


print("Running CCA...")

# Standardize and run CCA
scaler_embeddings = StandardScaler()
embeddings_scaled = scaler_embeddings.fit_transform(sample_embeddings)
faith_scaled = StandardScaler().fit_transform(faith_labels)

cca = CCA(n_components=3)
embedding_projections, faith_projections = cca.fit_transform(embeddings_scaled, faith_scaled)

# Print correlations
print("Canonical correlations:")
for i in range(3):
    corr = np.corrcoef(embedding_projections[:, i], faith_projections[:, i])[0, 1]
    print(f"  Dimension {i+1}: {corr:.4f}")

# Save transformation matrix and scaler
output_dir.mkdir(parents=True, exist_ok=True)

np.save(output_dir / 'good_faith_transform_matrix.npy', cca.x_weights_)
np.save(output_dir / 'embedding_scaler_mean.npy', scaler_embeddings.mean_)
np.save(output_dir / 'embedding_scaler_scale.npy', scaler_embeddings.scale_)

print("\nSaved:")
print("- good_faith_transform_matrix.npy")
print("- embedding_scaler_mean.npy")
print("- embedding_scaler_scale.npy")

print(f'transformation matrix made in {time.time() - t1} seconds')
t1 = time.time()

print('Applying to all 5.5m embeddings in batches!')
batch_size = 100_000
good_faith_embeddings_all = np.empty((all_embeddings.shape[0], 3), dtype=np.float16)

for i in range(0, len(all_embeddings), batch_size):
    end = min(i + batch_size, len(all_embeddings))
    standardized_batch = (all_embeddings[i:end] - scaler_embeddings.mean_) / scaler_embeddings.scale_
    good_faith_embeddings_all[i:end] = standardized_batch @ cca.x_weights_
    print(f"Processed batch {i}")

output_path = os.path.expanduser('~/Desktop/memedrive_experiments/output_data/good_faith_embeddings_3d.npz')
np.savez_compressed(output_path, good_faith_embeddings_all=good_faith_embeddings_all)
print(f"good_faith_embeddings_3d saved to {output_path}")
print(f'In {time.time() - t1} seconds')
