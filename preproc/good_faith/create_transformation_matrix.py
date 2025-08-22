import pandas as pd
import numpy as np
import os
from pathlib import Path
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
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

print("Correlations between faith dimensions:")
print(np.corrcoef(faith_labels.T))


print("Running CCA train and test procedure...")
scaler_embeddings = StandardScaler()
scaler_faith = StandardScaler()

X_train, X_test, y_train, y_test = train_test_split(sample_embeddings, faith_labels, test_size=0.2, random_state=42)

X_train_scaled = scaler_embeddings.fit_transform(X_train)
y_train_scaled = scaler_faith.fit_transform(y_train)



cca_test = CCA(n_components=3)
cca_test.fit(X_train_scaled, y_train_scaled)

# Test
X_test_scaled = scaler_embeddings.transform(X_test)
y_test_scaled = scaler_faith.transform(y_test)
predictions = X_test_scaled @ cca_test.x_weights_

print('Testing on 20% test set')
for i, feature in enumerate(['sincerity', 'charity', 'constructiveness']):
    corr = np.corrcoef(predictions[:, i], y_test_scaled[:, i])[0, 1]
    print(f"Corr of {corr:.4f} for {feature}")


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train_scaled, y_train_scaled)
lr_preds = lr.predict(X_test_scaled)

for i, feature in enumerate(['sincerity', 'charity', 'constructiveness']):
    corr = np.corrcoef(lr_preds[:, i], y_test[:, i])[0, 1]
    print(f"Linear regression: {corr:.4f} for {feature}")

for i, feature in enumerate(['sincerity', 'charity', 'constructiveness']):
    cca_1d = CCA(n_components=1)
    cca_1d.fit(X_train_scaled, y_train_scaled[:, [i]])

    pred_1d = X_test_scaled @ cca_1d.x_weights_
    corr = np.corrcoef(pred_1d.flatten(), y_test[:, i])[0, 1]
    print(f"1D CCA {feature}: {corr:.4f}")


from sklearn.linear_model import Ridge

# Test different regularization strengths
alphas = [0.1, 1.0, 10.0, 100.0]

for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train_scaled, y_train_scaled)
    ridge_preds = ridge.predict(X_test_scaled)

    print(f"\nRidge (alpha={alpha}):")
    for i, feature in enumerate(['sincerity', 'charity', 'constructiveness']):
        corr = np.corrcoef(ridge_preds[:, i], y_test[:, i])[0, 1]
        print(f"  {feature}: {corr:.4f}")


print('End of test: now making the main thing')



print("Running CCA...")
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
