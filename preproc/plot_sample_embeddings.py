import numpy as np
import matplotlib.pyplot as plt
import time
import os

def create_random_scatter():
    """Load UMAP embeddings and plot 10k random points from them"""

    # Load embeddings
    print("Loading UMAP embeddings...")
    t1 = time.time()
    embeddings_data = np.load(os.path.expanduser('~/Desktop/memedrive_experiments/output_data/umap_embeddings_2d.npz'))
    umap_embeddings = embeddings_data['umap_embeddings']
    print(f"Embeddings loaded in {time.time() - t1:.2f}s. Shape: {umap_embeddings.shape}")

    # Calculate actual center of the data
    center_x = np.mean(umap_embeddings[:, 0])
    center_y = np.mean(umap_embeddings[:, 1])

    # Prune top 1% outliers by distance from actual center
    distances = np.sqrt((umap_embeddings[:, 0] - center_x)**2 + (umap_embeddings[:, 1] - center_y)**2)
    threshold = np.percentile(distances, 99)
    inlier_mask = distances <= threshold
    pruned_embeddings = umap_embeddings[inlier_mask]

    print(f"Data center: ({center_x:.2f}, {center_y:.2f})")
    print(f"Pruned {len(umap_embeddings) - len(pruned_embeddings)} outliers (top 1%)")
    print(f"Remaining embeddings: {len(pruned_embeddings)}")

    # Sample 10k random points from pruned data
    n_points = min(10000, len(pruned_embeddings))
    indices = np.random.choice(len(pruned_embeddings), n_points, replace=False)
    sampled_points = pruned_embeddings[indices]

    x = sampled_points[:, 0]
    y = sampled_points[:, 1]

    # Color by density using distance from actual center
    colors = np.sqrt((x - center_x)**2 + (y - center_y)**2)

    # Create the plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(x, y, c=colors, alpha=0.6, s=1, cmap='viridis')
    plt.colorbar(scatter, label='Distance from center')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.title(f'{n_points} Random Tweet Embeddings')
    plt.grid(True, alpha=0.3)

    print(f"Plotted {n_points} points from {len(umap_embeddings)} total embeddings")
    print(f"X range: {x.min():.2f} to {x.max():.2f}")
    print(f"Y range: {y.min():.2f} to {y.max():.2f}")

    plt.show()

if __name__ == "__main__":
    create_random_scatter()
