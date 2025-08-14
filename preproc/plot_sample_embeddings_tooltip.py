import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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

    # Load parquet data
    t1 = time.time()
    df = pd.read_parquet('~/Desktop/memedrive_experiments/input_data/community_archive.parquet')
    print(f'seconds to read community archive file: {time.time() - t1}')

    # Ensure alignment
    min_len = min(len(umap_embeddings), len(df))
    umap_embeddings = umap_embeddings[:min_len]
    df = df.iloc[:min_len].copy()
    print(f"Aligned data length: {min_len}")

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

    # Get indices that correspond to the pruned data in original dataset
    original_indices = np.where(inlier_mask)[0]
    sample_indices = np.random.choice(len(original_indices), n_points, replace=False)
    final_indices = original_indices[sample_indices]

    # Get coordinates and text
    sampled_points = umap_embeddings[final_indices]
    sampled_text = df.iloc[final_indices]['full_text'].values

    x = sampled_points[:, 0]
    y = sampled_points[:, 1]

    # Color by density using distance from actual center
    colors = np.sqrt((x - center_x)**2 + (y - center_y)**2)

    # Create the plot with hover functionality
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(x, y, c=colors, alpha=0.6, s=1, cmap='viridis')
    plt.colorbar(scatter, label='Distance from center')
    ax.set_xlabel('UMAP Dimension 1')
    ax.set_ylabel('UMAP Dimension 2')
    ax.set_title(f'{n_points} Random Tweet Embeddings')
    ax.grid(True, alpha=0.3)

    # Add hover functionality
    annot = ax.annotate('', xy=(0,0), xytext=(20,20), textcoords="offset points",
                       bbox=dict(boxstyle="round", fc="w"),
                       arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)

    def update_annot(ind):
        pos = scatter.get_offsets()[ind["ind"][0]]
        annot.xy = pos
        text = sampled_text[ind["ind"][0]][:200] + "..." if len(sampled_text[ind["ind"][0]]) > 200 else sampled_text[ind["ind"][0]]
        annot.set_text(text)
        annot.get_bbox_patch().set_facecolor('yellow')
        annot.get_bbox_patch().set_alpha(0.8)

    def hover(event):
        if event.inaxes == ax:
            cont, ind = scatter.contains(event)
            if cont:
                update_annot(ind)
                annot.set_visible(True)
                fig.canvas.draw()
            else:
                annot.set_visible(False)
                fig.canvas.draw()

    fig.canvas.mpl_connect("motion_notify_event", hover)

    print(f"Plotted {n_points} points from {len(umap_embeddings)} total embeddings")
    print(f"X range: {x.min():.2f} to {x.max():.2f}")
    print(f"Y range: {y.min():.2f} to {y.max():.2f}")

    plt.show()

if __name__ == "__main__":
    create_random_scatter()
