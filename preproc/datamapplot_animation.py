# pip install datamapplot
## Seems to render really badly in browser: basically unusable in this current state
import numpy as np
import pandas as pd
import datamapplot
import time
import os
from matplotlib import cm

def create_interactive_tweet_timeline():
    """Use datamapplot for clean datetime animation"""

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

    # Data prep
    min_len = min(len(umap_embeddings), len(df))
    umap_embeddings = umap_embeddings[:min_len]
    df = df.iloc[:min_len].copy()

    df['x'] = umap_embeddings[:, 0]
    df['y'] = umap_embeddings[:, 1]
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.dropna(subset=['datetime'])
    df = df[df['datetime'] >= '2023-01-01']

    # Outlier pruning
    center_x = np.mean(df['x'])
    center_y = np.mean(df['y'])
    distances = np.sqrt((df['x'] - center_x)**2 + (df['y'] - center_y)**2)
    threshold = np.percentile(distances, 99)
    df = df[distances <= threshold]

    print(f"After processing: {len(df)} tweets")

    # Truncate text for hover
    df['hover_text'] = df['full_text'].apply(lambda x: x[:200] + "..." if len(x) > 200 else x)

    # Calculate distance from center for coloring
    df['distance_from_center'] = np.sqrt((df['x'] - center_x)**2 + (df['y'] - center_y)**2)

    # Convert colors to uint8 rgba format using viridis colormap
    normalized_distances = (df['distance_from_center'] - df['distance_from_center'].min()) / (df['distance_from_center'].max() - df['distance_from_center'].min())
    viridis = cm.get_cmap('viridis')
    colors = viridis(normalized_distances)

    # Create point_dataframe with required rgba columns, size=1, and week column
    point_dataframe = pd.DataFrame({
        'x': df['x'],
        'y': df['y'],
        'r': (colors[:, 0] * 255).astype(np.uint8),
        'g': (colors[:, 1] * 255).astype(np.uint8),
        'b': (colors[:, 2] * 255).astype(np.uint8),
        'a': (colors[:, 3] * 255).astype(np.uint8),
        'size': np.ones(len(df)),
        'week': df['datetime'].dt.isocalendar().week
    })

    # Use datamapplot with datetime histogram
    html_content = datamapplot.render_html(
        point_dataframe=point_dataframe,
        label_dataframe=point_dataframe,
        histogram_data=point_dataframe['week'],
        histogram_group_datetime_by='week',
        title="Tweet Embeddings Timeline",
        enable_search=True,
        search_field="hover_text",
        extra_point_data=df[['hover_text', 'distance_from_center']]
    )

    # Write HTML to file manually
    output_path = os.path.expanduser('~/Desktop/memedrive_experiments/visualisations/interactive_tweet_timeline_datamapplot.html')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"Interactive plot saved to: {output_path}")
    if os.path.exists(output_path):
        print(f"File size: ~{os.path.getsize(output_path) / 1024 / 1024:.1f}MB")

    return html_content

if __name__ == "__main__":
    create_interactive_tweet_timeline()
