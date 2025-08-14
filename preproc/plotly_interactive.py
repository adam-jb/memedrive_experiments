 # pip install plotly
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import time
import os

def create_interactive_tweet_timeline():
    """Convert matplotlib animation to interactive Plotly timeline"""

    # Load embeddings (same as your original)
    print("Loading UMAP embeddings...")
    t1 = time.time()
    embeddings_data = np.load(os.path.expanduser('~/Desktop/memedrive_experiments/output_data/umap_embeddings_2d.npz'))
    umap_embeddings = embeddings_data['umap_embeddings']
    print(f"Embeddings loaded in {time.time() - t1:.2f}s. Shape: {umap_embeddings.shape}")

    # Load parquet data
    t1 = time.time()
    df = pd.read_parquet('~/Desktop/memedrive_experiments/input_data/community_archive.parquet')
    print(f'seconds to read community archive file: {time.time() - t1}')

    # Data prep (identical to your original)
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

    # Create weekly bins
    df['week'] = df['datetime'].dt.to_period('W')
    df['week_start'] = df['week'].dt.start_time
    df['distance_from_center'] = np.sqrt((df['x'] - center_x)**2 + (df['y'] - center_y)**2)

    # Truncate text for hover (Plotly handles this better than matplotlib)
    df['hover_text'] = df['full_text'].apply(lambda x: x[:200] + "..." if len(x) > 200 else x)

    # Create the main scatter plot
    fig = go.Figure()

    # Add traces for each week (this creates the timeline slider automatically)
    weeks = sorted(df['week_start'].unique())

    for i, week in enumerate(weeks):
        week_data = df[df['week_start'] == week]

        fig.add_trace(
            go.Scatter(
                x=week_data['x'],
                y=week_data['y'],
                mode='markers',
                marker=dict(
                    size=3,
                    color=week_data['distance_from_center'],
                    colorscale='Viridis',
                    showscale=True if i == 0 else False,  # Only show colorbar once
                    colorbar=dict(title="Distance from Center") if i == 0 else None
                ),
                text=week_data['hover_text'],
                hovertemplate='<b>%{text}</b><br>x: %{x:.2f}<br>y: %{y:.2f}<extra></extra>',
                name=f"Week {week.strftime('%Y-%m-%d')}",
                visible=True if i == 0 else False  # Only show first week initially
            )
        )

    # Create slider steps
    steps = []
    for i, week in enumerate(weeks):
        week_data = df[df['week_start'] == week]
        step = dict(
            method="update",
            args=[
                {"visible": [False] * len(weeks)},  # Hide all traces
                {"title": f"Tweet Embeddings - Week of {week.strftime('%Y-%m-%d')} | {len(week_data)} tweets"}
            ],
            label=week.strftime('%m/%d')
        )
        step["args"][0]["visible"][i] = True  # Show only current week
        steps.append(step)

    # Add slider
    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Week: "},
        pad={"t": 50},
        steps=steps
    )]

    # Add play button functionality
    fig.update_layout(
        sliders=sliders,
        title=f"Tweet Embeddings - Week of {weeks[0].strftime('%Y-%m-%d')} | {len(df[df['week_start'] == weeks[0]])} tweets",
        xaxis_title="UMAP Dimension 1",
        yaxis_title="UMAP Dimension 2",
        width=1200,
        height=800,
        updatemenus=[
            dict(
                type="buttons",
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[None, {"frame": {"duration": 500, "redraw": True},
                                   "fromcurrent": True, "transition": {"duration": 100}}]
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[[None], {"frame": {"duration": 0, "redraw": False},
                                     "mode": "immediate", "transition": {"duration": 0}}]
                    )
                ],
                direction="left",
                pad={"r": 10, "t": 87},
                showactive=False,
                x=0.1,
                xanchor="right",
                y=0,
                yanchor="top"
            )
        ]
    )

    # Create animation frames (for the play button)
    frames = []
    for i, week in enumerate(weeks):
        week_data = df[df['week_start'] == week]
        frame = go.Frame(
            data=[
                go.Scatter(
                    x=week_data['x'],
                    y=week_data['y'],
                    mode='markers',
                    marker=dict(
                        size=3,
                        color=week_data['distance_from_center'],
                        colorscale='Viridis'
                    ),
                    text=week_data['hover_text'],
                    hovertemplate='<b>%{text}</b><br>x: %{x:.2f}<br>y: %{y:.2f}<extra></extra>'
                )
            ],
            layout=go.Layout(
                title=f"Tweet Embeddings - Week of {week.strftime('%Y-%m-%d')} | {len(week_data)} tweets"
            ),
            name=str(i)
        )
        frames.append(frame)

    fig.frames = frames

    # Save as standalone HTML
    output_path = os.path.expanduser('~/Desktop/memedrive_experiments/visualisations/interactive_tweet_timeline.html')
    fig.write_html(
        output_path,
        include_plotlyjs=True,  # Embeds Plotly.js directly in the file
        config={
            'displayModeBar': True,
            'toImageButtonOptions': {
                'format': 'png',
                'filename': 'tweet_timeline',
                'height': 800,
                'width': 1200,
                'scale': 2
            }
        }
    )

    print(f"Interactive plot saved to: {output_path}")
    print(f"File size: ~{os.path.getsize(output_path) / 1024 / 1024:.1f}MB")

    # Also show in browser/notebook
    fig.show()

    return fig

if __name__ == "__main__":
    create_interactive_tweet_timeline()
