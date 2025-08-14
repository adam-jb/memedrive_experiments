import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import os
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button

def create_tweet_animation():
    """Load embeddings and create animated weekly scatter plot"""

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

    # Add coordinates to dataframe
    df['x'] = umap_embeddings[:, 0]
    df['y'] = umap_embeddings[:, 1]

    # Convert datetime and filter from 2023-01-01
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.dropna(subset=['datetime'])
    df = df[df['datetime'] >= '2023-01-01']
    print(f"After filtering from 2023-01-01: {len(df)} tweets")

    # Calculate actual center of all data
    center_x = np.mean(df['x'])
    center_y = np.mean(df['y'])

    # Prune top 1% outliers by distance from actual center
    distances = np.sqrt((df['x'] - center_x)**2 + (df['y'] - center_y)**2)
    threshold = np.percentile(distances, 99)
    df = df[distances <= threshold]

    print(f"Data center: ({center_x:.2f}, {center_y:.2f})")
    print(f"After pruning outliers: {len(df)} tweets")

    # Create weekly bins
    df['week'] = df['datetime'].dt.to_period('W')
    week_counts = df['week'].value_counts().sort_index()
    print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    print(f"Total weeks: {len(week_counts)}")

    # Prepare data for animation
    weeks = sorted(df['week'].unique())
    weekly_data = {}

    for week in weeks:
        week_df = df[df['week'] == week]
        weekly_data[week] = {
            'x': week_df['x'].values,
            'y': week_df['y'].values,
            'text': week_df['full_text'].values,
            'colors': np.sqrt((week_df['x'] - center_x)**2 + (week_df['y'] - center_y)**2)
        }

    print(f"Prepared {len(weeks)} weeks of data")

    # Set up the plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Set fixed axis limits based on all data
    margin = 0.1
    x_range = df['x'].max() - df['x'].min()
    y_range = df['y'].max() - df['y'].min()
    ax.set_xlim(df['x'].min() - margin * x_range, df['x'].max() + margin * x_range)
    ax.set_ylim(df['y'].min() - margin * y_range, df['y'].max() + margin * y_range)

    ax.set_xlabel('UMAP Dimension 1')
    ax.set_ylabel('UMAP Dimension 2')
    ax.grid(True, alpha=0.3)

    # Initialize scatter plot
    scatter = ax.scatter([], [], c=[], alpha=0.6, s=1, cmap='viridis')
    title = ax.set_title('')

    # Animation control variables
    class AnimationController:
        def __init__(self):
            self.paused = False
            self.reverse = False
            self.current_frame = 0
            self.frame_step = 1

    controller = AnimationController()

    # Hover annotation
    annot = ax.annotate('', xy=(0,0), xytext=(20,20), textcoords="offset points",
                       bbox=dict(boxstyle="round", fc="w"),
                       arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)

    current_data = {'x': [], 'y': [], 'text': []}

    def update_annot(ind, x_data, y_data, text_data):
        if len(x_data) > 0 and ind < len(x_data):
            pos = (x_data[ind], y_data[ind])
            annot.xy = pos
            text = text_data[ind][:200] + "..." if len(text_data[ind]) > 200 else text_data[ind]
            annot.set_text(text)
            annot.get_bbox_patch().set_facecolor('yellow')
            annot.get_bbox_patch().set_alpha(0.8)

    def hover(event):
        if event.inaxes == ax and len(current_data['x']) > 0:
            # Simple distance-based hover detection
            if event.xdata is not None and event.ydata is not None:
                distances = np.sqrt((np.array(current_data['x']) - event.xdata)**2 +
                                  (np.array(current_data['y']) - event.ydata)**2)
                if len(distances) > 0:
                    closest_idx = np.argmin(distances)
                    if distances[closest_idx] < 0.5:  # Hover threshold
                        update_annot(closest_idx, current_data['x'], current_data['y'], current_data['text'])
                        annot.set_visible(True)
                        fig.canvas.draw_idle()
                        return

            annot.set_visible(False)
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", hover)

    def animate(frame):
        if controller.paused:
            return scatter,

        # Calculate actual frame index
        if controller.reverse:
            frame_idx = len(weeks) - 1 - (frame % len(weeks))
        else:
            frame_idx = frame % len(weeks)

        controller.current_frame = frame_idx
        week = weeks[frame_idx]
        data = weekly_data[week]

        # Update current data for hover
        current_data['x'] = data['x']
        current_data['y'] = data['y']
        current_data['text'] = data['text']

        # Update scatter plot
        if len(data['x']) > 0:
            scatter.set_offsets(np.column_stack([data['x'], data['y']]))
            scatter.set_array(data['colors'])
        else:
            scatter.set_offsets(np.empty((0, 2)))
            scatter.set_array(np.array([]))

        title.set_text(f'Week of {week.start_time.strftime("%Y-%m-%d")} | {len(data["x"])} tweets')

        return scatter,

    # Control buttons
    def toggle_pause(event):
        controller.paused = not controller.paused
        pause_button.label.set_text('Play' if controller.paused else 'Pause')
        fig.canvas.draw_idle()

    def toggle_reverse(event):
        controller.reverse = not controller.reverse
        reverse_button.label.set_text('Forward' if controller.reverse else 'Reverse')
        fig.canvas.draw_idle()

    # Add control buttons
    ax_pause = plt.axes([0.02, 0.02, 0.08, 0.04])
    ax_reverse = plt.axes([0.12, 0.02, 0.08, 0.04])

    pause_button = Button(ax_pause, 'Pause')
    reverse_button = Button(ax_reverse, 'Reverse')

    pause_button.on_clicked(toggle_pause)
    reverse_button.on_clicked(toggle_reverse)

    # Create animation
    anim = FuncAnimation(fig, animate, frames=len(weeks)*2, interval=100,
                        blit=False, repeat=True)

    # Save animation as MP4
    print("Saving animation as MP4...")
    anim.save(os.path.expanduser('~/Desktop/memedrive_experiments/visualisations/tweet_animation.gif'), writer='pillow', fps=10, dpi=100)
    print("Animation saved as tweet_animation.mp4!")

    # This opens a new window and lets you walk through the animation frame by frame
    plt.tight_layout()
    plt.show()

    return anim

if __name__ == "__main__":
    create_tweet_animation()
