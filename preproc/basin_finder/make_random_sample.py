import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from pathlib import Path
import random

def generate_power_law_count(max_val=1000, zero_prob=0.75, alpha=2.5):
    """
    Generate a power law distributed count where most values are 0.

    Parameters:
    - max_val: Maximum possible count value
    - zero_prob: Probability of getting 0 (most tweets have no engagement)
    - alpha: Power law exponent (higher = more skewed toward low values)

    Returns:
    - Integer count following power law distribution
    """
    if np.random.random() < zero_prob:
        return 0
    else:
        # Generate from power law distribution for non-zero values
        # Use Pareto distribution: x = (1/u)^(1/alpha) where u is uniform[0,1]
        u = np.random.random()
        raw_val = (1.0 / u) ** (1.0 / alpha)
        # Scale and cap to max_val
        scaled_val = int(min(raw_val * 10, max_val))
        return max(1, scaled_val)  # Ensure at least 1 if not 0

def generate_tweet_embeddings(n_samples=5000, n_clusters=5, noise_ratio=0.15, seed=42):
    """
    Generate fake tweet data with 2D embeddings that have realistic clustering behavior.

    Parameters:
    - n_samples: Number of tweet samples to generate
    - n_clusters: Number of distinct clusters in embedding space
    - noise_ratio: Proportion of points that should be noise (not in clusters)
    - seed: Random seed for reproducibility

    Returns:
    - pd.DataFrame with datetime, full_text, retweet_count, favourite_count, e_x, e_y columns
    """
    np.random.seed(seed)
    random.seed(seed)

    # Define date range
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 2, 28)
    date_range = (end_date - start_date).days

    # Generate cluster centers that are well-separated
    ##!! This makes 5 basins
    cluster_centers = np.array([
        [-2.5, -2.0],  # Bottom left cluster
        [2.8, -1.8],   # Bottom right cluster
        [-0.2, 2.5],   # Top center cluster
        [-3.2, 1.2],   # Left cluster
        [2.2, 2.8]     # Top right cluster
    ])[:n_clusters]

    # Sample tweet themes for each cluster (for realistic full_text generation)
    cluster_themes = [
        ["crypto", "bitcoin", "trading", "blockchain", "defi"],
        ["politics", "election", "policy", "government", "vote"],
        ["tech", "AI", "startup", "innovation", "coding"],
        ["sports", "game", "team", "player", "championship"],
        ["entertainment", "movie", "music", "celebrity", "tv"]
    ][:n_clusters]

    data = []

    # Calculate number of points per cluster and noise points
    n_noise = int(n_samples * noise_ratio)
    n_cluster_points = n_samples - n_noise
    points_per_cluster = n_cluster_points // n_clusters

    # Generate clustered points
    for cluster_idx in range(n_clusters):
        center = cluster_centers[cluster_idx]
        themes = cluster_themes[cluster_idx]

        for _ in range(points_per_cluster):
            # Generate timestamp
            days_offset = np.random.randint(0, date_range + 1)
            hours_offset = np.random.randint(0, 24)
            minutes_offset = np.random.randint(0, 60)
            timestamp = start_date + timedelta(days=days_offset, hours=hours_offset, minutes=minutes_offset)

            # Add slight temporal drift to cluster centers (very minimal)
            time_progress = days_offset / date_range  # 0 to 1
            temporal_drift = np.array([
                0.1 * np.sin(2 * np.pi * time_progress) * (cluster_idx - n_clusters/2) * 0.1,
                0.1 * np.cos(2 * np.pi * time_progress) * (cluster_idx - n_clusters/2) * 0.1
            ])

            # Generate point around cluster center with some variance
            cluster_variance = 0.8  # Standard deviation for cluster spread
            embedding = np.random.normal(center + temporal_drift, cluster_variance)

            # Generate realistic tweet text
            theme = random.choice(themes)
            tweet_templates = [
                f"Just thinking about {theme} and how it's changing everything 🚀",
                f"Hot take: {theme} is going to be huge this year #trending",
                f"Anyone else excited about the latest {theme} developments?",
                f"The {theme} community is amazing! Love the discussions here",
                f"Breaking: major news in the {theme} space today!",
                f"My thoughts on {theme}: this is just the beginning...",
                f"Why {theme} matters more than people realize",
                f"Interesting perspective on {theme} from today's events"
            ]
            full_text = random.choice(tweet_templates)

            # Generate power law distributed engagement counts
            retweet_count = generate_power_law_count(max_val=1000, zero_prob=0.75)
            favourite_count = generate_power_law_count(max_val=1000, zero_prob=0.70)

            data.append({
                'datetime': timestamp,
                'full_text': full_text,
                'retweet_count': retweet_count,
                'favourite_count': favourite_count,
                'e_x': embedding[0],
                'e_y': embedding[1]
            })

    # Generate noise points (scattered randomly)
    for _ in range(n_noise):
        # Random timestamp
        days_offset = np.random.randint(0, date_range + 1)
        hours_offset = np.random.randint(0, 24)
        minutes_offset = np.random.randint(0, 60)
        timestamp = start_date + timedelta(days=days_offset, hours=hours_offset, minutes=minutes_offset)

        # Random embedding in larger space (noise)
        embedding = np.random.uniform(-6, 6, 2)  # Wider range for noise

        # Random tweet text
        noise_themes = ["random", "thoughts", "life", "coffee", "weather", "mood", "day"]
        theme = random.choice(noise_themes)
        noise_templates = [
            f"Just had a thought about {theme}...",
            f"Random observation: {theme} is weird",
            f"Why does {theme} always happen to me?",
            f"Today's {theme} report: could be better",
            f"Unpopular opinion about {theme}",
            f"Does anyone else think {theme} is overrated?",
            f"{theme} update: still confused"
        ]
        full_text = random.choice(noise_templates)

        # Generate power law distributed engagement counts (noise tweets typically get less engagement)
        retweet_count = generate_power_law_count(max_val=500, zero_prob=0.85)
        favourite_count = generate_power_law_count(max_val=500, zero_prob=0.80)

        data.append({
            'datetime': timestamp,
            'full_text': full_text,
            'retweet_count': retweet_count,
            'favourite_count': favourite_count,
            'e_x': embedding[0],
            'e_y': embedding[1]
        })

    # Create DataFrame and sort by datetime
    df = pd.DataFrame(data)
    df = df.sort_values('datetime').reset_index(drop=True)

    # Add some additional embedding dimensions if needed (starting with 'e')
    # For now, keeping it 2D as requested, but this is where you'd add e_z, e_w, etc.

    return df

def save_embeddings_data(df, output_dir="~/Desktop/memedrive_experiments/output_data/basin_finder/"):
    """
    Save the embeddings DataFrame to the specified directory.
    """
    # Expand user path and create directory if it doesn't exist
    output_path = Path(output_dir).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)

    # Save without timestamp in filename
    output_file = output_path / "dummy_tweet_embeddings.csv"
    df.to_csv(output_file, index=False)

    print(f"Saved embeddings data to: {output_file}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    print(f"Embedding columns: {[col for col in df.columns if col.startswith('e')]}")

    return output_file

# Main execution function
def create_tweet_embeddings_dataset():
    """
    Generate and save the complete tweet embeddings dataset.
    """
    print("Generating tweet embeddings dataset...")

    # Generate the data
    df = generate_tweet_embeddings(
        n_samples=100_000,
        n_clusters=5,
        noise_ratio=0.15,
        seed=42
    )

    # Display some statistics
    print(f"\nDataset Statistics:")
    print(f"Total samples: {len(df)}")
    print(f"Date range: {df['datetime'].min().date()} to {df['datetime'].max().date()}")
    print(f"Embedding range - X: [{df['e_x'].min():.2f}, {df['e_x'].max():.2f}]")
    print(f"Embedding range - Y: [{df['e_y'].min():.2f}, {df['e_y'].max():.2f}]")
    print(f"Retweet counts - Mean: {df['retweet_count'].mean():.1f}, Max: {df['retweet_count'].max()}, Zeros: {(df['retweet_count'] == 0).sum()}")
    print(f"Favourite counts - Mean: {df['favourite_count'].mean():.1f}, Max: {df['favourite_count'].max()}, Zeros: {(df['favourite_count'] == 0).sum()}")

    # Show sample data
    print(f"\nSample data:")
    print(df.head())

    # Save the data
    output_file = save_embeddings_data(df)

    return df, output_file

if __name__ == "__main__":
    df, output_file = create_tweet_embeddings_dataset()
