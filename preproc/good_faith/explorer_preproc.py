import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime, timedelta

def calculate_tweet_size(favorite_count):
    """
    Calculate tweet diameter based on favorite count.
    Area is proportional to favorites, so diameter ∝ √(favorites)

    Target mapping:
    - 0 favorites → diameter = 1 unit
    - 100 favorites → diameter = 10 units
    - 10000 favorites → diameter = 100 units
    """
    if favorite_count == 0:
        return 1.0

    # Use logarithmic-ish scaling to match the target points
    # This approximates the desired mapping reasonably well
    diameter = np.sqrt(favorite_count) * 0.9 + 1.0
    return max(1.0, diameter)

def get_fadeout_multiplier(age_days):
    """
    Step function fadeout over 6 days.
    Returns (opacity_multiplier, size_multiplier)
    """
    if age_days == 0:
        return 1.0, 1.0  # Current day: full opacity and size
    elif age_days == 1:
        return 0.8, 0.5  # Yesterday: 80% opacity, 90% size
    elif age_days == 2:
        return 0.6, 0.25  # 2 days ago
    elif age_days == 3:
        return 0.4, 0.15  # 3 days ago
    elif age_days == 4:
        return 0.2, 0.1  # 4 days ago
    else:
        return 0.0, 0.0  # 6+ days ago: invisible

def create_daily_frames(df):
    """
    Create daily frames with fadeout logic.
    Each frame contains tweets from that day + previous 5 days with fadeout.

    Returns dict: {'2023-01-01': [tweet_objects], '2023-01-02': [...], ...}
    """
    # Group tweets by date
    df['date'] = df['datetime'].dt.date
    grouped = df.groupby('date')

    # Get all unique dates and sort them
    all_dates = sorted(df['date'].unique())

    frames = {}

    print(f"Processing {len(all_dates)} days of data...")

    for i, current_date in enumerate(all_dates):
        if i % 30 == 0:  # Progress indicator
            print(f"Processing day {i+1}/{len(all_dates)}: {current_date}")

        ##!! HACK TO LIMIT FILE SIZE
        if i > 50:
            return frames

        frame_tweets = []

        # Look back up to 6 days (including current day)
        for days_back in range(6):
            target_date = current_date - timedelta(days=days_back)

            if target_date not in grouped.groups:
                continue  # No tweets on this date

            day_tweets = grouped.get_group(target_date)
            opacity_mult, size_mult = get_fadeout_multiplier(days_back)

            if opacity_mult == 0:  # Skip invisible tweets
                continue

            # Process each tweet for this day
            for _, tweet in day_tweets.iterrows():
                base_size = calculate_tweet_size(tweet['favorite_count'])
                final_size = base_size * size_mult

                tweet_obj = {
                    'x': float(tweet['x']),
                    'y': float(tweet['y']),
                    'full_text': str(tweet['full_text'])[:500],  # Truncate very long tweets
                    'favorite_count': int(tweet['favorite_count']),
                    'retweet_count': int(tweet.get('retweet_count', 0)),  # Handle missing retweets
                    'screen_name': str(tweet['screen_name']),
                    'age_days': days_back,
                    'base_size': float(base_size),
                    'final_size': float(final_size),
                    'opacity': float(opacity_mult),
                    'tweet_date': str(target_date),
                    'current_date': str(current_date)
                }

                frame_tweets.append(tweet_obj)

        frames[str(current_date)] = frame_tweets

    print(f"Completed processing. Generated {len(frames)} daily frames.")
    return frames

def preprocess_tweet_data():
    """
    Main preprocessing function that loads, cleans, and outputs tweet data.
    """
    print("Starting tweet data preprocessing...")

    # 1. Load coordinates
    coords_path = Path('~/Desktop/memedrive_experiments/output_data/good_faith_embeddings_3d.npz').expanduser()
    print(f"Loading coordinates from {coords_path}")
    coords = np.load(coords_path)
    print(coords.keys())
    embeddings = coords['good_faith_embeddings_all']  # Shape: (n_tweets, 2)
    print(f"Loaded {len(embeddings)} coordinate pairs")

    ## Get errors in calc if leave it as float16 (as it was saved as)
    embeddings = embeddings.astype(np.float64)

    print('Overview of embeddings:')
    print(pd.Series(embeddings[:, 1]).describe())
    count_na = np.sum(np.isnan(embeddings[:, 1]))
    count_inf = np.sum(np.isinf(embeddings[:, 1]))
    print(f"Count of NaNs: {count_na}")
    print(f"Count of infs: {count_inf}")

    # 2. Load tweet metadata
    metadata_path = Path('~/Desktop/memedrive_experiments/input_data/community_archive.parquet').expanduser()
    print(f"Loading metadata from {metadata_path}")
    df = pd.read_parquet(metadata_path)
    print(f"Loaded {len(df)} tweet records")

    # 3. Align data (ensure same number of rows)
    if len(df) != len(embeddings):
        min_len = min(len(df), len(embeddings))
        print(f"Warning: Mismatch between coordinates ({len(embeddings)}) and metadata ({len(df)})")
        print(f"Truncating both to {min_len} records")
        df = df.iloc[:min_len]
        embeddings = embeddings[:min_len]

    # 4. Add coordinates to dataframe
    df['x'] = embeddings[:, 0]
    df['y'] = embeddings[:, 1]

    # 5. Filter and clean
    df['datetime'] = pd.to_datetime(df['datetime'])
    initial_count = len(df)
    df = df[df['datetime'] >= '2023-01-01']
    print(f"Filtered to tweets after 2023-01-01: {len(df)} tweets (removed {initial_count - len(df)})")

    # 6. Remove outliers
    center_x, center_y = df['x'].mean(), df['y'].mean()
    var_x, var_y = df['x'].std(), df['y'].std()
    print(center_x, center_y, 'center_x, center_y ')
    print(var_x, var_y, 'var x and y')

    distances = np.sqrt((df['x'] - center_x)**2 + (df['y'] - center_y)**2)
    threshold = np.percentile(distances, 99)
    print('threshold', 'of', threshold)
    outlier_mask = distances <= threshold
    outlier_count = len(df) - outlier_mask.sum()
    df = df[outlier_mask]
    print(f"Removed {outlier_count} outliers (>{threshold:.2f} units from center): {len(df)} tweets remaining")

    # 7. Handle missing values
    df['favorite_count'] = df['favorite_count'].fillna(0).astype(int)
    df['retweet_count'] = df.get('retweet_count', 0).fillna(0).astype(int)
    df['screen_name'] = df['screen_name'].fillna('unknown')
    df['full_text'] = df['full_text'].fillna('')

    # 8. Data quality summary
    print(f"\nData Quality Summary:")
    print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    print(f"Coordinate range: X=[{df['x'].min():.2f}, {df['x'].max():.2f}], Y=[{df['y'].min():.2f}, {df['y'].max():.2f}]")
    print(f"Favorite count range: {df['favorite_count'].min()} to {df['favorite_count'].max()}")
    print(f"Tweets with 0 favorites: {(df['favorite_count'] == 0).sum()} ({(df['favorite_count'] == 0).mean()*100:.1f}%)")

    # 9. Group by day and create time series data
    daily_data = create_daily_frames(df)

    # 10. Create metadata
    dates = list(daily_data.keys())
    metadata = {
        'total_days': len(daily_data),
        'date_range': [min(dates), max(dates)],
        'total_tweets': len(df),
        'total_unique_days': len(df['datetime'].dt.date.unique()),
        'avg_tweets_per_day': len(df) / len(df['datetime'].dt.date.unique()),
        'coordinate_bounds': {
            'x_min': float(df['x'].min()),
            'x_max': float(df['x'].max()),
            'y_min': float(df['y'].min()),
            'y_max': float(df['y'].max())
        }
    }

    # 11. Output JSON for JavaScript
    output = {
        'metadata': metadata,
        'frames': daily_data
    }

    output_path = Path('tweet_timeseries.json')
    print(f"Writing output to {output_path}")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Successfully preprocessed tweet data!")
    print(f"Output file size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")

    return output

if __name__ == "__main__":
    # Run the preprocessing
    result = preprocess_tweet_data()

    # Print some sample data for verification
    print("\nSample frame data:")
    sample_date = list(result['frames'].keys())[0]
    sample_tweets = result['frames'][sample_date][:3]  # First 3 tweets
    for tweet in sample_tweets:
        print(f"  {tweet['screen_name']}: {tweet['full_text'][:50]}... (fav: {tweet['favorite_count']}, size: {tweet['final_size']:.1f})")
