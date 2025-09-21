#!/usr/bin/env python3
"""
Script to create tweet_timeseries.json from community_archive_good_faith_embeddings.csv

The JSON format includes:
- metadata: stats about the dataset
- frames: tweets grouped by date with calculated fields

Fields in CSV: id, full_text, created_at, retweet_count, favorite_count,
               in_reply_to_status_id, in_reply_to_screen_name, screen_name,
               username, datetime, x, y

Additional fields needed for JSON:
- age_days: calculated from tweet_date and current_date
- base_size: calculated from engagement metrics
- final_size: same as base_size
- opacity: set to 1.0
- tweet_date: extracted from datetime
- current_date: same as tweet_date for each frame
"""

import pandas as pd
import json
import numpy as np
from datetime import datetime
from pathlib import Path


def calculate_tweet_size(favorite_count, retweet_count):
    """Calculate tweet size based on engagement metrics"""
    # Using logarithmic scaling similar to the original
    engagement = favorite_count + retweet_count
    if engagement == 0:
        return 1.0
    return np.log10(engagement + 1) * 1.5 + 1.0


def create_tweet_timeseries_json(csv_path, output_path=None, validate_against=None, start_date=None, end_date=None):
    """
    Convert CSV to tweet_timeseries.json format

    Args:
        csv_path: Path to the community_archive_good_faith_embeddings.csv
        output_path: Optional path to save the JSON (if None, won't save)
        validate_against: Optional path to existing JSON for validation

    Returns:
        dict: The generated JSON structure
    """
    print(f"Reading CSV from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Convert datetime column to pandas datetime
    #df['datetime'] = pd.to_datetime(df['datetime'])
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
    print(df['datetime'].head())


    if start_date:
        start_date = pd.to_datetime(start_date, utc=True)
        print(start_date, 'start_date')
        df = df[df['datetime'] >= start_date]

    if end_date:
        end_date = pd.to_datetime(end_date, utc=True)
        df = df[df['datetime'] <= end_date]

    # Extract date for grouping
    df['date'] = df['datetime'].dt.date.astype(str)

    # Calculate derived fields
    df['base_size'] = df.apply(lambda row: calculate_tweet_size(row['favorite_count'], row['retweet_count']), axis=1)
    df['final_size'] = df['base_size']
    df['opacity'] = 1.0
    df['tweet_date'] = df['date']
    df['current_date'] = df['date']
    df['age_days'] = 0  # For each frame, age_days is 0 (tweets are shown on their own date)

    # Calculate metadata
    unique_dates = sorted(df['date'].unique())
    total_tweets = len(df)

    coordinate_bounds = {
        "x_min": df['x'].min(),
        "x_max": df['x'].max(),
        "y_min": df['y'].min(),
        "y_max": df['y'].max()
    }

    # Create frames
    frames = {}
    for date in unique_dates:
        date_tweets = df[df['date'] == date]

        frame_tweets = []
        for _, tweet in date_tweets.iterrows():
            frame_tweets.append({
                "x": tweet['x'],
                "y": tweet['y'],
                "full_text": tweet['full_text'],
                "favorite_count": int(tweet['favorite_count']),
                "retweet_count": int(tweet['retweet_count']),
                "screen_name": tweet['screen_name'],
                "age_days": int(tweet['age_days']),
                "base_size": float(tweet['base_size']),
                "final_size": float(tweet['final_size']),
                "opacity": float(tweet['opacity']),
                "tweet_date": tweet['tweet_date'],
                "current_date": tweet['current_date']
            })

        frames[date] = frame_tweets

    # Calculate additional metadata
    avg_tweets_per_day = total_tweets / len(unique_dates) if unique_dates else 0

    # Create final structure
    result = {
        "metadata": {
            "total_days": len(unique_dates),
            "date_range": [unique_dates[0], unique_dates[-1]] if unique_dates else [],
            "total_tweets": total_tweets,
            "total_unique_days": len(unique_dates),  # This seems redundant with total_days
            "avg_tweets_per_day": avg_tweets_per_day,
            "coordinate_bounds": coordinate_bounds
        },
        "frames": frames
    }

    print(f"Generated JSON with {total_tweets} tweets across {len(unique_dates)} days")
    print(f"Date range: {unique_dates[0]} to {unique_dates[-1]}")
    print(f"Coordinate bounds: x=[{coordinate_bounds['x_min']:.3f}, {coordinate_bounds['x_max']:.3f}], y=[{coordinate_bounds['y_min']:.3f}, {coordinate_bounds['y_max']:.3f}]")

    # Validation against existing file
    if validate_against:
        print(f"\nValidating against existing file: {validate_against}")
        try:
            with open(validate_against, 'r') as f:
                original = json.load(f)

            # Compare metadata
            orig_meta = original['metadata']
            new_meta = result['metadata']

            missing_fields = []
            different_values = []

            # Check all metadata fields
            for field in orig_meta:
                if field not in new_meta:
                    missing_fields.append(f"metadata.{field}")
                elif orig_meta[field] != new_meta[field]:
                    different_values.append(f"metadata.{field}: original={orig_meta[field]}, new={new_meta[field]}")

            # Check frame structure for a sample date
            sample_dates = list(original['frames'].keys())[:3]  # Check first 3 dates
            for date in sample_dates:
                if date in result['frames']:
                    orig_tweets = original['frames'][date]
                    new_tweets = result['frames'][date]

                    if len(orig_tweets) != len(new_tweets):
                        different_values.append(f"frames.{date}: different tweet count (original={len(orig_tweets)}, new={len(new_tweets)})")

                    # Check first tweet structure
                    if orig_tweets and new_tweets:
                        orig_fields = set(orig_tweets[0].keys())
                        new_fields = set(new_tweets[0].keys())

                        missing_tweet_fields = orig_fields - new_fields
                        extra_tweet_fields = new_fields - orig_fields

                        if missing_tweet_fields:
                            missing_fields.extend([f"tweet.{field}" for field in missing_tweet_fields])
                        if extra_tweet_fields:
                            print(f"Extra fields in new tweets: {extra_tweet_fields}")

            # Report validation results
            if missing_fields:
                print(f"❌ MISSING FIELDS: {missing_fields}")
            if different_values:
                print(f"⚠️  DIFFERENT VALUES: {different_values}")
            if not missing_fields and not different_values:
                print("✅ Validation passed - structure matches!")

        except Exception as e:
            print(f"❌ Validation failed: {e}")

    # Save if output path provided
    if output_path:
        print(f"\nSaving to {output_path}...")
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        print("✅ File saved successfully")

    return result


def test_missing_fields(csv_path, original_json_path):
    """Test which fields are missing from CSV compared to original JSON"""
    print("=== FIELD COMPARISON TEST ===")

    # Load original JSON to see required fields
    with open(original_json_path, 'r') as f:
        original = json.load(f)

    # Get original tweet structure
    sample_frame = list(original['frames'].values())[0]
    if sample_frame:
        original_tweet_fields = set(sample_frame[0].keys())
        print(f"Original tweet fields: {sorted(original_tweet_fields)}")

    # Load CSV to see available fields
    df = pd.read_csv(csv_path, nrows=1)
    csv_fields = set(df.columns)
    print(f"CSV fields: {sorted(csv_fields)}")

    # Fields that need to be calculated/derived
    calculated_fields = {
        'age_days': 'Always 0 for each frame (tweets shown on their date)',
        'base_size': 'Calculated from favorite_count + retweet_count',
        'final_size': 'Same as base_size',
        'opacity': 'Always 1.0',
        'tweet_date': 'Extracted from datetime column',
        'current_date': 'Same as tweet_date for each frame'
    }

    # Direct mappings
    direct_mappings = {
        'x': 'x',
        'y': 'y',
        'full_text': 'full_text',
        'favorite_count': 'favorite_count',
        'retweet_count': 'retweet_count',
        'screen_name': 'screen_name'
    }

    missing_from_csv = original_tweet_fields - csv_fields
    print(f"\nFields missing from CSV: {sorted(missing_from_csv)}")
    print("\nField mapping strategy:")
    for field in sorted(original_tweet_fields):
        if field in direct_mappings:
            print(f"  {field}: direct from CSV column '{direct_mappings[field]}'")
        elif field in calculated_fields:
            print(f"  {field}: {calculated_fields[field]}")
        else:
            print(f"  {field}: ❌ UNKNOWN HOW TO GENERATE")


if __name__ == "__main__":
    csv_path = Path('~/Desktop/memedrive_experiments/output_data/community_archive_good_faith_embeddings.csv').expanduser()

    # Comparison has been done now
    #original_json_path = Path('~/Desktop/memedrive_experiments/preproc/good_faith/tweet_timeseries_ORIGINAL_LEGACY.json').expanduser()

    # Test for missing fields first
    if original_json_path.exists():
        test_missing_fields(csv_path, original_json_path)
        print("\n" + "="*50 + "\n")

    start_date = "2023-01-01"
    end_date = "2023-02-20"

    # Generate the JSON
    result = create_tweet_timeseries_json(
        csv_path=csv_path,
        output_path=None,  # Don't save yet
        validate_against=original_json_path if original_json_path.exists() else None,
        start_date=start_date,
        end_date=end_date
    )

    # Uncomment to save the generated file
    output_path = Path('~/Desktop/memedrive_experiments/preproc/good_faith/tweet_timeseries_TEST.json').expanduser()
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Generated file saved to: {output_path}")