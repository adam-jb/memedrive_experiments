#!/usr/bin/env python3
"""
Script to create community_archive_good_faith_embeddings.csv by combining:
1. Good faith ratings (2D coordinates) from tweet_good_faith_ratings.csv
2. Tweet data from community_archive.parquet

Process described in CLAUDE.md:
- preproc/good_faith/get_good_faith_ratings.py (for 2 dimensions as assessed by GPT)
- preproc/good_faith/create_transformation_matrix.py (to model 2d points for all tweets)
- sideways concat of 2d coords and tweet data from input_data/community_archive.parquet
"""

import pandas as pd
import numpy as np
from pathlib import Path

def create_community_archive_good_faith_embeddings():
    """
    Create community_archive_good_faith_embeddings.csv by combining good faith ratings with tweet data
    """

    # File paths
    good_faith_embeddings_path = Path('~/Desktop/memedrive_experiments/output_data/good_faith_umap_embeddings_2d.npz').expanduser()
    community_archive_path = Path('~/Desktop/memedrive_experiments/input_data/community_archive.parquet').expanduser()
    output_path = Path('~/Desktop/memedrive_experiments/output_data/community_archive_good_faith_embeddings.csv').expanduser()

    print(f"Loading good faith embeddings from: {good_faith_embeddings_path}")
    embeddings_data = np.load(good_faith_embeddings_path)
    good_faith_coords = embeddings_data['embeddings']  # Assuming 2D coordinates are stored as 'embeddings'
    print(f"Good faith embeddings shape: {good_faith_coords.shape}")
    print(f"Expected format: (n_tweets, 2) for 2D coordinates")

    print(f"\nLoading community archive from: {community_archive_path}")
    community_df = pd.read_parquet(community_archive_path)
    print(f"Community archive shape: {community_df.shape}")
    print(f"Community archive columns: {community_df.columns.tolist()}")

    # Check that we have the right number of coordinates for the tweets
    print(f"\nData alignment check:")
    print(f"Community archive tweets: {len(community_df)}")
    print(f"Good faith coordinates: {len(good_faith_coords)}")

    # The good faith coordinates should correspond 1:1 with community archive tweets
    # Assuming the npz file contains coordinates for all tweets in the same order

    if len(good_faith_coords) != len(community_df):
        raise ValueError(f"Mismatch: {len(good_faith_coords)} coordinates vs {len(community_df)} tweets")

    if good_faith_coords.shape[1] != 2:
        raise ValueError(f"Expected 2D coordinates, got shape {good_faith_coords.shape}")

    # Create coordinate dataframe
    coords_df = pd.DataFrame(good_faith_coords, columns=['x', 'y'])

    # Merge community archive with 2D coordinates
    print(f"\nMerging data...")

    # Combine community archive with coordinates (same row order)
    merged_df = pd.concat([community_df.reset_index(drop=True), coords_df.reset_index(drop=True)], axis=1)

    print(f"Merged shape: {merged_df.shape}")
    print(f"All tweets have coordinates: {merged_df[['x', 'y']].notna().all(axis=1).sum()}")
    print(f"Total rows: {len(merged_df)}")

    # Reorder columns to put coordinates near the front
    cols = merged_df.columns.tolist()
    if 'x' in cols and 'y' in cols:
        # Move x, y to the front
        other_cols = [col for col in cols if col not in ['x', 'y']]
        new_order = ['x', 'y'] + other_cols
        merged_df = merged_df[new_order]

    # Drop any unwanted columns if they exist
    cols_to_drop = [col for col in ['level_0'] if col in merged_df.columns]
    if cols_to_drop:
        merged_df = merged_df.drop(cols_to_drop, axis=1)

    print(f"\nFinal columns: {merged_df.columns.tolist()}")
    print(f"Final shape: {merged_df.shape}")

    # Show sample data
    print(f"\nSample of merged data:")
    print(merged_df.head(3))

    # Write to file (commented out as requested)
    print(f"\nWould write to: {output_path}")
    # merged_df.to_csv(output_path, index=False)
    # print(f"File written to: {output_path}")

    return merged_df

def validate_against_real_file(generated_df):
    """
    Compare the generated dataframe with the real file to check if they match
    """
    real_file_path = Path('~/Desktop/memedrive_experiments/output_data/community_archive_good_faith_embeddings.csv').expanduser()

    if not real_file_path.exists():
        print(f"Real file does not exist at: {real_file_path}")
        return False

    print(f"\nValidating against real file: {real_file_path}")

    try:
        real_df = pd.read_csv(real_file_path)
        print(f"Real file shape: {real_df.shape}")
        print(f"Real file columns: {real_df.columns.tolist()}")

        set(generated_df.columns) - set(real_df.columns)
        set(real_df.columns) - set(generated_df.columns)
        print(f"Columns in generated but not in real: {set(generated_df.columns) - set(real_df.columns)}")
        print(f"Columns in real but not in generated: {set(real_df.columns) - set(generated_df.columns)}")


        # Compare shapes
        if generated_df.shape != real_df.shape:
            print(f"❌ Shape mismatch: generated {generated_df.shape} vs real {real_df.shape}")
            return False

        # Compare columns
        if list(generated_df.columns) != list(real_df.columns):
            print(f"❌ Column mismatch:")
            print(f"Generated: {generated_df.columns.tolist()}")
            print(f"Real: {real_df.columns.tolist()}")
            return False

        # Compare a sample of data
        sample_indices = np.random.choice(len(generated_df), min(100, len(generated_df)), replace=False)

        for col in generated_df.columns:
            generated_sample = generated_df.iloc[sample_indices][col]
            real_sample = real_df.iloc[sample_indices][col]

            if generated_sample.dtype != real_sample.dtype:
                print(f"❌ Column {col} dtype mismatch: generated {generated_sample.dtype} vs real {real_sample.dtype}")
                return False

            # For numeric columns, check if values are close
            if pd.api.types.is_numeric_dtype(generated_sample):
                if not np.allclose(generated_sample.fillna(0), real_sample.fillna(0), equal_nan=True, rtol=1e-5):
                    print(f"❌ Column {col} numeric values differ significantly")
                    diff_mask = ~np.isclose(generated_sample.fillna(0), real_sample.fillna(0), equal_nan=True, rtol=1e-5)
                    print(f"First few differences in {col}:")
                    print(f"Generated: {generated_sample[diff_mask].head().tolist()}")
                    print(f"Real: {real_sample[diff_mask].head().tolist()}")
                    return False
            else:
                # For non-numeric columns, check exact matches
                matches = (generated_sample.fillna('') == real_sample.fillna(''))
                if not matches.all():
                    print(f"❌ Column {col} text values differ")
                    print(f"Mismatches: {(~matches).sum()} out of {len(matches)}")
                    return False

        print(f"✅ Files match successfully!")
        return True

    except Exception as e:
        print(f"❌ Error validating: {e}")
        return False

if __name__ == "__main__":
    print("Creating community_archive_good_faith_embeddings.csv...")
    generated_df = create_community_archive_good_faith_embeddings()

    print("\n" + "="*50)
    validate_against_real_file(generated_df)