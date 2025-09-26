#!/usr/bin/env python3
"""
Script to create community_archive_{lens}_embeddings.csv by combining:
1. Transformed embeddings (2D coordinates) from {lens}_embeddings_2d.npz
2. Tweet data from community_archive.parquet

Parameterized version that works for any lens type (good_faith, excitement_directedness, etc.)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse

def create_community_archive_embeddings(lens_type='good_faith'):
    """
    Create community_archive_{lens}_embeddings.csv by combining transformed embeddings with tweet data
    """

    # Try direct 2D output from transformation matrix first, then UMAP version
    direct_2d_path = Path(f'~/Desktop/memedrive_experiments/output_data/{lens_type}_embeddings_2d.npz').expanduser()
    umap_2d_path = Path(f'~/Desktop/memedrive_experiments/output_data/{lens_type}_umap_embeddings_2d.npz').expanduser()

    # For good_faith (3D→2D), use UMAP version; for others (already 2D), use direct
    if lens_type == 'good_faith' and umap_2d_path.exists():
        embeddings_path = umap_2d_path
        data_key = 'good_faith_umap_embeddings'  # Legacy key name
    elif direct_2d_path.exists():
        embeddings_path = direct_2d_path
        data_key = 'embeddings'
    else:
        raise FileNotFoundError(f"No 2D embeddings found for {lens_type}. Tried: {direct_2d_path}, {umap_2d_path}")

    community_archive_path = Path('~/Desktop/memedrive_experiments/input_data/community_archive.parquet').expanduser()
    output_path = Path(f'~/Desktop/memedrive_experiments/output_data/community_archive_{lens_type}_embeddings.csv').expanduser()

    print(f"Processing lens type: {lens_type}")
    print(f"Loading {lens_type} embeddings from: {embeddings_path}")

    embeddings_data = np.load(embeddings_path)

    # Try the specific key first, fall back to 'embeddings'
    if data_key in embeddings_data:
        coords = embeddings_data[data_key]
    else:
        coords = embeddings_data['embeddings']
    print(f"{lens_type} embeddings shape: {coords.shape}")
    print(f"Expected format: (n_tweets, 2) for 2D coordinates")

    print(f"\nLoading community archive from: {community_archive_path}")
    community_df = pd.read_parquet(community_archive_path)
    print(f"Community archive shape: {community_df.shape}")
    print(f"Community archive columns: {community_df.columns.tolist()}")

    # Check that we have the right number of coordinates for the tweets
    print(f"\nData alignment check:")
    print(f"Community archive tweets: {len(community_df)}")
    print(f"{lens_type} coordinates: {len(coords)}")

    if len(coords) != len(community_df):
        raise ValueError(f"Mismatch: {len(coords)} coordinates vs {len(community_df)} tweets")

    if coords.shape[1] != 2:
        raise ValueError(f"Expected 2D coordinates, got shape {coords.shape}")

    # Create coordinate dataframe
    coords_df = pd.DataFrame(coords, columns=['x', 'y'])

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

    # Write to file
    print(f"\nWriting to: {output_path}")
    merged_df.to_csv(output_path, index=False)
    print(f"File written to: {output_path}")

    return merged_df


def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description='Create community archive embeddings for different lens types')
    parser.add_argument('--lens_type', type=str, default='good_faith',
                       help='Lens type to use (e.g., good_faith, excitement_directedness)')

    args = parser.parse_args()

    print(f"Creating community_archive_{args.lens_type}_embeddings.csv...")
    generated_df = create_community_archive_embeddings(args.lens_type)
    print(f"\n✅ Successfully created community_archive_{args.lens_type}_embeddings.csv")


if __name__ == "__main__":
    main()