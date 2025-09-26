import pandas as pd
import numpy as np
import os
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import time
import argparse

# Lens configuration mapping
LENS_CONFIGS = {
    'good_faith': {
        'dimensions': ['sincerity', 'charity', 'constructiveness'],
        'csv_suffix': 'good_faith',
        'output_prefix': 'good_faith'
    },
    'excitement_directedness': {
        'dimensions': ['excitement', 'directedness'],
        'csv_suffix': 'excitement_directedness',
        'output_prefix': 'excitement_directedness'
    }
}

def get_paths(lens_type):
    """Get file paths for specified lens type"""
    config = LENS_CONFIGS[lens_type]
    csv_path = Path(f'~/Desktop/memedrive_experiments/output_data/tweet_{config["csv_suffix"]}_ratings.csv').expanduser()
    embeddings_path = Path('~/Desktop/memedrive_experiments/output_data/community_archive_embeddings.npz').expanduser()
    output_dir = Path('~/Desktop/memedrive_experiments/output_data').expanduser()

    return csv_path, embeddings_path, output_dir, config

def main():
    """Main execution function with argument parsing"""
    parser = argparse.ArgumentParser(description='Create transformation matrix for different lens types')
    parser.add_argument('--lens_type', choices=list(LENS_CONFIGS.keys()),
                       default='good_faith', help='Lens type to use')

    args = parser.parse_args()

    # Get configuration and paths
    csv_path, embeddings_path, output_dir, config = get_paths(args.lens_type)
    dimensions = config['dimensions']
    output_prefix = config['output_prefix']

    print(f"Processing lens type: {args.lens_type}")
    print("Loading data...")
    t1 = time.time()

    # Load ratings CSV
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} tweets with {args.lens_type} ratings")

    print("Columns in CSV:", df.columns.tolist())

    # Load all embeddings
    all_embeddings = np.load(embeddings_path)['embeddings']
    print(f'Embeddings loaded after {time.time() - t1} seconds')

    # Get embeddings for our sample using the 'index' column
    sample_embeddings = all_embeddings[df['index'].values]
    labels = df[dimensions].values

    # Check for and remove NaN values
    print(f"NaN values in labels: {np.isnan(labels).sum()}")
    nan_mask = np.isnan(labels).any(axis=1)
    if nan_mask.sum() > 0:
        print(f"Removing {nan_mask.sum()} rows with NaN values")
        sample_embeddings = sample_embeddings[~nan_mask]
        labels = labels[~nan_mask]
        print(f"Remaining samples: {len(sample_embeddings)}")

    print(f"Correlations between {args.lens_type} dimensions:")
    print(np.corrcoef(labels.T))

    return process_embeddings(sample_embeddings, labels, all_embeddings, output_dir,
                             output_prefix, dimensions, t1)


def process_embeddings(sample_embeddings, labels, all_embeddings, output_dir,
                      output_prefix, dimensions, start_time):
    """Process embeddings and create transformation matrix"""
    print("Running Linear Regression train and test procedure...")
    scaler_embeddings = StandardScaler()
    scaler_labels = StandardScaler()

    X_train, X_test, y_train, y_test = train_test_split(sample_embeddings, labels, test_size=0.2, random_state=42)

    X_train_scaled = scaler_embeddings.fit_transform(X_train)
    y_train_scaled = scaler_labels.fit_transform(y_train)

    # Train linear regression model
    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train_scaled)

    # Test
    X_test_scaled = scaler_embeddings.transform(X_test)
    y_test_scaled = scaler_labels.transform(y_test)
    predictions = lr_model.predict(X_test_scaled)

    print('Testing on 20% test set (scaled targets):')
    for i, feature in enumerate(dimensions):
        corr = np.corrcoef(predictions[:, i], y_test_scaled[:, i])[0, 1]
        print(f"  {feature}: {corr:.4f}")

    print('Testing on 20% test set (original scale):')
    for i, feature in enumerate(dimensions):
        corr = np.corrcoef(predictions[:, i], y_test[:, i])[0, 1]
        print(f"  {feature}: {corr:.4f}")

    print('End of test: now making the main thing')

    print("Running Linear Regression on full dataset...")
    scaler_embeddings_final = StandardScaler()
    embeddings_scaled = scaler_embeddings_final.fit_transform(sample_embeddings)
    labels_scaled = StandardScaler().fit_transform(labels)

    # Train final model on all available data
    lr_final = LinearRegression()
    lr_final.fit(embeddings_scaled, labels_scaled)

    # Save transformation components and scaler
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / f'{output_prefix}_lr_weights.npy', lr_final.coef_.T)
    np.save(output_dir / f'{output_prefix}_lr_intercept.npy', lr_final.intercept_)
    np.save(output_dir / f'{output_prefix}_embedding_scaler_mean.npy', scaler_embeddings_final.mean_)
    np.save(output_dir / f'{output_prefix}_embedding_scaler_scale.npy', scaler_embeddings_final.scale_)

    print(f"\nSaved:")
    print(f"- {output_prefix}_lr_weights.npy")
    print(f"- {output_prefix}_lr_intercept.npy")
    print(f"- {output_prefix}_embedding_scaler_mean.npy")
    print(f"- {output_prefix}_embedding_scaler_scale.npy")

    print(f'Linear regression model trained in {time.time() - start_time} seconds')
    batch_start_time = time.time()

    print('Applying to all 5.5m embeddings in batches!')
    batch_size = 100_000
    num_dimensions = len(dimensions)
    transformed_embeddings_all = np.empty((all_embeddings.shape[0], num_dimensions), dtype=np.float16)

    for i in range(0, len(all_embeddings), batch_size):
        end = min(i + batch_size, len(all_embeddings))
        standardized_batch = (all_embeddings[i:end] - scaler_embeddings_final.mean_) / scaler_embeddings_final.scale_
        transformed_embeddings_all[i:end] = standardized_batch @ lr_final.coef_.T + lr_final.intercept_
        print(f"Processed batch {i}")

    print('Correlations of final output:\n', np.corrcoef(transformed_embeddings_all.T))

    output_path = os.path.expanduser(f'~/Desktop/memedrive_experiments/output_data/{output_prefix}_embeddings_{num_dimensions}d.npz')
    np.savez_compressed(output_path, embeddings=transformed_embeddings_all)
    print(f"{output_prefix}_embeddings_{num_dimensions}d saved to {output_path}")
    print(f'Batch processing completed in {time.time() - batch_start_time} seconds')

    return transformed_embeddings_all


if __name__ == "__main__":
    main()
