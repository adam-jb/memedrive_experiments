"""
Ultra-Simplified Basin Analysis

Only optimizes the most essential parameters with clear ranges defined at top.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import hdbscan
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

#==============================================================================
# OPTIMIZATION PARAMETERS - EDIT THESE TO TUNE COMPLEXITY
#==============================================================================

# Parameters to optimize (set to None to skip optimization)
OPTIMIZE_PARAMS = {
    'bin_size_days': [1, 2, 3, 5, 7],           # Temporal window sizes to test
    'min_cluster_size': None,                    # Set to None = auto-calculate from data size
}

# Fixed parameters (not optimized)
FIXED_PARAMS = {
    'train_split': 0.6,                         # Use 60% for parameter learning
    'engagement_weight': True,                   # Weight by retweets + likes
    'min_track_length': 2,                      # Minimum bins for a valid basin
    'max_distance_threshold': 1.5,             # Max distance for cluster tracking
}

#==============================================================================

# File paths
INPUT_FILE = "~/Desktop/memedrive_experiments/output_data/basin_finder/dummy_tweet_embeddings.csv"
OUTPUT_DIR = "~/Desktop/memedrive_experiments/output_data/basin_finder/"

class UltraSimpleBasinAnalyzer:
    """Minimalist basin analyzer with just 1-2 optimized parameters."""

    def __init__(self):
        self.output_dir = Path(OUTPUT_DIR).expanduser()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Data
        self.df = None
        self.embedding_cols = None
        self.train_df = None
        self.test_df = None

        # Learned parameters
        self.best_bin_size = 3  # Default
        self.best_min_cluster_size = 10  # Default

        # Results
        self.train_basins = []
        self.test_basins = []

    def load_data(self):
        """Load and split data temporally."""
        print("Loading data...")

        self.df = pd.read_csv(Path(INPUT_FILE).expanduser())
        self.df['datetime'] = pd.to_datetime(self.df['datetime'])
        self.df = self.df.sort_values('datetime')

        # Find embedding columns
        self.embedding_cols = [col for col in self.df.columns if col.startswith('e')]

        # Temporal split
        split_idx = int(len(self.df) * FIXED_PARAMS['train_split'])
        self.train_df = self.df.iloc[:split_idx]
        self.test_df = self.df.iloc[split_idx:]

        print(f"Data: {len(self.df)} tweets, {len(self.embedding_cols)} dimensions")
        print(f"Train: {len(self.train_df)}, Test: {len(self.test_df)}")

    def optimize_parameters(self):
        """Optimize only the parameters specified in OPTIMIZE_PARAMS."""
        print("\nOptimizing parameters on training data...")

        # Auto-set min_cluster_size if not optimizing
        if OPTIMIZE_PARAMS['min_cluster_size'] is None:
            self.best_min_cluster_size = max(5, int(np.sqrt(len(self.train_df)) / 20))
            print(f"Auto min_cluster_size: {self.best_min_cluster_size}")

        # Optimize bin_size if specified
        if OPTIMIZE_PARAMS['bin_size_days'] is not None:
            self.best_bin_size = self._optimize_bin_size()
            print(f"Best bin_size: {self.best_bin_size} days")

        # Optimize min_cluster_size if specified
        if OPTIMIZE_PARAMS['min_cluster_size'] is not None:
            self.best_min_cluster_size = self._optimize_min_cluster_size()
            print(f"Best min_cluster_size: {self.best_min_cluster_size}")

    def _optimize_bin_size(self):
        """Find best bin size by testing clustering quality."""
        best_size = 3
        best_score = -1

        for bin_size in OPTIMIZE_PARAMS['bin_size_days']:
            score = self._evaluate_bin_size(bin_size)
            print(f"  Bin size {bin_size}: score = {score:.3f}")

            if score > best_score:
                best_score = score
                best_size = bin_size

        return best_size

    def _evaluate_bin_size(self, bin_size):
        """Evaluate a bin size by average clustering quality."""
        bins = self._create_bins(self.train_df, bin_size)

        if len(bins) < 2:
            return 0

        scores = []
        for bin_data in bins:
            if len(bin_data) < self.best_min_cluster_size:
                continue

            # Quick clustering
            embeddings = bin_data[self.embedding_cols].values
            scaler = StandardScaler()
            embeddings_scaled = scaler.fit_transform(embeddings)

            # Add engagement weights if enabled
            if FIXED_PARAMS['engagement_weight']:
                weights = bin_data['retweet_count'] + bin_data['favourite_count'] + 1
            else:
                weights = None

            try:
                clusterer = hdbscan.HDBSCAN(min_cluster_size=self.best_min_cluster_size)
                labels = clusterer.fit_predict(embeddings_scaled)

                if len(np.unique(labels)) > 1:
                    score = silhouette_score(embeddings_scaled, labels)
                    scores.append(score)
            except:
                continue

        return np.mean(scores) if scores else 0

    def _optimize_min_cluster_size(self):
        """Find best minimum cluster size."""
        best_size = 10
        best_score = -1

        for min_size in OPTIMIZE_PARAMS['min_cluster_size']:
            # Test on sample of training data
            sample = self.train_df.sample(n=min(500, len(self.train_df)), random_state=42)
            score = self._evaluate_min_cluster_size(sample, min_size)
            print(f"  Min cluster size {min_size}: score = {score:.3f}")

            if score > best_score:
                best_score = score
                best_size = min_size

        return best_size

    def _evaluate_min_cluster_size(self, data, min_size):
        """Evaluate clustering quality for given min_cluster_size."""
        embeddings = data[self.embedding_cols].values
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings)

        if FIXED_PARAMS['engagement_weight']:
            weights = data['retweet_count'] + data['favourite_count'] + 1
        else:
            weights = None

        try:
            clusterer = hdbscan.HDBSCAN(min_cluster_size=min_size)
            labels = clusterer.fit_predict(embeddings_scaled)

            if len(np.unique(labels)) > 1:
                return silhouette_score(embeddings_scaled, labels)
        except:
            pass

        return -1

    def _create_bins(self, data, bin_size_days):
        """Create temporal bins."""
        start_date = data['datetime'].min()
        end_date = data['datetime'].max()

        bins = []
        current_date = start_date

        while current_date < end_date:
            next_date = current_date + timedelta(days=bin_size_days)
            mask = (data['datetime'] >= current_date) & (data['datetime'] < next_date)
            bin_data = data[mask]

            if len(bin_data) >= self.best_min_cluster_size:
                bins.append(bin_data)

            current_date = next_date

        return bins

    def find_basins(self, data, label=""):
        """Find basins in data using learned parameters."""
        print(f"\nFinding basins in {label} data...")

        # Create temporal bins
        bins = self._create_bins(data, self.best_bin_size)
        print(f"Created {len(bins)} bins")

        # Cluster each bin
        all_clusters = []
        for i, bin_data in enumerate(bins):
            clusters = self._cluster_bin(bin_data, i)
            all_clusters.extend(clusters)

        # Track clusters across time to form basins
        basins = self._track_clusters(all_clusters)

        print(f"Found {len(basins)} basins")
        return basins

    def _cluster_bin(self, data, bin_id):
        """Cluster a single temporal bin."""
        embeddings = data[self.embedding_cols].values
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings)

        # Engagement weights
        if FIXED_PARAMS['engagement_weight']:
            weights = data['retweet_count'] + data['favourite_count'] + 1
        else:
            weights = None

        # Cluster
        clusterer = hdbscan.HDBSCAN(min_cluster_size=self.best_min_cluster_size)
        labels = clusterer.fit_predict(embeddings_scaled)

        # Extract clusters
        clusters = []
        for label in np.unique(labels):
            if label == -1:  # Skip noise
                continue

            mask = labels == label
            if np.sum(mask) >= self.best_min_cluster_size:
                center = np.mean(embeddings[mask], axis=0)
                size = np.sum(mask)

                clusters.append({
                    'bin_id': bin_id,
                    'cluster_id': len(clusters),
                    'center': center,
                    'size': size
                })

        return clusters

    def _track_clusters(self, all_clusters):
        """Track clusters across time bins to form basin tracks."""
        # Group by bin_id
        bins_dict = {}
        for cluster in all_clusters:
            bin_id = cluster['bin_id']
            if bin_id not in bins_dict:
                bins_dict[bin_id] = []
            bins_dict[bin_id].append(cluster)

        # Track across consecutive bins
        basins = []
        next_basin_id = 0

        sorted_bins = sorted(bins_dict.keys())

        for i, bin_id in enumerate(sorted_bins):
            current_clusters = bins_dict[bin_id]

            if i == 0:
                # Start new basins for first bin
                for cluster in current_clusters:
                    basins.append({
                        'basin_id': next_basin_id,
                        'timeline': [cluster],
                        'active': True
                    })
                    next_basin_id += 1
            else:
                # Match to existing active basins
                active_basins = [b for b in basins if b['active']]

                if active_basins and current_clusters:
                    # Distance-based matching
                    prev_centers = [b['timeline'][-1]['center'] for b in active_basins]
                    curr_centers = [c['center'] for c in current_clusters]

                    distances = cdist(prev_centers, curr_centers)
                    row_ind, col_ind = linear_sum_assignment(distances)

                    matched_basins = set()
                    matched_clusters = set()

                    for r, c in zip(row_ind, col_ind):
                        if distances[r, c] <= FIXED_PARAMS['max_distance_threshold']:
                            active_basins[r]['timeline'].append(current_clusters[c])
                            matched_basins.add(r)
                            matched_clusters.add(c)

                    # End unmatched basins
                    for b_idx in range(len(active_basins)):
                        if b_idx not in matched_basins:
                            active_basins[b_idx]['active'] = False

                    # Start new basins for unmatched clusters
                    for c_idx, cluster in enumerate(current_clusters):
                        if c_idx not in matched_clusters:
                            basins.append({
                                'basin_id': next_basin_id,
                                'timeline': [cluster],
                                'active': True
                            })
                            next_basin_id += 1

        # End remaining active basins
        for basin in basins:
            basin['active'] = False

        # Filter by minimum length
        valid_basins = []
        for basin in basins:
            if len(basin['timeline']) >= FIXED_PARAMS['min_track_length']:
                # Add summary stats
                sizes = [c['size'] for c in basin['timeline']]
                basin['duration'] = len(basin['timeline'])
                basin['avg_size'] = np.mean(sizes)
                basin['total_size'] = np.sum(sizes)
                basin['size_stability'] = 1 - (np.std(sizes) / (np.mean(sizes) + 1e-10))

                valid_basins.append(basin)

        return valid_basins

    def validate_results(self):
        """Compare train vs test results."""
        train_count = len(self.train_basins)
        test_count = len(self.test_basins)

        if train_count == 0 or test_count == 0:
            print("No basins found in one of the datasets")
            return {'validation': 'failed', 'reason': 'no_basins'}

        # Compare average properties
        train_avg_duration = np.mean([b['duration'] for b in self.train_basins])
        test_avg_duration = np.mean([b['duration'] for b in self.test_basins])

        train_avg_size = np.mean([b['avg_size'] for b in self.train_basins])
        test_avg_size = np.mean([b['avg_size'] for b in self.test_basins])

        # Simple similarity score
        duration_sim = min(train_avg_duration, test_avg_duration) / max(train_avg_duration, test_avg_duration)
        size_sim = min(train_avg_size, test_avg_size) / max(train_avg_size, test_avg_size)
        similarity = (duration_sim + size_sim) / 2

        validation = {
            'similarity_score': similarity,
            'train_basins': train_count,
            'test_basins': test_count,
            'overfitting_risk': 'Low' if similarity > 0.7 else 'High' if similarity < 0.4 else 'Medium'
        }

        print(f"Validation: {validation['overfitting_risk']} overfitting risk (similarity: {similarity:.3f})")
        return validation

    def save_results(self):
        """Save simple results."""
        all_basins = []

        for basin in self.train_basins:
            all_basins.append({
                'dataset': 'train',
                'basin_id': basin['basin_id'],
                'duration': basin['duration'],
                'avg_size': basin['avg_size'],
                'size_stability': basin['size_stability']
            })

        for basin in self.test_basins:
            all_basins.append({
                'dataset': 'test',
                'basin_id': basin['basin_id'],
                'duration': basin['duration'],
                'avg_size': basin['avg_size'],
                'size_stability': basin['size_stability']
            })

        if all_basins:
            df = pd.DataFrame(all_basins)
            df.to_csv(self.output_dir / "simple_basin_results.csv", index=False)
            print(f"Results saved to: {self.output_dir / 'simple_basin_results.csv'}")

    def run_analysis(self):
        """Run complete simplified analysis."""
        print("ULTRA-SIMPLE BASIN ANALYSIS")
        print("=" * 40)
        print(f"Optimizing parameters: {[k for k, v in OPTIMIZE_PARAMS.items() if v is not None]}")
        print()

        # Load data
        self.load_data()

        # Optimize parameters on train data
        self.optimize_parameters()

        # Find basins
        self.train_basins = self.find_basins(self.train_df, "training")
        self.test_basins = self.find_basins(self.test_df, "test")

        # Validate
        validation = self.validate_results()

        # Save
        self.save_results()

        return {
            'parameters': {
                'bin_size': self.best_bin_size,
                'min_cluster_size': self.best_min_cluster_size
            },
            'validation': validation,
            'train_basins': len(self.train_basins),
            'test_basins': len(self.test_basins)
        }

# Main execution
if __name__ == "__main__":
    analyzer = UltraSimpleBasinAnalyzer()
    results = analyzer.run_analysis()

    print(f"\nFINAL RESULTS:")
    print(f"Parameters: {results['parameters']}")
    print(f"Basins found: {results['train_basins']} (train), {results['test_basins']} (test)")
    print(f"Validation: {results['validation']['overfitting_risk']} risk")
