from abc import ABC, abstractmethod
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import pandas as pd
from sklearn.neighbors import KernelDensity
from scipy import stats
import datetime
import matplotlib.pyplot as plt
import os

class TweetPredictor(ABC):
    """Abstract base class for tweet prediction models"""

    @abstractmethod
    def fit(self, train_data: np.ndarray, train_times: np.ndarray) -> None:
        """Train the model on historical data

        Args:
            train_data: (N, 2) array of tweet positions in 2D good-faith space
            train_times: (N,) array of timestamps for each tweet
        """
        pass

    @abstractmethod
    def predict_density(self, test_times: np.ndarray, grid_size: int = 50) -> np.ndarray:
        """Predict tweet density for future time periods

        Args:
            test_times: Array of future timestamps to predict for
            grid_size: Resolution of density grid

        Returns:
            (len(test_times), grid_size, grid_size) array of density predictions
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return model name for logging"""
        pass

class DataLoader:
    """Handles loading and preprocessing tweet data"""

    def __init__(self, csv_path: str, sample_size: Optional[int] = None,
                 start_date: Optional[str] = None, end_date: Optional[str] = None):
        self.csv_path = Path(csv_path).expanduser()
        self.sample_size = sample_size
        self.start_date = start_date
        self.end_date = end_date

    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load tweet data with good-faith coordinates and timestamps

        Returns:
            positions: (N, 2) array of good-faith coordinates
            times: (N,) array of timestamps
        """
        df = pd.read_csv(self.csv_path)

        print('DF read with shape:', df.shape)

        if self.sample_size:
            df = df.sample(n=min(self.sample_size, len(df)), random_state=42)

        # Use actual column names from the CSV - handle both formats
        if 'sincerity' in df.columns and 'charity' in df.columns:
            positions = df[['sincerity', 'charity']].values
        elif 'x' in df.columns and 'y' in df.columns:
            positions = df[['x', 'y']].values
        else:
            raise ValueError("CSV must contain either 'sincerity'/'charity' or 'x'/'y' columns")

        times = pd.to_datetime(df['datetime']).values

        # Remove rows with NaN values
        valid_mask = ~(np.isnan(positions).any(axis=1) | pd.isna(times))
        positions = positions[valid_mask]
        times = times[valid_mask]

        # Apply date window filtering if specified
        if self.start_date or self.end_date:
            times_pd = pd.to_datetime(times)
            date_mask = pd.Series(True, index=range(len(times)))

            if self.start_date:
                date_mask = date_mask & (times_pd >= pd.to_datetime(self.start_date))
            if self.end_date:
                date_mask = date_mask & (times_pd <= pd.to_datetime(self.end_date))

            positions = positions[date_mask]
            times = times[date_mask]

            print(f"Date filtering applied: {self.start_date} to {self.end_date}")
            print(f"Tweets after date filtering: {len(positions)}")

        return positions, times

    def temporal_split(self, positions: np.ndarray, times: np.ndarray,
                      test_weeks: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data temporally for training/testing

        Args:
            positions: Tweet positions
            times: Tweet timestamps
            test_weeks: Number of weeks to hold out for testing

        Returns:
            train_positions, train_times, test_positions, test_times
        """
        # Convert to pandas datetime if not already
        times_pd = pd.to_datetime(times)

        # Sort by time
        sort_idx = np.argsort(times_pd)
        positions = positions[sort_idx]
        times = times[sort_idx]
        times_pd = times_pd[sort_idx]

        # Split point: hold out last test_weeks for testing
        split_time = times_pd[-1] - pd.Timedelta(weeks=test_weeks)
        split_mask = times_pd < split_time

        train_positions = positions[split_mask]
        train_times = times[split_mask]
        test_positions = positions[~split_mask]
        test_times = times[~split_mask]

        return train_positions, train_times, test_positions, test_times

class ProbabilisticEvaluator:
    """Evaluates models using Brier-score-like metrics for density prediction"""

    def __init__(self, grid_bounds: Tuple[Tuple[float, float], Tuple[float, float]] = None):
        # Default bounds for 2D good-faith space - will be updated based on data
        self.grid_bounds = grid_bounds

    def create_density_grid(self, positions: np.ndarray, grid_size: int = 50,
                           bandwidth: float = 0.1) -> np.ndarray:
        """Create true density grid from observed tweet positions"""

        if len(positions) == 0:
            return np.ones((grid_size, grid_size)) / (grid_size ** 2)  # Uniform if no data

        # Dynamic grid bounds based on actual data with padding
        x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
        y_min, y_max = positions[:, 1].min(), positions[:, 1].max()

        # Add padding to ensure KDE has room around data points
        x_padding = max(0.5, (x_max - x_min) * 0.2)
        y_padding = max(0.5, (y_max - y_min) * 0.2)

        x_bounds = (x_min - x_padding, x_max + x_padding)
        y_bounds = (y_min - y_padding, y_max + y_padding)


        x_grid = np.linspace(x_bounds[0], x_bounds[1], grid_size)
        y_grid = np.linspace(y_bounds[0], y_bounds[1], grid_size)
        xx, yy = np.meshgrid(x_grid, y_grid)
        grid_points = np.column_stack([xx.ravel(), yy.ravel()])

        kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
        kde.fit(positions)
        log_density = kde.score_samples(grid_points)
        density = np.exp(log_density).reshape(grid_size, grid_size)

        # Normalize to sum to 1 - check for zero sum
        density_sum = density.sum()
        if density_sum == 0:
            return np.ones((grid_size, grid_size)) / (grid_size ** 2)

        return density / density_sum

    def precision_weighted_brier_score(self, predicted_density: np.ndarray,
                                     true_density: np.ndarray) -> float:
        """Calculate Brier-like score that rewards precision

        Higher scores are better (opposite of traditional Brier score)
        Rewards models for being confident and correct
        """
        # Ensure densities sum to 1
        pred_norm = predicted_density / predicted_density.sum()
        true_norm = true_density / true_density.sum()

        # Precision reward: higher weight where prediction is concentrated
        precision_weights = pred_norm + 1e-8  # Avoid division by zero

        # Modified Brier score: reward correct high-confidence predictions
        score = np.sum(precision_weights * (2 * pred_norm * true_norm - pred_norm**2))

        return score

    def _plot_prediction_heatmap(self, predicted_density: np.ndarray, true_density: np.ndarray,
                                year: int, week: int, model_name: str, pws_score: float):
        """Plot heatmap of predictions vs true density"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Plot predicted density
        im1 = ax1.imshow(predicted_density, cmap='viridis', origin='lower')
        ax1.set_title(f'{model_name}\nPredicted Density - Week {year}-{week}\nPWS: {pws_score:.4f}')
        ax1.set_xlabel('Charity →')
        ax1.set_ylabel('Sincerity →')
        plt.colorbar(im1, ax=ax1, label='Probability Density')

        # Plot true density
        im2 = ax2.imshow(true_density, cmap='viridis', origin='lower')
        ax2.set_title(f'True Density - Week {year}-{week}')
        ax2.set_xlabel('Charity →')
        ax2.set_ylabel('Sincerity →')
        plt.colorbar(im2, ax=ax2, label='Probability Density')

        plt.tight_layout()

        # Save to image_outputs
        filename = f"prediction_heatmap_{year}_week{week}_{model_name.replace(' ', '_').replace('(', '').replace(')', '').replace('=', '')}.png"
        filepath = os.path.join('image_outputs', filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  Saved heatmap: {filepath}")

    def evaluate_model(self, model: TweetPredictor,
                      train_positions: np.ndarray, train_times: np.ndarray,
                      test_positions: np.ndarray, test_times: np.ndarray,
                      grid_size: int = 10) -> Dict[str, float]:
        """Comprehensive model evaluation with rolling window"""

        # Combine all data and sort by time
        all_positions = np.vstack([train_positions, test_positions])
        all_times = np.concatenate([train_times, test_times])

        # Sort everything by time
        sort_idx = np.argsort(all_times)
        all_positions = all_positions[sort_idx]
        all_times = all_times[sort_idx]

        # Find the split point (where test data starts)
        original_split_time = train_times.max()

        # Group by week for evaluation
        all_df = pd.DataFrame({'time': all_times, 'pos': list(all_positions)})
        all_df['datetime'] = pd.to_datetime(all_df['time'])
        all_df['week'] = all_df['datetime'].dt.isocalendar().week
        all_df['year'] = all_df['datetime'].dt.year

        # Only evaluate weeks that are in the test period
        test_weeks = all_df[all_df['datetime'] > original_split_time]

        scores = []
        kl_divergences = []
        tweet_counts = []
        total_test_tweets = 0

        for (year, week) in test_weeks[['year', 'week']].drop_duplicates().values:
            # Get data for this specific week
            week_mask = (all_df['year'] == year) & (all_df['week'] == week)
            week_data = all_df[week_mask]
            week_positions = np.array(week_data['pos'].tolist())
            week_times = week_data['time'].values

            # Only proceed if this week is in test period
            if not any(pd.to_datetime(week_times) > original_split_time):
                continue

            total_test_tweets += len(week_positions)

            # Train model on all data up to (but not including) this week
            train_mask = all_df['datetime'] < week_data['datetime'].min()
            if train_mask.sum() == 0:
                continue  # Skip if no training data

            week_train_positions = np.array(all_df[train_mask]['pos'].tolist())
            week_train_times = all_df[train_mask]['time'].values

            # Train model on expanding window
            model.fit(week_train_positions, week_train_times)

            # Predict density for this week
            predicted_density = model.predict_density(week_times, grid_size)[0]

            # Create true density from observed positions
            true_density = self.create_density_grid(week_positions, grid_size)

            # Output prediction overview for this week
            # print(f"Week {year}-{week}: {len(week_positions)} tweets")
            # print(f"  Predicted density shape: {predicted_density.shape}")
            # print(f"  Predicted density range: [{predicted_density.min():.6f}, {predicted_density.max():.6f}]")
            # print(f"  Predicted density sum: {predicted_density.sum():.6f}")
            # print(f"  True density range: [{true_density.min():.6f}, {true_density.max():.6f}]")
            # print(f"  True density sum: {true_density.sum():.6f}")

            # Show where density is concentrated
            pred_top_indices = np.unravel_index(np.argpartition(predicted_density.flatten(), -5)[-5:], predicted_density.shape)
            true_top_indices = np.unravel_index(np.argpartition(true_density.flatten(), -5)[-5:], true_density.shape)
            # print(f"  Predicted high-density locations (top 5): {list(zip(pred_top_indices[0], pred_top_indices[1]))}")
            # print(f"  True high-density locations (top 5): {list(zip(true_top_indices[0], true_top_indices[1]))}")

            # Calculate scores
            pws = self.precision_weighted_brier_score(predicted_density, true_density)
            scores.append(pws)
            tweet_counts.append(len(week_positions))

            # Debug scoring - analyze why scores might be low
            pred_norm = predicted_density / predicted_density.sum()
            true_norm = true_density / true_density.sum()

            # Check overlap between predicted and true high-density areas
            pred_top_10_percent = pred_norm >= np.percentile(pred_norm, 90)
            true_top_10_percent = true_norm >= np.percentile(true_norm, 90)
            overlap = np.sum(pred_top_10_percent & true_top_10_percent) / np.sum(true_top_10_percent)

            # Check if predictions are too uniform (not confident enough)
            pred_entropy = -np.sum(pred_norm * np.log(pred_norm + 1e-10))
            true_entropy = -np.sum(true_norm * np.log(true_norm + 1e-10))
            max_entropy = np.log(pred_norm.size)  # Uniform distribution entropy

            print(f"Week {year}-{week} Score Analysis:")
            print(f"  PWS: {pws:.6f}")
            print(f"  Top 10% overlap: {overlap:.3f} (1.0 = perfect)")
            print(f"  Pred entropy: {pred_entropy:.3f} / {max_entropy:.3f} (higher = more uniform)")
            print(f"  True entropy: {true_entropy:.3f} / {max_entropy:.3f}")
            print(f"  Pred max density: {pred_norm.max():.6f}")
            print(f"  True max density: {true_norm.max():.6f}")
            print()

            # Plot heatmap for HistoricalAverageModel
            if "Historical Average" in model.get_name():
                self._plot_prediction_heatmap(predicted_density, true_density, year, week, model.get_name(), pws)

            # KL divergence (traditional metric)
            pred_norm = predicted_density / predicted_density.sum() + 1e-10
            true_norm = true_density / true_density.sum() + 1e-10
            kl = stats.entropy(true_norm.flatten(), pred_norm.flatten())
            kl_divergences.append(kl)

        # Calculate weighted means by tweet count
        scores = np.array(scores)
        kl_divergences = np.array(kl_divergences)
        tweet_counts = np.array(tweet_counts)

        # Weighted mean: sum(score * weight) / sum(weight)
        weighted_pws = np.sum(scores * tweet_counts) / np.sum(tweet_counts) if np.sum(tweet_counts) > 0 else np.mean(scores)
        weighted_kl = np.sum(kl_divergences * tweet_counts) / np.sum(tweet_counts) if np.sum(tweet_counts) > 0 else np.mean(kl_divergences)

        return {
            'precision_weighted_score': weighted_pws,
            'kl_divergence': weighted_kl,
            'score_std': np.std(scores),
            'weeks_evaluated': len(scores),
            'total_test_tweets': total_test_tweets
        }

class TestingFramework:
    """Main framework for testing tweet prediction models"""

    def __init__(self, data_path: str, sample_size: Optional[int] = None,
                 start_date: Optional[str] = None, end_date: Optional[str] = None):
        self.data_loader = DataLoader(data_path, sample_size, start_date, end_date)
        self.evaluator = ProbabilisticEvaluator()
        self.models = []

    def add_model(self, model: TweetPredictor):
        """Add a model to be tested"""
        self.models.append(model)

    def run_evaluation(self, test_weeks: int = 1) -> Dict[str, Dict[str, float]]:
        """Run full evaluation pipeline"""
        print("Loading data...")
        positions, times = self.data_loader.load_data()

        print(f"Loaded {len(positions)} tweets")
        print("Creating temporal split...")
        train_pos, train_times, test_pos, test_times = \
            self.data_loader.temporal_split(positions, times, test_weeks)

        print(f"Training data: {len(train_pos)} tweets")
        print(f"Test data: {len(test_pos)} tweets")

        results = {}

        for model in self.models:
            print(f"\nEvaluating {model.get_name()}...")
            scores = self.evaluator.evaluate_model(
                model, train_pos, train_times, test_pos, test_times
            )
            results[model.get_name()] = scores

        return results

    def print_results(self, results: Dict[str, Dict[str, float]]):
        """Print formatted results"""
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)

        for model_name, scores in results.items():
            print(f"\n{model_name}:")
            print(f"  Precision-Weighted Score: {scores['precision_weighted_score']:.4f}")
            print(f"  KL Divergence: {scores['kl_divergence']:.4f}")
            print(f"  Score Std Dev: {scores['score_std']:.4f}")
            print(f"  Weeks Evaluated: {scores['weeks_evaluated']}")
            print(f"  Total Test Tweets: {scores['total_test_tweets']}")