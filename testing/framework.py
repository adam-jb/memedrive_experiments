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

    def __init__(self, grid_bounds: Tuple[Tuple[float, float], Tuple[float, float]] = None,
                 tolerance_radius: int = 3):
        # Default bounds for 2D good-faith space - will be updated based on data
        self.grid_bounds = grid_bounds
        # How forgiving the evaluation should be (grid cells of tolerance)
        self.tolerance_radius = tolerance_radius

    def create_density_grid(self, positions: np.ndarray, grid_size: int = 100,
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

    def create_gaussian_smoothed_density(self, positions: np.ndarray, grid_size: int = 100,
                                       bandwidth: float = 0.05) -> np.ndarray:
        """Create density grid by placing Gaussians around each training tweet position

        This approach puts probabilistic Gaussians around each training data point
        instead of treating them as delta functions
        """
        if len(positions) == 0:
            return np.ones((grid_size, grid_size)) / (grid_size ** 2)

        # Dynamic grid bounds based on actual data with padding
        x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
        y_min, y_max = positions[:, 1].min(), positions[:, 1].max()

        # Add padding
        x_padding = max(0.5, (x_max - x_min) * 0.2)
        y_padding = max(0.5, (y_max - y_min) * 0.2)

        x_bounds = (x_min - x_padding, x_max + x_padding)
        y_bounds = (y_min - y_padding, y_max + y_padding)

        # Create dense grid
        x_grid = np.linspace(x_bounds[0], x_bounds[1], grid_size)
        y_grid = np.linspace(y_bounds[0], y_bounds[1], grid_size)
        xx, yy = np.meshgrid(x_grid, y_grid)

        # Initialize density grid
        density = np.zeros((grid_size, grid_size))

        # Place Gaussian around each training tweet
        for position in positions:
            # Calculate squared distances from this tweet to all grid points
            dist_sq = ((xx - position[0])**2 + (yy - position[1])**2)

            # Add Gaussian contribution (not using KDE, direct calculation)
            gaussian_contrib = np.exp(-dist_sq / (2 * bandwidth**2))
            density += gaussian_contrib

        # Normalize to sum to 1
        density_sum = density.sum()
        if density_sum == 0:
            return np.ones((grid_size, grid_size)) / (grid_size ** 2)

        return density / density_sum

    def create_point_based_density(self, positions: np.ndarray, grid_size: int = 100) -> np.ndarray:
        """Create true density grid from actual tweet points without Gaussians

        Each tweet is represented as a point mass (delta function) on the grid
        """
        if len(positions) == 0:
            return np.ones((grid_size, grid_size)) / (grid_size ** 2)

        # Dynamic grid bounds based on actual data with padding
        x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
        y_min, y_max = positions[:, 1].min(), positions[:, 1].max()

        # Add padding
        x_padding = max(0.5, (x_max - x_min) * 0.2)
        y_padding = max(0.5, (y_max - y_min) * 0.2)

        x_bounds = (x_min - x_padding, x_max + x_padding)
        y_bounds = (y_min - y_padding, y_max + y_padding)

        # Create grid
        x_grid = np.linspace(x_bounds[0], x_bounds[1], grid_size)
        y_grid = np.linspace(y_bounds[0], y_bounds[1], grid_size)

        # Initialize density grid
        density = np.zeros((grid_size, grid_size))

        # Place each tweet as point mass on nearest grid cell
        for position in positions:
            # Find nearest grid indices
            x_idx = np.argmin(np.abs(x_grid - position[0]))
            y_idx = np.argmin(np.abs(y_grid - position[1]))

            # Add point mass (each tweet contributes 1.0)
            density[y_idx, x_idx] += 1.0

        # Normalize to sum to 1
        density_sum = density.sum()
        if density_sum == 0:
            return np.ones((grid_size, grid_size)) / (grid_size ** 2)

        return density / density_sum

    def field_density_score(self, predicted_grid: np.ndarray,
                          true_tweet_positions: np.ndarray, grid_bounds: tuple,
                          tweet_importance_weights: np.ndarray = None) -> float:
        """Calculate field density score based on tweet positions on probability grid

        Args:
            predicted_grid: N×N grid where each cell is relative probability (mean=1)
            true_tweet_positions: Actual tweet positions in coordinate space
            grid_bounds: ((x_min, x_max), (y_min, y_max)) bounds of the grid
            tweet_importance_weights: Optional weights for each tweet (default: all equal)

        Returns:
            Score where:
            - 1.0 = random performance (baseline)
            - 0.0 = perfectly wrong (impossible worst case)
            - >1.0 = better than random (higher is better)
        """
        if len(true_tweet_positions) == 0:
            return 1.0  # No tweets to evaluate

        # Ensure predicted grid has mean = 1 (proper probability field)
        predicted_grid = predicted_grid / predicted_grid.mean()

        # Default to equal importance for all tweets
        if tweet_importance_weights is None:
            tweet_importance_weights = np.ones(len(true_tweet_positions))

        # Normalize importance weights
        tweet_importance_weights = tweet_importance_weights / tweet_importance_weights.sum()

        # Map tweet positions to grid coordinates
        (x_min, x_max), (y_min, y_max) = grid_bounds
        grid_size = predicted_grid.shape[0]

        scores = []
        for tweet_pos in true_tweet_positions:
            # Convert tweet position to grid indices
            x_idx = int((tweet_pos[0] - x_min) / (x_max - x_min) * (grid_size - 1))
            y_idx = int((tweet_pos[1] - y_min) / (y_max - y_min) * (grid_size - 1))

            # Clamp to grid bounds
            x_idx = max(0, min(grid_size - 1, x_idx))
            y_idx = max(0, min(grid_size - 1, y_idx))

            # Get probability at this tweet's location
            tweet_probability = predicted_grid[y_idx, x_idx]
            scores.append(tweet_probability)

        # Calculate weighted average score
        weighted_score = np.average(scores, weights=tweet_importance_weights)

        return weighted_score

    def precision_weighted_brier_score(self, predicted_density: np.ndarray,
                                     true_density: np.ndarray, tolerance_radius: int = 2) -> float:
        """Calculate Brier-like score that rewards precision with spatial forgiveness

        Higher scores are better (opposite of traditional Brier score)
        Rewards models for being confident and correct, with spatial tolerance

        Args:
            tolerance_radius: How many grid cells away predictions can be and still get credit
        """
        # Ensure densities sum to 1
        pred_norm = predicted_density / predicted_density.sum()
        true_norm = true_density / true_density.sum()

        if tolerance_radius == 0:
            # Original harsh scoring - exact match required
            precision_weights = pred_norm + 1e-8
            score = np.sum(precision_weights * (2 * pred_norm * true_norm - pred_norm**2))
        else:
            # Forgiving scoring - spread true density around actual locations
            true_forgiving = self._create_spatially_tolerant_density(true_norm, tolerance_radius)

            # Use forgiving true density for evaluation
            precision_weights = pred_norm + 1e-8
            score = np.sum(precision_weights * (2 * pred_norm * true_forgiving - pred_norm**2))

        return score

    def _create_spatially_tolerant_density(self, true_density: np.ndarray,
                                         tolerance_radius: int) -> np.ndarray:
        """Spread true density within tolerance radius of actual points"""
        from scipy import ndimage

        if tolerance_radius <= 0:
            return true_density

        # Create circular kernel for spreading
        kernel_size = 2 * tolerance_radius + 1
        y, x = np.ogrid[-tolerance_radius:tolerance_radius+1, -tolerance_radius:tolerance_radius+1]
        kernel = (x**2 + y**2) <= tolerance_radius**2
        kernel = kernel.astype(float)
        kernel = kernel / kernel.sum()  # Normalize

        # Apply convolution to spread density around each point
        forgiving_density = ndimage.convolve(true_density, kernel, mode='constant')

        # Re-normalize to sum to 1
        return forgiving_density / forgiving_density.sum()

    def _plot_prediction_heatmap(self, predicted_density: np.ndarray, true_density: np.ndarray,
                                year: int, week: int, model_name: str, field_score: float):
        """Plot heatmap of predictions vs true density"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Plot predicted density
        im1 = ax1.imshow(predicted_density, cmap='viridis', origin='lower')
        ax1.set_title(f'{model_name}\nPredicted Density - Week {year}-{week}\nField Score: {field_score:.4f}')
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
                      grid_size: int = 100) -> Dict[str, float]:
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

            # Create true density from observed positions (point-based, no Gaussians)
            true_density = self.create_point_based_density(week_positions, grid_size)

            # Calculate grid bounds for field density score
            if len(week_positions) > 0:
                x_min, x_max = week_positions[:, 0].min(), week_positions[:, 0].max()
                y_min, y_max = week_positions[:, 1].min(), week_positions[:, 1].max()

                # Add padding (same as used in density creation)
                x_padding = max(0.5, (x_max - x_min) * 0.2)
                y_padding = max(0.5, (y_max - y_min) * 0.2)

                grid_bounds = ((x_min - x_padding, x_max + x_padding),
                              (y_min - y_padding, y_max + y_padding))
            else:
                # Default bounds if no tweets
                grid_bounds = ((0, 8), (0, 6))

            # Calculate new field density score (replaces PWS)
            field_score = self.field_density_score(predicted_density, week_positions, grid_bounds)
            scores.append(field_score)
            tweet_counts.append(len(week_positions))

            # Plot heatmap for HistoricalAverageModel or GaussianSmoothed models
            if "Historical Average" in model.get_name() or "Gaussian Smoothed" in model.get_name():
                self._plot_prediction_heatmap(predicted_density, true_density, year, week, model.get_name(), field_score)

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
            'field_density_score': weighted_pws,  # Now contains field density scores
            'kl_divergence': weighted_kl,
            'score_std': np.std(scores),
            'weeks_evaluated': len(scores),
            'total_test_tweets': total_test_tweets
        }

class TestingFramework:
    """Main framework for testing tweet prediction models"""

    def __init__(self, data_path: str, sample_size: Optional[int] = None,
                 start_date: Optional[str] = None, end_date: Optional[str] = None,
                 tolerance_radius: int = 3):
        self.data_loader = DataLoader(data_path, sample_size, start_date, end_date)
        self.evaluator = ProbabilisticEvaluator(tolerance_radius=tolerance_radius)
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
        print("Field Density Score: 1.0=random, >1.0=better than random, higher=better")

        for model_name, scores in results.items():
            print(f"\n{model_name}:")
            print(f"  Field Density Score: {scores['field_density_score']:.4f}")
            print(f"  KL Divergence: {scores['kl_divergence']:.4f}")
            print(f"  Score Std Dev: {scores['score_std']:.4f}")
            print(f"  Weeks Evaluated: {scores['weeks_evaluated']}")
            print(f"  Total Test Tweets: {scores['total_test_tweets']}")