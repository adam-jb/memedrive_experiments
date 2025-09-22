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
import json
import time

def calculate_tweet_importance(retweet_count, favorite_count, retweet_weight=1.0, favorite_weight=1.0, min_weight=1.0):
    """Calculate tweet importance based on engagement metrics

    Args:
        retweet_count: Number of retweets
        favorite_count: Number of favorites/likes
        retweet_weight: Weight multiplier for retweets (default 1.0)
        favorite_weight: Weight multiplier for favorites (default 1.0)
        min_weight: Minimum weight for any tweet (default 1.0)

    Returns:
        Linear combination of engagement metrics with minimum floor
    """
    weight = (retweet_count * retweet_weight) + (favorite_count * favorite_weight) + min_weight
    return float(weight)


class TweetPredictor(ABC):
    """Abstract base class for tweet prediction models"""

    @abstractmethod
    def fit(self, train_data: np.ndarray, train_times: np.ndarray, train_weights: np.ndarray = None, grid_size: int = 50) -> None:
        """Train the model on historical data

        Args:
            train_data: (N, 2) array of tweet positions in 2D good-faith space
            train_times: (N,) array of timestamps for each tweet
            train_weights: (N,) array of importance weights for each tweet (optional)
            grid_size: Resolution of density grid
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

    def calculate_fds_score(self, predicted_density: np.ndarray, actual_tweets: np.ndarray,
                           grid_bounds: Tuple[Tuple[float, float], Tuple[float, float]],
                           tweet_weights: np.ndarray = None) -> float:
        """Calculate Field Density Score for prediction

        Args:
            predicted_density: (grid_size, grid_size) array of predicted probabilities
            actual_tweets: (N, 2) array of actual tweet positions
            grid_bounds: ((x_min, x_max), (y_min, y_max)) spatial bounds of the grid
            tweet_weights: (N,) array of importance weights for each tweet (optional)

        Returns:
            FDS score where 1.0 is random performance, higher is better
        """
        if len(actual_tweets) == 0:
            return 1.0  # Neutral score

        total_score = 0
        total_weight = 0

        # Use uniform weights if none provided
        if tweet_weights is None:
            tweet_weights = np.ones(len(actual_tweets))

        # Unpack grid bounds
        (x_min, x_max), (y_min, y_max) = grid_bounds

        # Map tweet positions to grid coordinates
        current_grid_size = predicted_density.shape[0]
        x_coords = (actual_tweets[:, 0] - x_min) / (x_max - x_min) * (current_grid_size - 1)
        y_coords = (actual_tweets[:, 1] - y_min) / (y_max - y_min) * (current_grid_size - 1)

        # Clamp to grid bounds
        x_coords = np.clip(x_coords, 0, current_grid_size - 1).astype(int)
        y_coords = np.clip(y_coords, 0, current_grid_size - 1).astype(int)

        # Calculate expected random probability (uniform distribution)
        expected_random_prob = 1.0 / (current_grid_size ** 2)

        # Calculate FDS
        for i in range(len(actual_tweets)):
            grid_x, grid_y = x_coords[i], y_coords[i]
            predicted_prob = predicted_density[grid_y, grid_x]  # Correct: [row, col] = [y, x] indexing

            # FDS = actual_probability / expected_random_probability
            # Score of 1.0 = random performance, >1.0 = better than random
            fds_contribution = predicted_prob / expected_random_prob

            # Use tweet importance weight
            weight = tweet_weights[i]

            total_score += fds_contribution * weight
            total_weight += weight

        return total_score / total_weight if total_weight > 0 else 1.0

class DataLoader:
    """Handles loading and preprocessing tweet data"""

    def __init__(self, csv_path: str, sample_size: Optional[int] = None,
                 start_date: Optional[str] = None, end_date: Optional[str] = None):
        self.csv_path = Path(csv_path).expanduser()
        self.sample_size = sample_size
        self.start_date = start_date
        self.end_date = end_date

    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load tweet data with good-faith coordinates, timestamps, and engagement weights

        Returns:
            positions: (N, 2) array of good-faith coordinates
            times: (N,) array of timestamps
            weights: (N,) array of importance weights based on engagement
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

        # Calculate importance weights from engagement metrics
        if 'retweet_count' in df.columns and 'favorite_count' in df.columns:
            weights = np.array([
                calculate_tweet_importance(row['retweet_count'], row['favorite_count'])
                for _, row in df.iterrows()
            ])
        else:
            print("Warning: No engagement columns found, using uniform weights")
            weights = np.ones(len(df))

        # Remove rows with NaN values
        valid_mask = ~(np.isnan(positions).any(axis=1) | pd.isna(times))
        positions = positions[valid_mask]
        times = times[valid_mask]
        weights = weights[valid_mask]

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
            weights = weights[date_mask]

            print(f"Date filtering applied: {self.start_date} to {self.end_date}")
            print(f"Tweets after date filtering: {len(positions)}")

        return positions, times, weights

    def temporal_split(self, positions: np.ndarray, times: np.ndarray, weights: np.ndarray,
                      test_weeks: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data temporally for training/testing

        Args:
            positions: Tweet positions
            times: Tweet timestamps
            weights: Tweet importance weights
            test_weeks: Number of weeks to hold out for testing

        Returns:
            train_positions, train_times, train_weights, test_positions, test_times, test_weights
        """
        # Convert to pandas datetime if not already
        times_pd = pd.to_datetime(times)

        # Sort by time
        sort_idx = np.argsort(times_pd)
        positions = positions[sort_idx]
        times = times[sort_idx]
        weights = weights[sort_idx]
        times_pd = times_pd[sort_idx]

        # Split point: hold out last test_weeks for testing
        split_time = times_pd[-1] - pd.Timedelta(weeks=test_weeks)
        split_mask = times_pd < split_time

        train_positions = positions[split_mask]
        train_times = times[split_mask]
        train_weights = weights[split_mask]
        test_positions = positions[~split_mask]
        test_times = times[~split_mask]
        test_weights = weights[~split_mask]

        return train_positions, train_times, train_weights, test_positions, test_times, test_weights

class ProbabilisticEvaluator:
    """Evaluates models using field density scores for tweet prediction"""

    def __init__(self, grid_bounds: Tuple[Tuple[float, float], Tuple[float, float]] = None):
        # Default bounds for 2D good-faith space - will be updated based on data
        self.grid_bounds = grid_bounds
        # Models to create animations for (set by TestingFramework)
        self.animate_models = []
        # Storage for animation frames
        self.animation_frames = {}

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


    def _create_model_animations(self):
        """Create final animations from collected frames"""
        import matplotlib.animation as animation

        for model_name, frames in self.animation_frames.items():
            if not frames:
                continue

            print(f"Creating animation for {model_name} with {len(frames)*2} frames...")

            fig, ax = plt.subplots(1, 1, figsize=(10, 8))

            # Determine global bounds from all frames
            all_x_mins = [frame['grid_bounds'][0][0] for frame in frames]
            all_x_maxs = [frame['grid_bounds'][0][1] for frame in frames]
            all_y_mins = [frame['grid_bounds'][1][0] for frame in frames]
            all_y_maxs = [frame['grid_bounds'][1][1] for frame in frames]

            global_x_min, global_x_max = min(all_x_mins), max(all_x_maxs)
            global_y_min, global_y_max = min(all_y_mins), max(all_y_maxs)

            def animate(frame_idx):
                ax.clear()

                # Each week has 2 frames: prediction only, then prediction + tweets
                week_idx = frame_idx // 2
                is_second_frame = frame_idx % 2 == 1

                if week_idx >= len(frames):
                    return

                frame_data = frames[week_idx]
                predicted_density = frame_data['predicted_density']
                true_positions = frame_data['true_positions']
                frame_id = frame_data['frame_id']
                frame_start = frame_data['frame_start']
                frame_end = frame_data['frame_end']
                field_score = frame_data['field_score']

                # Show predicted density heatmap
                im = ax.imshow(predicted_density,
                              extent=[global_x_min, global_x_max, global_y_min, global_y_max],
                              origin='lower', cmap='viridis', alpha=0.8)

                # Get frame duration from TestingFramework instance
                # Access via the evaluator's parent framework
                testing_framework = getattr(self, 'parent_framework', None)
                if testing_framework and hasattr(testing_framework, 'frame_duration_days'):
                    frame_duration_days = testing_framework.frame_duration_days
                else:
                    frame_duration_days = 7.0  # Default fallback

                # Format frame duration for display
                if frame_duration_days >= 1:
                    duration_str = f"{frame_duration_days:.0f}d" if frame_duration_days == int(frame_duration_days) else f"{frame_duration_days:.1f}d"
                else:
                    hours = frame_duration_days * 24
                    duration_str = f"{hours:.0f}h" if hours == int(hours) else f"{hours:.1f}h"

                frame_label = f"Frame {frame_id} ({duration_str})"

                if not is_second_frame:
                    # Frame 1: Prediction only
                    title = f'{model_name}\n{frame_label} | Prediction Only | Score: {field_score:.3f}'
                else:
                    # Frame 2: Prediction + actual tweets overlaid
                    title = f'{model_name}\n{frame_label} | + Actual Tweets | Score: {field_score:.3f}'

                    # Overlay actual tweet positions
                    if len(true_positions) > 0:
                        ax.scatter(true_positions[:, 0], true_positions[:, 1],
                                  c='red', s=40, marker='x', linewidth=2, alpha=0.9)

                ax.set_title(title, fontsize=11, fontweight='bold')
                ax.set_xlabel('Charity →', fontsize=10)
                ax.set_ylabel('Sincerity →', fontsize=10)
                ax.set_xlim(global_x_min, global_x_max)
                ax.set_ylim(global_y_min, global_y_max)

                # Add frame counter
                ax.text(0.02, 0.98, f'Frame {frame_idx+1}/{len(frames)*2}',
                       transform=ax.transAxes, fontsize=9,
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            # Create animation: 2 frames per week, 1.5 seconds per frame
            total_frames = len(frames) * 2
            anim = animation.FuncAnimation(fig, animate, frames=total_frames,
                                         interval=1500, repeat=True)

            # Save animation
            clean_model_name = model_name.replace(' ', '_').replace('(', '').replace(')', '').replace('=', '').replace('.', '')
            filename = f"model_animation_{clean_model_name}.gif"
            filepath = os.path.join('image_outputs', filename)

            anim.save(filepath, writer='pillow', fps=0.67, dpi=100)
            plt.close()

            print(f"  Saved: {filepath}")

        # Clear frames after creating animations
        self.animation_frames.clear()

    def evaluate_model(self, model: TweetPredictor,
                      train_positions: np.ndarray, train_times: np.ndarray, train_weights: np.ndarray,
                      test_positions: np.ndarray, test_times: np.ndarray, test_weights: np.ndarray,
                      grid_size: int = 100) -> Dict[str, float]:
        """Comprehensive model evaluation with single training, multiple predictions"""

        # TRAIN ONCE: Train the model on all training data
        print(f"Training {model.get_name()} on {len(train_positions)} training tweets...")
        model.fit(train_positions, train_times, train_weights, grid_size)

        # Combine all data and sort by time for sliding prediction
        all_positions = np.vstack([train_positions, test_positions])
        all_times = np.concatenate([train_times, test_times])
        all_weights = np.concatenate([train_weights, test_weights])

        # Sort everything by time
        sort_idx = np.argsort(all_times)
        all_positions = all_positions[sort_idx]
        all_times = all_times[sort_idx]
        all_weights = all_weights[sort_idx]

        # Find the split point (where test data starts)
        original_split_time = train_times.max()

        # Group by custom time frames for evaluation
        all_df = pd.DataFrame({'time': all_times, 'pos': list(all_positions), 'weight': all_weights})
        all_df['datetime'] = pd.to_datetime(all_df['time'])

        # Create custom time frame groupings based on frame_duration_days
        min_time = all_df['datetime'].min()
        max_time = all_df['datetime'].max()

        # Get frame duration from parent framework
        frame_duration_days = getattr(self.parent_framework, 'frame_duration_days')
        frame_duration = pd.Timedelta(days=frame_duration_days)
        time_frames = []
        current_time = min_time
        frame_id = 0

        while current_time < max_time:
            frame_end = current_time + frame_duration
            time_frames.append({
                'frame_id': frame_id,
                'start_time': current_time,
                'end_time': frame_end
            })
            current_time = frame_end
            frame_id += 1

        # Assign each tweet to a time frame
        all_df['frame_id'] = -1
        for frame in time_frames:
            mask = (all_df['datetime'] >= frame['start_time']) & (all_df['datetime'] < frame['end_time'])
            all_df.loc[mask, 'frame_id'] = frame['frame_id']

        # Only evaluate frames that are in the test period
        test_frames_df = all_df[all_df['datetime'] > original_split_time]
        test_frame_ids = test_frames_df['frame_id'].unique()
        test_frame_ids = test_frame_ids[test_frame_ids >= 0]  # Remove any -1 values

        scores = []
        kl_divergences = []
        tweet_counts = []
        total_test_tweets = 0

        for frame_id in test_frame_ids:
            # Get data for this specific frame
            frame_mask = all_df['frame_id'] == frame_id
            frame_data = all_df[frame_mask]
            frame_positions = np.array(frame_data['pos'].tolist())
            frame_times = frame_data['time'].values
            frame_weights = frame_data['weight'].values

            # Only proceed if this frame is in test period
            if not any(pd.to_datetime(frame_times) > original_split_time):
                continue

            total_test_tweets += len(frame_positions)

            # Get frame boundaries for labeling
            frame_info = next(f for f in time_frames if f['frame_id'] == frame_id)
            frame_start = frame_info['start_time']
            frame_end = frame_info['end_time']

            # Update model state with data up to (but not including) this frame
            # Use sliding window for drift field models, expanding window for others
            if hasattr(model, 'params') and 'history_window' in model.params:
                # Drift field model: use sliding window based on history_window
                history_window_days = model.params['history_window'] * frame_duration_days
                current_time = frame_start

                # Calculate start time for sliding window (history_window frames before current frame)
                start_time = current_time - pd.Timedelta(days=history_window_days)
                train_mask = (all_df['datetime'] >= start_time) & (all_df['datetime'] < current_time)

                # If sliding window is empty, fall back to all historical data
                if train_mask.sum() == 0:
                    train_mask = all_df['datetime'] < current_time
            else:
                # Other models: use all historical data (expanding window)
                train_mask = all_df['datetime'] < frame_start

            if train_mask.sum() == 0:
                continue  # Skip if no training data

            frame_train_positions = np.array(all_df[train_mask]['pos'].tolist())
            frame_train_times = all_df[train_mask]['time'].values

            # Update model state for this prediction window (no re-training)
            if hasattr(model, 'update_state'):
                model.update_state(frame_train_positions, frame_train_times)

            # Predict density for this frame
            predicted_density = model.predict_density(frame_times, grid_size)[0]

            # Create true density from observed positions (point-based, no Gaussians)
            true_density = self.create_point_based_density(frame_positions, grid_size)

            # Calculate grid bounds for field density score
            if len(frame_positions) > 0:
                x_min, x_max = frame_positions[:, 0].min(), frame_positions[:, 0].max()
                y_min, y_max = frame_positions[:, 1].min(), frame_positions[:, 1].max()

                # Add padding (same as used in density creation)
                x_padding = max(0.5, (x_max - x_min) * 0.2)
                y_padding = max(0.5, (y_max - y_min) * 0.2)

                grid_bounds = ((x_min - x_padding, x_max + x_padding),
                              (y_min - y_padding, y_max + y_padding))
            else:
                # Default bounds if no tweets
                grid_bounds = ((0, 8), (0, 6))

            # Calculate new field density score (replaces PWS)
            field_score = model.calculate_fds_score(predicted_density, frame_positions, grid_bounds, frame_weights)
            scores.append(field_score)
            tweet_counts.append(len(frame_positions))

            # Plot heatmap for HistoricalAverageModel or GaussianSmoothed models (disabled)
            # if "Historical Average" in model.get_name() or "Gaussian Smoothed" in model.get_name():
            #     frame_label = f"frame_{frame_id}_{frame_start.strftime('%Y%m%d_%H%M')}"
            #     self._plot_prediction_heatmap(predicted_density, true_density, frame_label, model.get_name(), field_score)

            # Collect animation frames for specified models
            should_animate = any(animate_name in model.get_name() for animate_name in self.animate_models)
            if should_animate:
                model_name = model.get_name()
                if model_name not in self.animation_frames:
                    self.animation_frames[model_name] = []

                # Store frame data for this time frame
                animation_frame_data = {
                    'predicted_density': predicted_density.copy(),
                    'true_positions': frame_positions.copy(),
                    'grid_bounds': grid_bounds,
                    'frame_id': frame_id,
                    'frame_start': frame_start,
                    'frame_end': frame_end,
                    'field_score': field_score
                }
                self.animation_frames[model_name].append(animation_frame_data)

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
            'field_density_score': float(weighted_pws),  # Convert to Python float for JSON serialization
            'kl_divergence': float(weighted_kl),
            'score_std': float(np.std(scores)),
            'weeks_evaluated': int(len(scores)),
            'total_test_tweets': int(total_test_tweets)
        }

class TestingFramework:
    """Main framework for testing tweet prediction models"""

    def __init__(self, data_path: str, sample_size: Optional[int] = None,
                 start_date: Optional[str] = None, end_date: Optional[str] = None,
                 animate_models: list = None, grid_size: int = 100,
                 experiment_log_path: str = None, target_topic: str = 'general',
                 frame_duration_days: float = 7.0):
        self.data_loader = DataLoader(data_path, sample_size, start_date, end_date)
        self.evaluator = ProbabilisticEvaluator()
        self.evaluator.parent_framework = self  # Set parent reference for animation access
        self.models = []
        self.animate_models = animate_models or []  # Models to create animations for
        self.grid_size = grid_size  # Grid resolution
        self.target_topic = target_topic
        self.frame_duration_days = frame_duration_days  # Configurable frame duration

        # Create experiment_results directory if it doesn't exist
        experiment_dir = Path('experiment_results')
        experiment_dir.mkdir(exist_ok=True)

        # Generate filename with timestamp if not provided
        if experiment_log_path is None:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            self.experiment_log_path = experiment_dir / f'experiment_{timestamp}.jsonl'
        else:
            self.experiment_log_path = Path(experiment_log_path)
        self.experiment_config = {
            'data_path': data_path,
            'sample_size': sample_size,
            'start_date': start_date,
            'end_date': end_date,
            'grid_size': grid_size,
            'animate_models': animate_models,
            'frame_duration_days': frame_duration_days
        }
        self.sample_size = sample_size

    def add_model(self, model: TweetPredictor):
        """Add a model to be tested"""
        self.models.append(model)

    def run_evaluation(self, test_weeks: int = 1) -> Dict[str, Dict[str, float]]:
        """Run full evaluation pipeline"""
        experiment_start_time = time.time()
        self.experiment_timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

        print("Loading data...")
        positions, times, weights = self.data_loader.load_data()

        print(f"Loaded {len(positions)} tweets")
        print("Creating temporal split...")
        train_pos, train_times, train_weights, test_pos, test_times, test_weights = \
            self.data_loader.temporal_split(positions, times, weights, test_weeks)

        print(f"Training data: {len(train_pos)} tweets")
        print(f"Test data: {len(test_pos)} tweets")

        results = {}

        for model in self.models:
            print(f"\nEvaluating {model.get_name()}...")
            model_start_time = time.time()

            # Pass animation models to evaluator
            self.evaluator.animate_models = self.animate_models
            scores = self.evaluator.evaluate_model(
                model, train_pos, train_times, train_weights, test_pos, test_times, test_weights, self.grid_size
            )

            model_end_time = time.time()
            model_runtime = model_end_time - model_start_time

            results[model.get_name()] = scores

            # Capture model parameters after training
            model_params = self._extract_model_params(model)

            # Log experiment result
            self._log_experiment_result(
                model_name=model.get_name(),
                model_params=model_params,
                scores=scores,
                runtime=model_runtime,
                test_weeks=test_weeks,
                train_size=len(train_pos),
                test_size=len(test_pos)
            )

        # Create animations after all models are evaluated
        if self.animate_models:
            print(f"\nCreating animations for {len(self.evaluator.animation_frames)} models...")
            self.evaluator._create_model_animations()

        return results

    def _extract_model_params(self, model: TweetPredictor) -> Dict[str, Any]:
        """Extract parameters from model after training"""
        params = {}

        # Extract common model parameters
        if hasattr(model, 'params'):
            params = model.params.copy()
        elif hasattr(model, 'bandwidth'):  # HistoricalAverageModel
            params = {'bandwidth': model.bandwidth}
        elif hasattr(model, 'gaussian_bandwidth'):  # GaussianSmoothedHistoricalModel
            params = {'gaussian_bandwidth': model.gaussian_bandwidth}

        return params

    def _log_experiment_result(self, model_name: str, model_params: Dict[str, Any],
                              scores: Dict[str, float], runtime: float, test_weeks: int,
                              train_size: int, test_size: int):
        """Log experiment result to JSONL file with model-specific filename"""
        experiment_entry = {
            'timestamp': datetime.datetime.now().isoformat(),
            'model_name': model_name,
            'model_params': model_params,
            'config': self.experiment_config,
            'scores': scores,
            'runtime_seconds': runtime,
            'test_weeks': test_weeks,
            'train_size': train_size,
            'test_size': test_size
        }

        # Create model-specific filename with target_topic and sample_size
        # Extract base model name (before any parentheses with parameters)
        base_model_name = model_name.split('(')[0].strip()
        clean_model_name = base_model_name.replace(' ', '_')

        # Add sample size to filename
        sample_str = f"sample{self.sample_size}" if self.sample_size else "full"
        model_log_path = Path('experiment_results') / f'{self.target_topic}_{clean_model_name}_{sample_str}_{self.experiment_timestamp}.jsonl'

        # Append to model-specific JSONL file
        with open(model_log_path, 'a') as f:
            f.write(json.dumps(experiment_entry) + '\n')

    def print_results(self, results: Dict[str, Dict[str, float]]):
        """Print formatted results"""
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        print("Field Density Score: 1.0=random, >1.0=better than random, higher=better")
        print(f"Results logged to: experiment_results/ directory")

        for model_name, scores in results.items():
            print(f"\n{model_name}:")
            print(f"  Field Density Score: {scores['field_density_score']:.4f}")
            print(f"  KL Divergence: {scores['kl_divergence']:.4f}")
            print(f"  Score Std Dev: {scores['score_std']:.4f}")
            print(f"  Weeks Evaluated: {scores['weeks_evaluated']}")
            print(f"  Total Test Tweets: {scores['total_test_tweets']}")