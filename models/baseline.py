import numpy as np
from testing.framework import TweetPredictor
from sklearn.neighbors import KernelDensity

class HistoricalAverageModel(TweetPredictor):
    """Baseline model that predicts based on historical tweet density patterns"""

    def __init__(self, bandwidth: float = 0.2):
        self.bandwidth = bandwidth
        self.train_positions = None

    def fit(self, train_data: np.ndarray, train_times: np.ndarray) -> None:
        """Learn historical density pattern from training data"""
        self.train_positions = train_data
        if len(train_data) == 0:
            return

        # Fit KDE to all historical tweet positions
        self.kde = KernelDensity(bandwidth=self.bandwidth, kernel='gaussian')
        self.kde.fit(train_data)

    def predict_density(self, test_times: np.ndarray, grid_size: int = 50) -> np.ndarray:
        """Predict density as historical average for all time periods"""
        if self.train_positions is None or len(self.train_positions) == 0:
            print('Return uniform distribution because no training data')
            uniform_density = np.ones((grid_size, grid_size)) / (grid_size ** 2)
            return np.array([uniform_density] * len(test_times))

        # Dynamic grid bounds based on training data
        x_min, x_max = self.train_positions[:, 0].min(), self.train_positions[:, 0].max()
        y_min, y_max = self.train_positions[:, 1].min(), self.train_positions[:, 1].max()

        # Add padding
        x_padding = max(0.5, (x_max - x_min) * 0.2)
        y_padding = max(0.5, (y_max - y_min) * 0.2)

        x_bounds = (x_min - x_padding, x_max + x_padding)
        y_bounds = (y_min - y_padding, y_max + y_padding)

        # Create grid
        x_grid = np.linspace(x_bounds[0], x_bounds[1], grid_size)
        y_grid = np.linspace(y_bounds[0], y_bounds[1], grid_size)
        xx, yy = np.meshgrid(x_grid, y_grid)
        grid_points = np.column_stack([xx.ravel(), yy.ravel()])

        # Evaluate KDE on grid
        log_density = self.kde.score_samples(grid_points)
        density = np.exp(log_density).reshape(grid_size, grid_size)

        # Normalize to sum to 1
        density_sum = density.sum()
        if density_sum == 0:
            density = np.ones((grid_size, grid_size)) / (grid_size ** 2)
        else:
            density = density / density_sum

        # Return same density for all time periods (historical average)
        return np.array([density] * len(test_times))

    def get_name(self) -> str:
        return f"Historical Average (bandwidth={self.bandwidth})"

class RandomModel(TweetPredictor):
    """Random baseline that predicts uniform distribution"""

    def __init__(self):
        pass

    def fit(self, train_data: np.ndarray, train_times: np.ndarray) -> None:
        """No training needed for random model"""
        pass

    def predict_density(self, test_times: np.ndarray, grid_size: int = 50) -> np.ndarray:
        """Always predict uniform distribution"""
        uniform_density = np.ones((grid_size, grid_size)) / (grid_size ** 2)
        return np.array([uniform_density] * len(test_times))

    def get_name(self) -> str:
        return "Random Uniform"

class WeeklyAverageModel(TweetPredictor):
    """Model that learns different density patterns for different days of week"""

    def __init__(self, bandwidth: float = 0.2):
        self.bandwidth = bandwidth
        self.weekly_patterns = {}
        self.train_positions = None

    def fit(self, train_data: np.ndarray, train_times: np.ndarray) -> None:
        """Learn density patterns for each day of week"""
        import pandas as pd

        self.train_positions = train_data
        if len(train_data) == 0:
            return

        # Group data by day of week
        times_df = pd.to_datetime(train_times)
        for day in range(7):  # 0=Monday, 6=Sunday
            day_mask = times_df.dayofweek == day
            day_data = train_data[day_mask]

            if len(day_data) > 0:
                kde = KernelDensity(bandwidth=self.bandwidth, kernel='gaussian')
                kde.fit(day_data)
                self.weekly_patterns[day] = kde

    def predict_density(self, test_times: np.ndarray, grid_size: int = 50) -> np.ndarray:
        """Predict density based on day of week patterns"""
        import pandas as pd

        if not self.weekly_patterns or self.train_positions is None:
            # Return uniform if no patterns learned
            uniform_density = np.ones((grid_size, grid_size)) / (grid_size ** 2)
            return np.array([uniform_density] * len(test_times))

        # Dynamic grid bounds based on training data
        x_min, x_max = self.train_positions[:, 0].min(), self.train_positions[:, 0].max()
        y_min, y_max = self.train_positions[:, 1].min(), self.train_positions[:, 1].max()

        # Add padding
        x_padding = max(0.5, (x_max - x_min) * 0.2)
        y_padding = max(0.5, (y_max - y_min) * 0.2)

        x_bounds = (x_min - x_padding, x_max + x_padding)
        y_bounds = (y_min - y_padding, y_max + y_padding)

        # Create grid
        x_grid = np.linspace(x_bounds[0], x_bounds[1], grid_size)
        y_grid = np.linspace(y_bounds[0], y_bounds[1], grid_size)
        xx, yy = np.meshgrid(x_grid, y_grid)
        grid_points = np.column_stack([xx.ravel(), yy.ravel()])

        predictions = []
        times_df = pd.to_datetime(test_times)

        for time in times_df:
            day_of_week = time.dayofweek

            if day_of_week in self.weekly_patterns:
                kde = self.weekly_patterns[day_of_week]
                log_density = kde.score_samples(grid_points)
                density = np.exp(log_density).reshape(grid_size, grid_size)
                density_sum = density.sum()
                if density_sum == 0:
                    density = np.ones((grid_size, grid_size)) / (grid_size ** 2)
                else:
                    density = density / density_sum
            else:
                # Fall back to uniform if no data for this day
                density = np.ones((grid_size, grid_size)) / (grid_size ** 2)

            predictions.append(density)

        return np.array(predictions)

    def get_name(self) -> str:
        return f"Weekly Average (bandwidth={self.bandwidth})"