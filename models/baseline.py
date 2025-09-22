import numpy as np
from testing.framework import TweetPredictor
from sklearn.neighbors import KernelDensity

class HistoricalAverageModel(TweetPredictor):
    """Baseline model that predicts based on historical tweet density patterns"""

    def __init__(self, bandwidth: float = 0.2):
        self.bandwidth = bandwidth
        self.train_positions = None

    def fit(self, train_data: np.ndarray, train_times: np.ndarray, train_weights: np.ndarray = None, grid_size: int = 50) -> None:
        """Learn historical density pattern from training data"""
        self.train_positions = train_data
        if len(train_data) == 0:
            return

        # Fit KDE to all historical tweet positions
        # Note: scikit-learn KDE doesn't support sample weights, so we ignore weights for now
        # Could be enhanced to use weighted KDE in the future
        self.kde = KernelDensity(bandwidth=self.bandwidth, kernel='gaussian')
        self.kde.fit(train_data)

    def update_state(self, current_data: np.ndarray, current_times: np.ndarray) -> None:
        """Update model state - for historical average, no state update needed"""
        pass  # Historical average doesn't change based on sliding window

    def predict_density(self, test_times: np.ndarray, grid_size: int = 100) -> np.ndarray:
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

    def fit(self, train_data: np.ndarray, train_times: np.ndarray, train_weights: np.ndarray = None, grid_size: int = 50) -> None:
        """No training needed for random model"""
        pass

    def update_state(self, current_data: np.ndarray, current_times: np.ndarray) -> None:
        """Update model state - random model has no state to update"""
        pass  # Random model doesn't use historical data

    def predict_density(self, test_times: np.ndarray, grid_size: int = 100) -> np.ndarray:
        """Always predict uniform distribution"""
        uniform_density = np.ones((grid_size, grid_size)) / (grid_size ** 2)
        return np.array([uniform_density] * len(test_times))

    def get_name(self) -> str:
        return "Random Uniform"


class GaussianSmoothedHistoricalModel(TweetPredictor):
    """Model that places Gaussians around each training tweet instead of delta functions"""

    def __init__(self, gaussian_bandwidth: float = 0.05):
        self.gaussian_bandwidth = gaussian_bandwidth
        self.train_positions = None

    def fit(self, train_data: np.ndarray, train_times: np.ndarray, train_weights: np.ndarray = None, grid_size: int = 50) -> None:
        """Store training data for Gaussian smoothing"""
        self.train_positions = train_data
        self.train_weights = train_weights  # Store weights for potential weighted Gaussian placement

    def update_state(self, current_data: np.ndarray, current_times: np.ndarray) -> None:
        """Update model state - for Gaussian smoothed, no state update needed"""
        pass  # Gaussian smoothed uses all training data, not sliding window

    def predict_density(self, test_times: np.ndarray, grid_size: int = 100) -> np.ndarray:
        """Predict density by placing Gaussians around each training tweet"""
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

        # Create dense grid
        x_grid = np.linspace(x_bounds[0], x_bounds[1], grid_size)
        y_grid = np.linspace(y_bounds[0], y_bounds[1], grid_size)
        xx, yy = np.meshgrid(x_grid, y_grid)

        # Initialize density grid
        density = np.zeros((grid_size, grid_size))

        # Place Gaussian around each training tweet
        weights = self.train_weights if self.train_weights is not None else np.ones(len(self.train_positions))
        for i, position in enumerate(self.train_positions):
            # Calculate squared distances from this tweet to all grid points
            dist_sq = ((xx - position[0])**2 + (yy - position[1])**2)

            # Add weighted Gaussian contribution
            gaussian_contrib = np.exp(-dist_sq / (2 * self.gaussian_bandwidth**2))
            density += gaussian_contrib * weights[i]

        # Normalize to sum to 1
        density_sum = density.sum()
        if density_sum == 0:
            density = np.ones((grid_size, grid_size)) / (grid_size ** 2)
        else:
            density = density / density_sum

        # Return same density for all time periods (historical average with Gaussians)
        return np.array([density] * len(test_times))

    def get_name(self) -> str:
        return f"Gaussian Smoothed Historical (σ={self.gaussian_bandwidth})"