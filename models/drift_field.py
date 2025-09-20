import numpy as np
from testing.framework import TweetPredictor
from scipy.ndimage import gaussian_filter
from scipy import stats
from typing import Dict, List, Tuple
import itertools

class DriftFieldModel(TweetPredictor):
    """Drift field model that learns momentum patterns in tweet density flow"""

    # Default learnable parameters
    LEARNABLE_PARAMETERS = {
        # Temporal
        'history_window': [7],                      # How many past timesteps to use for learning momentum patterns
        'temporal_decay': [0.05, 0.1, 0.15, 0.2, 0.3],  # How much total density fades away each timestep (prevents infinite accumulation)

        # Movement & Flow
        'drift_scale': [0.5, 1.0, 1.5, 2.0, 2.5],       # How many grid cells density moves per timestep when following momentum
        'momentum_weight': [0.7],                   # What fraction of moving density follows learned patterns vs spreads randomly
        'density_persistence': [0.1, 0.3, 0.5, 0.7, 0.9],  # What fraction of density stays in the same grid cell vs moves somewhere

        # Spatial Processing
        'correlation_window_size': [3],             # Size of neighborhood (3x3) used to detect where density flowed from
        'local_region_size': [12],                  # Size of spatial chunks for independent momentum calculations

        # Final Prediction
        'diffusion_strength': [0.3],                # How much Gaussian blur to apply to final prediction (uncertainty spreading)

        # Tweet Importance in FDS
        'retweet_importance_weight': [1.5],         # How much more retweeted tweets count in the Field Density Score metric
    }

    def __init__(self, params: Dict = None, grid_size: int = None):
        """Initialize with specific parameters or defaults"""
        if params is None:
            # Use first value from each parameter list as default
            params = {key: values[0] for key, values in self.LEARNABLE_PARAMETERS.items()}

        self.params = params
        self.grid_size = grid_size  # Will be set when predict_density is called
        self.training_grid_size = None  # Will be set during fit()
        self.density_history = []
        self.velocity_history = []
        self.spatial_bounds = None

    def fit(self, train_data: np.ndarray, train_times: np.ndarray, grid_size: int = 50) -> None:
        """Learn density flow patterns and optimize parameters from historical data"""
        if len(train_data) == 0:
            return

        # Set training grid size
        self.training_grid_size = grid_size

        print(f"Training drift field model with {len(train_data)} tweets on {grid_size}x{grid_size} grid...")

        # Set spatial bounds based on training data
        self._set_spatial_bounds(train_data)

        # Group data by time periods (assume weekly for now)
        time_groups = self._group_by_time_periods(train_data, train_times)

        if len(time_groups) < 3:
            print("Warning: Not enough time periods for parameter optimization")
            # Use default parameters and build basic model
            self._build_model_with_params(time_groups, self.params)
            return

        # Optimize parameters using sliding window cross-validation within training data
        print("Optimizing parameters...")
        best_params = self._optimize_parameters(time_groups)
        self.params = best_params

        print(f"Best parameters found: {best_params}")

        # Build final model with best parameters using ALL training data
        self._build_model_with_params(time_groups, best_params)

        # Store original training data for potential state updates
        self.original_train_data = train_data.copy()
        self.original_train_times = train_times.copy()

    def update_state(self, current_data: np.ndarray, current_times: np.ndarray) -> None:
        """Update model's internal state (density/velocity history) with new data
        WITHOUT re-training parameters. Used for sliding window predictions.
        """
        if len(current_data) == 0:
            return

        # Rebuild density/velocity history with current data using EXISTING parameters
        time_groups = self._group_by_time_periods(current_data, current_times)
        self._build_model_with_params(time_groups, self.params)

    def _optimize_parameters(self, time_groups: List[np.ndarray]) -> Dict:
        """Optimize parameters using sliding window validation"""
        param_combinations = self._generate_parameter_combinations()

        # Use subset for faster optimization
        max_combinations = 2  # Limit to top combinations
        if len(param_combinations) > max_combinations:
            # Sample diverse combinations
            import random
            random.seed(42)  # Reproducible
            param_combinations = random.sample(param_combinations, max_combinations)

        print(f"Testing {len(param_combinations)} parameter combinations...")

        best_fds = -float('inf')
        best_params = None

        for i, params in enumerate(param_combinations):
            # Cross-validate this parameter set
            avg_fds = self._cross_validate_params(time_groups, params)

            if avg_fds > best_fds:
                best_fds = avg_fds
                best_params = params

            if (i + 1) % 10 == 0:
                print(f"  Tested {i+1}/{len(param_combinations)}, best FDS: {best_fds:.6f}")

        return best_params

    def _cross_validate_params(self, time_groups: List[np.ndarray], params: Dict) -> float:
        """Cross-validate a parameter set using sliding window"""
        min_train_periods = max(3, params['history_window'])

        if len(time_groups) < min_train_periods + 1:
            return 0.0  # Not enough data

        fds_scores = []
        total_tweets = 0

        # Sliding window: predict each period using only previous periods
        test_periods = list(range(min_train_periods, len(time_groups)))
        total_tests = len(test_periods)

        for i, test_idx in enumerate(test_periods):
            # Show progress bar for this parameter combination's cross-validation
            progress = (i + 1) / total_tests
            bar_length = 20
            filled_length = int(bar_length * progress)
            bar = '█' * filled_length + '-' * (bar_length - filled_length)
            print(f"\r    CV Progress: [{bar}] {i+1}/{total_tests} ({progress:.1%})", end='', flush=True)
            train_periods = time_groups[:test_idx]
            test_period = time_groups[test_idx]

            if len(test_period) == 0:
                continue

            # Build temporary model with these parameters
            temp_density_history = []
            for period_data in train_periods:
                density_grid = self._tweets_to_density_grid(period_data)
                temp_density_history.append(density_grid)

            temp_velocity_history = []
            for i in range(1, len(temp_density_history)):
                velocity_field = self._calculate_velocity_field(
                    temp_density_history[i-1], temp_density_history[i]
                )
                temp_velocity_history.append(velocity_field)

            # Make prediction
            if len(temp_density_history) > 0:
                predicted_density = self._predict_with_params(
                    temp_density_history, temp_velocity_history, params
                )

                # Calculate FDS for this prediction
                fds_score = self._calculate_fds_score(predicted_density, test_period, params)
                fds_scores.append(fds_score * len(test_period))  # Weight by tweet count
                total_tweets += len(test_period)

        # Clear the progress bar line and return weighted average FDS
        print()  # New line after progress bar
        if total_tweets > 0:
            return sum(fds_scores) / total_tweets
        else:
            return 0.0

    def _predict_with_params(self, density_history: List[np.ndarray],
                           velocity_history: List[np.ndarray], params: Dict) -> np.ndarray:
        """Make prediction using specific parameters"""
        if len(density_history) == 0:
            return np.ones((self.grid_size, self.grid_size)) / (self.grid_size ** 2)

        current_density = density_history[-1].copy()

        if len(velocity_history) == 0:
            # No history - apply temporal decay only
            next_density = current_density * (1 - params['temporal_decay'])
        else:
            # Use recent velocity fields
            history_window = min(params['history_window'], len(velocity_history))
            recent_velocities = velocity_history[-history_window:]

            # Average velocity field
            avg_velocity = np.zeros_like(recent_velocities[0])
            total_weight = 0

            for i, vel_field in enumerate(reversed(recent_velocities)):
                weight = 0.8 ** i  # Exponential decay
                avg_velocity += vel_field * weight
                total_weight += weight

            if total_weight > 0:
                avg_velocity /= total_weight

            # Apply drift field with these parameters
            next_density = self._apply_momentum_and_persistence_with_params(
                current_density, avg_velocity, params
            )

        # Apply diffusion
        next_density = gaussian_filter(next_density, sigma=params['diffusion_strength'])

        # Normalize
        density_sum = next_density.sum()
        if density_sum > 0:
            next_density = next_density / density_sum
        else:
            next_density = np.ones_like(next_density) / next_density.size

        return next_density

    def _apply_momentum_and_persistence_with_params(self, current_density: np.ndarray,
                                                   avg_velocity: np.ndarray, params: Dict) -> np.ndarray:
        """Apply drift field with specific parameters"""
        current_grid_size = current_density.shape[0]
        next_density = np.zeros_like(current_density)

        for i in range(current_grid_size):
            for j in range(current_grid_size):
                current_cell_density = current_density[i, j]

                if current_cell_density < 0.001:
                    continue

                # Split into staying vs moving
                staying_density = current_cell_density * params['density_persistence']
                moving_density = current_cell_density * (1 - params['density_persistence'])

                next_density[i, j] += staying_density

                # Split moving into momentum vs random
                momentum_driven = moving_density * params['momentum_weight']
                random_portion = moving_density * (1 - params['momentum_weight'])

                # Apply momentum
                velocity = avg_velocity[i, j]
                drift_distance = params['drift_scale']

                target_i = i + velocity[0] * drift_distance
                target_j = j + velocity[1] * drift_distance

                self._distribute_density_bilinear(next_density, momentum_driven, target_i, target_j)
                self._distribute_density_locally(next_density, random_portion, i, j)

        # Apply temporal decay
        next_density *= (1 - params['temporal_decay'])

        return next_density

    def _calculate_fds_score(self, predicted_density: np.ndarray, actual_tweets: np.ndarray, params: Dict) -> float:
        """Calculate Field Density Score for prediction"""
        if len(actual_tweets) == 0:
            return 1.0  # Neutral score

        total_score = 0
        total_weight = 0

        # Map tweet positions to grid coordinates
        current_grid_size = predicted_density.shape[0]
        x_coords = (actual_tweets[:, 0] - self.spatial_bounds['x_min']) / (
            self.spatial_bounds['x_max'] - self.spatial_bounds['x_min']) * (current_grid_size - 1)
        y_coords = (actual_tweets[:, 1] - self.spatial_bounds['y_min']) / (
            self.spatial_bounds['y_max'] - self.spatial_bounds['y_min']) * (current_grid_size - 1)

        # Clamp to grid bounds
        x_coords = np.clip(x_coords, 0, current_grid_size - 1).astype(int)
        y_coords = np.clip(y_coords, 0, current_grid_size - 1).astype(int)

        # Calculate FDS
        for i in range(len(actual_tweets)):
            grid_x, grid_y = x_coords[i], y_coords[i]
            predicted_prob = predicted_density[grid_y, grid_x]  # Note: y,x for matrix indexing

            # Simple uniform weighting for now (could add retweet weighting later)
            weight = 1.0

            total_score += predicted_prob * weight
            total_weight += weight

        return total_score / total_weight if total_weight > 0 else 1.0

    def _build_model_with_params(self, time_groups: List[np.ndarray], params: Dict):
        """Build the final model with optimized parameters"""
        # Convert each time period to density grid
        self.density_history = []
        for period_data in time_groups:
            density_grid = self._tweets_to_density_grid(period_data)
            self.density_history.append(density_grid)

        # Calculate velocity fields between consecutive periods
        self.velocity_history = []
        for i in range(1, len(self.density_history)):
            velocity_field = self._calculate_velocity_field(
                self.density_history[i-1],
                self.density_history[i]
            )
            self.velocity_history.append(velocity_field)

    def _generate_parameter_combinations(self):
        """Generate parameter combinations for optimization"""
        param_names = list(self.LEARNABLE_PARAMETERS.keys())
        param_values = list(self.LEARNABLE_PARAMETERS.values())

        combinations = []
        for combo in itertools.product(*param_values):
            param_dict = dict(zip(param_names, combo))
            combinations.append(param_dict)

        return combinations

    def predict_density(self, test_times: np.ndarray, grid_size: int = None) -> np.ndarray:
        """Predict density using drift field approach"""
        if grid_size is None:
            grid_size = self.training_grid_size or 50
        
        # Set the grid size for predictions
        self.grid_size = grid_size

        if len(self.density_history) == 0:
            # No training data - return uniform
            uniform_density = np.ones((grid_size, grid_size)) / (grid_size ** 2)
            return np.array([uniform_density] * len(test_times))

        predictions = []

        # Scale most recent density to prediction grid size if needed
        current_density = self.density_history[-1].copy()
        if current_density.shape[0] != grid_size:
            current_density = self._rescale_density_grid(current_density, grid_size)

        # For each test period, apply drift field prediction
        for _ in test_times:
            predicted_density = self._apply_drift_field(current_density)
            predictions.append(predicted_density)

            # Update current density for next prediction
            current_density = predicted_density

        return np.array(predictions)

    def _rescale_density_grid(self, density: np.ndarray, target_size: int) -> np.ndarray:
        """Rescale density grid to different size"""
        from scipy.ndimage import zoom

        if density.shape[0] == target_size:
            return density

        scale_factor = target_size / density.shape[0]
        rescaled = zoom(density, scale_factor, order=1)  # Bilinear interpolation

        # Renormalize
        rescaled = rescaled / rescaled.sum()

        return rescaled

    def _set_spatial_bounds(self, train_data: np.ndarray):
        """Set grid bounds based on training data"""
        x_min, x_max = train_data[:, 0].min(), train_data[:, 0].max()
        y_min, y_max = train_data[:, 1].min(), train_data[:, 1].max()

        # Add padding
        x_padding = max(0.5, (x_max - x_min) * 0.2)
        y_padding = max(0.5, (y_max - y_min) * 0.2)

        self.spatial_bounds = {
            'x_min': x_min - x_padding,
            'x_max': x_max + x_padding,
            'y_min': y_min - y_padding,
            'y_max': y_max + y_padding
        }

    def _group_by_time_periods(self, train_data: np.ndarray, train_times: np.ndarray) -> List[np.ndarray]:
        """Group tweets into time periods (weekly bins for now)"""
        if len(train_times) == 0:
            return []

        # Convert times to weekly periods
        min_time = train_times.min()
        week_indices = ((train_times - min_time) / np.timedelta64(7, 'D')).astype(int)

        time_groups = []
        for week in range(week_indices.max() + 1):
            week_mask = week_indices == week
            if week_mask.any():
                time_groups.append(train_data[week_mask])

        return time_groups

    def _tweets_to_density_grid(self, tweets: np.ndarray) -> np.ndarray:
        """Convert tweet positions to density grid"""
        current_grid_size = self.training_grid_size or self.grid_size or 50

        if len(tweets) == 0:
            return np.ones((current_grid_size, current_grid_size)) / (current_grid_size ** 2)

        # Create grid coordinates
        x_grid = np.linspace(self.spatial_bounds['x_min'], self.spatial_bounds['x_max'], current_grid_size)
        y_grid = np.linspace(self.spatial_bounds['y_min'], self.spatial_bounds['y_max'], current_grid_size)
        xx, yy = np.meshgrid(x_grid, y_grid)

        # Initialize density grid
        density = np.zeros((current_grid_size, current_grid_size))

        # Place small Gaussian around each tweet
        gaussian_sigma = 0.5  # Small radius of influence
        for tweet_pos in tweets:
            # Calculate squared distances
            dist_sq = ((xx - tweet_pos[0])**2 + (yy - tweet_pos[1])**2)
            # Add Gaussian contribution
            density += np.exp(-dist_sq / (2 * gaussian_sigma**2))

        # Normalize
        density_sum = density.sum()
        if density_sum > 0:
            density = density / density_sum
        else:
            density = np.ones((current_grid_size, current_grid_size)) / (current_grid_size ** 2)

        return density

    def _calculate_velocity_field(self, prev_density: np.ndarray, curr_density: np.ndarray) -> np.ndarray:
        """Calculate velocity field showing how density flowed between timesteps"""
        current_grid_size = prev_density.shape[0]  # Use actual grid size from density arrays
        velocity_field = np.zeros((current_grid_size, current_grid_size, 2))  # (x_vel, y_vel) for each cell

        window_size = self.params['correlation_window_size']
        half_window = window_size // 2

        # For each grid cell, find where density likely came from
        for i in range(current_grid_size):
            for j in range(current_grid_size):
                current_density_val = curr_density[i, j]

                if current_density_val < 0.001:  # Skip very low density cells
                    continue

                best_correlation = -1
                best_direction = (0, 0)

                # Search in neighborhood
                for di in range(-half_window, half_window + 1):
                    for dj in range(-half_window, half_window + 1):
                        if di == 0 and dj == 0:
                            continue

                        source_i, source_j = i + di, j + dj

                        if 0 <= source_i < current_grid_size and 0 <= source_j < current_grid_size:
                            # Check correlation between previous source and current target
                            prev_source_val = prev_density[source_i, source_j]

                            # Simple correlation: if source had high density before
                            # and target has high density now, that's a good flow direction
                            correlation = prev_source_val * current_density_val

                            if correlation > best_correlation:
                                best_correlation = correlation
                                best_direction = (di, dj)

                # Normalize direction vector
                if best_direction != (0, 0):
                    norm = np.sqrt(best_direction[0]**2 + best_direction[1]**2)
                    velocity_field[i, j] = (best_direction[0]/norm, best_direction[1]/norm)

        return velocity_field

    def _apply_drift_field(self, current_density: np.ndarray) -> np.ndarray:
        """Apply drift field to predict next density"""
        # Calculate average momentum from recent velocity fields
        if len(self.velocity_history) == 0:
            # No history - apply temporal decay only
            next_density = current_density * (1 - self.params['temporal_decay'])
        else:
            # Use recent velocity fields to calculate momentum
            history_window = min(self.params['history_window'], len(self.velocity_history))
            recent_velocities = self.velocity_history[-history_window:]

            # Average velocity field with temporal decay weights
            avg_velocity = np.zeros_like(recent_velocities[0])
            total_weight = 0

            for i, vel_field in enumerate(reversed(recent_velocities)):
                weight = 0.8 ** i  # Exponential decay (recent more important)
                avg_velocity += vel_field * weight
                total_weight += weight

            if total_weight > 0:
                avg_velocity /= total_weight

            # Apply drift field formula
            next_density = self._apply_momentum_and_persistence(current_density, avg_velocity)

        # Apply final diffusion (uncertainty blur)
        next_density = gaussian_filter(next_density, sigma=self.params['diffusion_strength'])

        # Normalize to sum to 1
        density_sum = next_density.sum()
        if density_sum > 0:
            next_density = next_density / density_sum
        else:
            next_density = np.ones_like(next_density) / next_density.size

        return next_density

    def _apply_momentum_and_persistence(self, current_density: np.ndarray, avg_velocity: np.ndarray) -> np.ndarray:
        """Apply the core drift field formula with persistence and momentum"""
        current_grid_size = current_density.shape[0]
        next_density = np.zeros_like(current_density)

        for i in range(current_grid_size):
            for j in range(current_grid_size):
                current_cell_density = current_density[i, j]

                if current_cell_density < 0.001:  # Skip very low density
                    continue

                # Step 1: Split into staying vs moving portions
                staying_density = current_cell_density * self.params['density_persistence']
                moving_density = current_cell_density * (1 - self.params['density_persistence'])

                # Add staying portion to same cell
                next_density[i, j] += staying_density

                # Step 2: Split moving portion into momentum vs random
                momentum_driven = moving_density * self.params['momentum_weight']
                random_portion = moving_density * (1 - self.params['momentum_weight'])

                # Step 3: Apply momentum to momentum-driven portion
                velocity = avg_velocity[i, j]
                drift_distance = self.params['drift_scale']

                target_i = i + velocity[0] * drift_distance
                target_j = j + velocity[1] * drift_distance

                # Distribute momentum-driven density with bilinear interpolation
                self._distribute_density_bilinear(next_density, momentum_driven, target_i, target_j)

                # Step 4: Spread random portion locally (no momentum)
                self._distribute_density_locally(next_density, random_portion, i, j)

        # Apply temporal decay
        next_density *= (1 - self.params['temporal_decay'])

        return next_density

    def _distribute_density_bilinear(self, density_grid: np.ndarray, amount: float, target_i: float, target_j: float):
        """Distribute density using bilinear interpolation"""
        current_grid_size = density_grid.shape[0]

        # Clamp to grid bounds. small epsilon to handle floating-point edge cases in bilinear interpolation.
        target_i = max(0, min(current_grid_size - 1 - 1e-6, target_i))
        target_j = max(0, min(current_grid_size - 1 - 1e-6, target_j))

        # Get integer and fractional parts
        i_low, i_high = int(target_i), int(target_i) + 1
        j_low, j_high = int(target_j), int(target_j) + 1

        i_frac = target_i - i_low
        j_frac = target_j - j_low

        # Clamp indices
        i_high = min(i_high, current_grid_size - 1)
        j_high = min(j_high, current_grid_size - 1)

        # Bilinear weights
        w_ll = (1 - i_frac) * (1 - j_frac)  # lower-left
        w_lh = (1 - i_frac) * j_frac        # lower-right
        w_hl = i_frac * (1 - j_frac)        # upper-left
        w_hh = i_frac * j_frac              # upper-right

        # Distribute density
        density_grid[i_low, j_low] += amount * w_ll
        density_grid[i_low, j_high] += amount * w_lh
        density_grid[i_high, j_low] += amount * w_hl
        density_grid[i_high, j_high] += amount * w_hh

    def _distribute_density_locally(self, density_grid: np.ndarray, amount: float, center_i: int, center_j: int):
        """Distribute density randomly in local neighborhood"""
        # Simple: add to current cell (could make more sophisticated)
        density_grid[center_i, center_j] += amount

    def get_name(self) -> str:
        """Return model name for logging"""
        param_str = f"decay={self.params['temporal_decay']:.2f}_drift={self.params['drift_scale']:.1f}_persist={self.params['density_persistence']:.1f}"
        return f"DriftField({param_str})"

    @classmethod
    def generate_all_parameter_combinations(cls):
        """Generate all possible parameter combinations for grid search"""
        param_names = list(cls.LEARNABLE_PARAMETERS.keys())
        param_values = list(cls.LEARNABLE_PARAMETERS.values())

        combinations = []
        for combo in itertools.product(*param_values):
            param_dict = dict(zip(param_names, combo))
            combinations.append(param_dict)

        return combinations