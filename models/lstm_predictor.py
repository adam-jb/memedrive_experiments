import numpy as np
import pandas as pd
from typing import Optional
from testing.framework import TweetPredictor
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

class LSTMDensityPredictor(nn.Module):
    """LSTM model for predicting tweet density grids"""

    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.1,
                 learn_sigma: bool = False, initial_sigma: float = 0.05):
        super(LSTMDensityPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.learn_sigma = learn_sigma

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.fc = nn.Linear(hidden_size, input_size)
        self.softmax = nn.Softmax(dim=-1)

        # Learnable sigma parameter (log-space for numerical stability)
        if learn_sigma:
            self.log_sigma = nn.Parameter(torch.log(torch.tensor(initial_sigma)))
        else:
            self.register_buffer('log_sigma', torch.log(torch.tensor(initial_sigma)))

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        batch_size = x.size(0)

        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

        # Forward propagate LSTM
        lstm_out, (hn, cn) = self.lstm(x, (h0, c0))

        # Use the last output for prediction
        prediction = self.fc(lstm_out[:, -1, :])

        # Apply softmax to ensure valid probability distribution
        prediction = self.softmax(prediction)

        return prediction

    def get_sigma(self):
        """Get current sigma value"""
        return torch.exp(self.log_sigma).item()

class LSTMTweetPredictor(TweetPredictor):
    """LSTM-based tweet prediction model using time series of density grids"""

    def __init__(self, sequence_length: int = 7, hidden_size: int = 64, num_layers: int = 2,
                 learning_rate: float = 0.001, epochs: int = 50, batch_size: int = 32,
                 gaussian_sigma: float = 0.05, frame_duration_days: float = 7.0, learn_sigma: bool = True):
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.gaussian_sigma = gaussian_sigma
        self.frame_duration_days = frame_duration_days
        self.learn_sigma = learn_sigma

        # Store parameters for JSON serialization (framework expects this)
        self.params = {
            'sequence_length': int(sequence_length),
            'hidden_size': int(hidden_size),
            'num_layers': int(num_layers),
            'learning_rate': float(learning_rate),
            'epochs': int(epochs),
            'batch_size': int(batch_size),
            'gaussian_sigma': float(gaussian_sigma),
            'frame_duration_days': float(frame_duration_days),
            'learn_sigma': bool(learn_sigma)
        }

        self.model = None
        self.scaler = StandardScaler()
        self.grid_size = None
        # Use GPU on Apple Silicon (MPS) or CUDA if available
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        # Store training data for sequence generation
        self.train_positions = None
        self.train_times = None

        # Store frame data for learnable sigma training
        self.frame_tweet_positions = None  # List of tweet positions for each frame
        self.frame_tweet_weights = None    # List of tweet weights for each frame
        self.frame_bounds = None
        self.frame_sequence_indices = None  # Track which frames are used in sequences

    def _create_density_grid_from_tweets(self, positions: np.ndarray, grid_size: int,
                                        bounds: tuple) -> np.ndarray:
        """Create density grid by placing Gaussians around tweet positions"""
        if len(positions) == 0:
            return np.ones((grid_size, grid_size)) / (grid_size ** 2)

        (x_min, x_max), (y_min, y_max) = bounds

        # Create coordinate grids
        x_grid = np.linspace(x_min, x_max, grid_size)
        y_grid = np.linspace(y_min, y_max, grid_size)
        xx, yy = np.meshgrid(x_grid, y_grid)

        # Initialize density grid
        density = np.zeros((grid_size, grid_size))

        # Place Gaussian around each tweet
        for position in positions:
            # Calculate squared distances from this tweet to all grid points
            dist_sq = ((xx - position[0])**2 + (yy - position[1])**2)

            # Add Gaussian contribution
            gaussian_contrib = np.exp(-dist_sq / (2 * self.gaussian_sigma**2))
            density += gaussian_contrib

        # Normalize to sum to 1
        density_sum = density.sum()
        if density_sum == 0:
            return np.ones((grid_size, grid_size)) / (grid_size ** 2)

        return density / density_sum

    def _create_density_grid_torch(self, positions: np.ndarray, grid_size: int,
                                  bounds: tuple, sigma: float) -> torch.Tensor:
        """Create density grid using PyTorch (for learnable sigma)"""
        if len(positions) == 0:
            return torch.ones((grid_size, grid_size), device=self.device) / (grid_size ** 2)

        (x_min, x_max), (y_min, y_max) = bounds

        # Create coordinate grids on GPU
        x_grid = torch.linspace(x_min, x_max, grid_size, device=self.device)
        y_grid = torch.linspace(y_min, y_max, grid_size, device=self.device)
        xx, yy = torch.meshgrid(x_grid, y_grid, indexing='xy')

        # Convert positions to tensor
        positions_tensor = torch.tensor(positions, dtype=torch.float32, device=self.device)

        # Initialize density grid
        density = torch.zeros((grid_size, grid_size), device=self.device)

        # Place Gaussian around each tweet
        for position in positions_tensor:
            # Calculate squared distances
            dist_sq = ((xx - position[0])**2 + (yy - position[1])**2)
            # Add Gaussian contribution
            gaussian_contrib = torch.exp(-dist_sq / (2 * sigma**2))
            density += gaussian_contrib

        # Normalize to sum to 1
        density_sum = density.sum()
        if density_sum == 0:
            return torch.ones((grid_size, grid_size), device=self.device) / (grid_size ** 2)

        return density / density_sum

    def _create_density_grid_torch_differentiable(self, positions: np.ndarray, grid_size: int,
                                                  bounds: tuple, sigma: torch.Tensor) -> torch.Tensor:
        """Create density grid using PyTorch with differentiable sigma tensor"""
        if len(positions) == 0:
            return torch.ones((grid_size, grid_size), device=self.device) / (grid_size ** 2)

        (x_min, x_max), (y_min, y_max) = bounds

        # Create coordinate grids on GPU
        x_grid = torch.linspace(x_min, x_max, grid_size, device=self.device)
        y_grid = torch.linspace(y_min, y_max, grid_size, device=self.device)
        xx, yy = torch.meshgrid(x_grid, y_grid, indexing='xy')

        # Convert positions to tensor
        positions_tensor = torch.tensor(positions, dtype=torch.float32, device=self.device)

        # Initialize density grid
        density = torch.zeros((grid_size, grid_size), device=self.device)

        # Place Gaussian around each tweet - CRITICAL: use sigma tensor directly
        for position in positions_tensor:
            # Calculate squared distances
            dist_sq = ((xx - position[0])**2 + (yy - position[1])**2)
            # Add Gaussian contribution - use differentiable sigma
            gaussian_contrib = torch.exp(-dist_sq / (2 * sigma**2))
            density += gaussian_contrib

        # Normalize to sum to 1
        density_sum = density.sum()
        if density_sum == 0:
            return torch.ones((grid_size, grid_size), device=self.device) / (grid_size ** 2)

        return density / density_sum

    def _fds_loss(self, predicted_density: torch.Tensor, tweet_positions: np.ndarray,
                  grid_bounds: tuple, grid_size: int, tweet_weights: np.ndarray = None) -> torch.Tensor:
        """Calculate FDS-based loss: weighted negative log-likelihood of tweets under predicted density"""
        if len(tweet_positions) == 0:
            # If no tweets in this frame, return small penalty to encourage uniform distribution
            return torch.tensor(0.0, device=self.device)

        (x_min, x_max), (y_min, y_max) = grid_bounds

        # Convert tweet positions to grid coordinates
        x_coords = (tweet_positions[:, 0] - x_min) / (x_max - x_min) * (grid_size - 1)
        y_coords = (tweet_positions[:, 1] - y_min) / (y_max - y_min) * (grid_size - 1)

        # Clamp to grid bounds and convert to integers (use float32 for MPS compatibility)
        x_coords = torch.clamp(torch.tensor(x_coords, dtype=torch.float32, device=self.device), 0, grid_size - 1).long()
        y_coords = torch.clamp(torch.tensor(y_coords, dtype=torch.float32, device=self.device), 0, grid_size - 1).long()

        # Reshape predicted density to 2D grid
        pred_grid = predicted_density.view(grid_size, grid_size)

        # Get predicted probabilities at actual tweet locations
        predicted_probs = pred_grid[y_coords, x_coords]  # Note: [row, col] = [y, x] indexing

        # Calculate weighted negative log-likelihood (FDS-style loss)
        # Add small epsilon for numerical stability
        epsilon = 1e-8
        log_probs = torch.log(predicted_probs + epsilon)

        # Apply tweet importance weights
        if tweet_weights is not None:
            weights_tensor = torch.tensor(tweet_weights, dtype=torch.float32, device=self.device)
            weighted_log_probs = log_probs * weights_tensor
            fds_loss = -torch.sum(weighted_log_probs) / torch.sum(weights_tensor)
        else:
            fds_loss = -torch.mean(log_probs)

        return fds_loss

    def _prepare_time_series_data(self, positions: np.ndarray, times: np.ndarray,
                                 grid_size: int, weights: np.ndarray = None) -> tuple:
        """Convert tweet data into time series of density grids"""
        # Convert to pandas for easier time manipulation
        data_dict = {
            'time': pd.to_datetime(times),
            'positions': [pos for pos in positions]
        }
        if weights is not None:
            data_dict['weights'] = weights

        df = pd.DataFrame(data_dict)
        df = df.sort_values('time')

        # Determine grid bounds from all data
        all_positions = np.array(df['positions'].tolist())
        x_min, x_max = all_positions[:, 0].min(), all_positions[:, 0].max()
        y_min, y_max = all_positions[:, 1].min(), all_positions[:, 1].max()

        # Add padding
        x_padding = max(0.5, (x_max - x_min) * 0.2)
        y_padding = max(0.5, (y_max - y_min) * 0.2)
        bounds = ((x_min - x_padding, x_max + x_padding),
                 (y_min - y_padding, y_max + y_padding))

        # Create time frames
        min_time = df['time'].min()
        max_time = df['time'].max()
        frame_duration = pd.Timedelta(days=self.frame_duration_days)

        time_frames = []
        current_time = min_time
        while current_time < max_time:
            frame_end = current_time + frame_duration
            time_frames.append((current_time, frame_end))
            current_time = frame_end

        # Create density grids for each time frame
        density_grids = []
        frame_positions_list = []  # Store for learnable sigma
        frame_weights_list = []    # Store weights for each frame

        for start_time, end_time in time_frames:
            frame_mask = (df['time'] >= start_time) & (df['time'] < end_time)
            frame_data = df[frame_mask]

            if len(frame_data) > 0:
                frame_positions = np.array(frame_data['positions'].tolist())
                frame_weights = frame_data['weights'].values if 'weights' in frame_data.columns else np.ones(len(frame_positions))
            else:
                frame_positions = np.array([]).reshape(0, 2)
                frame_weights = np.array([])

            frame_positions_list.append(frame_positions)
            frame_weights_list.append(frame_weights)

            density_grid = self._create_density_grid_from_tweets(frame_positions, grid_size, bounds)
            density_grids.append(density_grid.flatten())

        # Store frame data for learnable sigma
        self.frame_tweet_positions = frame_positions_list
        self.frame_tweet_weights = frame_weights_list
        self.frame_bounds = bounds

        return np.array(density_grids), bounds

    def _create_sequences(self, density_grids: np.ndarray) -> tuple:
        """Create input-output sequences for LSTM training"""
        sequences_x = []
        sequences_y = []
        sequence_target_indices = []  # Track which frame index each sequence targets

        for i in range(len(density_grids) - self.sequence_length):
            # Input: sequence_length frames
            x = density_grids[i:i + self.sequence_length]
            # Output: next frame
            y = density_grids[i + self.sequence_length]
            target_frame_idx = i + self.sequence_length

            sequences_x.append(x)
            sequences_y.append(y)
            sequence_target_indices.append(target_frame_idx)

        self.frame_sequence_indices = sequence_target_indices
        return np.array(sequences_x), np.array(sequences_y)

    def fit(self, train_data: np.ndarray, train_times: np.ndarray, train_weights: np.ndarray = None, grid_size: int = 50) -> None:
        """Train the LSTM model on historical density sequences"""
        self.grid_size = grid_size
        self.train_positions = train_data
        self.train_times = train_times
        self.train_weights = train_weights  # Store weights for potential use in loss function

        if len(train_data) == 0:
            print("Warning: No training data provided")
            return

        print(f"Preparing time series data with {len(train_data)} tweets...")

        # Convert tweet data to time series of density grids
        density_grids, self.bounds = self._prepare_time_series_data(
            train_data, train_times, grid_size, train_weights
        )

        if len(density_grids) <= self.sequence_length:
            print(f"Warning: Not enough time frames ({len(density_grids)}) for sequence length ({self.sequence_length})")
            return

        # Create sequences for training
        sequences_x, _ = self._create_sequences(density_grids)  # Only need input sequences for FDS loss

        if len(sequences_x) == 0:
            print("Warning: No sequences created for training")
            return

        print(f"Created {len(sequences_x)} training sequences")

        # Analyze temporal signal in the data before assuming LSTM should learn patterns
        self._analyze_temporal_signal()

        # Scale input sequences only
        flat_x = sequences_x.reshape(-1, sequences_x.shape[-1])
        self.scaler.fit(flat_x)
        sequences_x_scaled = self.scaler.transform(flat_x).reshape(sequences_x.shape)

        # Convert to PyTorch tensors
        sequences_x = torch.FloatTensor(sequences_x_scaled).to(self.device)

        # Initialize model
        input_size = grid_size * grid_size
        self.model = LSTMDensityPredictor(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            learn_sigma=self.learn_sigma,
            initial_sigma=self.gaussian_sigma
        ).to(self.device)

        # Training setup with separate learning rates for sigma and LSTM weights
        if self.learn_sigma:
            # Higher learning rate for sigma, normal for LSTM weights
            lstm_params = [p for name, p in self.model.named_parameters() if 'log_sigma' not in name]
            sigma_params = [p for name, p in self.model.named_parameters() if 'log_sigma' in name]

            optimizer = optim.Adam([
                {'params': lstm_params, 'lr': self.learning_rate},
                {'params': sigma_params, 'lr': self.learning_rate * 10}  # Higher LR for sigma
            ])
        else:
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)


        print(f"Training LSTM model for {self.epochs} epochs with FDS loss...")
        print(f"Device: {self.device}")
        print(f"Initial sigma: {self.model.get_sigma():.4f}")
        self.model.train()

        # Always use FDS loss - optimizes directly for tweet prediction
        self._train_with_fds_loss(sequences_x, optimizer)

        if self.learn_sigma:
            print(f"Final sigma: {self.model.get_sigma():.4f}")

        self.model.eval()
        print("LSTM training completed")

    def _train_with_fds_loss(self, initial_sequences_x, optimizer):
        """Training loop with FDS loss - directly optimizes for tweet prediction"""
        for epoch in range(self.epochs):
            total_loss = 0
            num_batches = 0

            # Create batches from sequences
            num_sequences = len(self.frame_sequence_indices)
            indices = torch.randperm(num_sequences)

            for start_idx in range(0, num_sequences, self.batch_size):
                end_idx = min(start_idx + self.batch_size, num_sequences)
                batch_indices = indices[start_idx:end_idx]

                batch_loss = 0
                valid_sequences = 0

                optimizer.zero_grad()

                for seq_idx in batch_indices:
                    seq_idx = seq_idx.item()
                    target_frame_idx = self.frame_sequence_indices[seq_idx]
                    target_tweet_positions = self.frame_tweet_positions[target_frame_idx]

                    # Skip sequences with no target tweets
                    if len(target_tweet_positions) == 0:
                        continue

                    # Generate input sequence on-the-fly with current sigma (preserves gradients!)
                    if self.learn_sigma:
                        current_sigma = torch.exp(self.model.log_sigma)

                        # Create input sequence frames with differentiable sigma
                        input_frames = []
                        for i in range(self.sequence_length):
                            frame_idx = target_frame_idx - self.sequence_length + 1 + i
                            if 0 <= frame_idx < len(self.frame_tweet_positions):
                                frame_positions = self.frame_tweet_positions[frame_idx]
                                # CRITICAL FIX: Use current_sigma tensor directly, don't call .item()
                                frame_grid = self._create_density_grid_torch_differentiable(
                                    frame_positions, self.grid_size, self.frame_bounds, current_sigma
                                )
                                input_frames.append(frame_grid.flatten())
                            else:
                                # Use uniform distribution for missing frames
                                uniform_grid = torch.ones(self.grid_size * self.grid_size, device=self.device) / (self.grid_size ** 2)
                                input_frames.append(uniform_grid)

                        # Stack frames into sequence
                        input_sequence_raw = torch.stack(input_frames).unsqueeze(0)  # Add batch dim
                        input_seq = input_sequence_raw
                    else:
                        # Use pre-computed sequences for fixed sigma
                        input_seq = initial_sequences_x[seq_idx:seq_idx+1]

                    # Forward pass
                    predicted_density = self.model(input_seq)[0]  # Remove batch dimension

                    # Calculate FDS loss with tweet weights
                    target_tweet_weights = self.frame_tweet_weights[target_frame_idx] if self.frame_tweet_weights else None
                    fds_loss = self._fds_loss(predicted_density, target_tweet_positions,
                                            self.frame_bounds, self.grid_size, target_tweet_weights)

                    batch_loss += fds_loss

                    valid_sequences += 1

                # Backward pass for batch
                if valid_sequences > 0:
                    avg_batch_loss = batch_loss / valid_sequences
                    avg_batch_loss.backward()

                    # Check gradients on first epoch for debugging
                    if epoch == 0 and start_idx == 0 and self.learn_sigma:
                        sigma_grad = self.model.log_sigma.grad
                        if sigma_grad is not None:
                            print(f"  Sigma gradient magnitude: {sigma_grad.abs().item():.6f}")
                        else:
                            print("  WARNING: No gradient for sigma!")

                    # Gradient clipping for stability
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                    optimizer.step()

                    total_loss += avg_batch_loss.item()
                    num_batches += 1


            if (epoch + 1) % 10 == 0 and num_batches > 0:
                avg_loss = total_loss / num_batches
                current_sigma = self.model.get_sigma() if self.learn_sigma else self.gaussian_sigma
                print(f"Epoch [{epoch+1}/{self.epochs}], FDS Loss: {avg_loss:.6f}, Sigma: {current_sigma:.4f}")

                # Debug: Check if predictions are diverse
                if epoch == 9:  # After 10 epochs
                    with torch.no_grad():
                        test_seq = initial_sequences_x[0:1] if not self.learn_sigma else None
                        if self.learn_sigma and len(self.frame_sequence_indices) > 0:
                            # Create a test sequence for debugging
                            test_target_frame = self.frame_sequence_indices[0]
                            test_frames = []
                            for i in range(self.sequence_length):
                                frame_idx = test_target_frame - self.sequence_length + 1 + i
                                if 0 <= frame_idx < len(self.frame_tweet_positions):
                                    frame_positions = self.frame_tweet_positions[frame_idx]
                                    frame_grid = self._create_density_grid_torch_differentiable(
                                        frame_positions, self.grid_size, self.frame_bounds, torch.exp(self.model.log_sigma)
                                    )
                                    test_frames.append(frame_grid.flatten())
                                else:
                                    uniform_grid = torch.ones(self.grid_size * self.grid_size, device=self.device) / (self.grid_size ** 2)
                                    test_frames.append(uniform_grid)
                            test_seq = torch.stack(test_frames).unsqueeze(0)

                        if test_seq is not None:
                            pred = self.model(test_seq)[0].view(self.grid_size, self.grid_size)
                            pred_std = torch.std(pred).item()
                            pred_max = torch.max(pred).item()
                            pred_min = torch.min(pred).item()
                            print(f"  Prediction diversity - Std: {pred_std:.6f}, Range: {pred_min:.6f} to {pred_max:.6f}")

                            if pred_std < 1e-6:
                                print("  WARNING: Model predicting nearly uniform distribution!")

                            # Check if model is ignoring temporal patterns
                            # Generate prediction with different input sequences
                            if len(self.frame_sequence_indices) > 1:
                                test_seq2 = initial_sequences_x[1:2] if not self.learn_sigma else None
                                if self.learn_sigma:
                                    test_target_frame2 = self.frame_sequence_indices[1]
                                    test_frames2 = []
                                    for i in range(self.sequence_length):
                                        frame_idx = test_target_frame2 - self.sequence_length + 1 + i
                                        if 0 <= frame_idx < len(self.frame_tweet_positions):
                                            frame_positions = self.frame_tweet_positions[frame_idx]
                                            frame_grid = self._create_density_grid_torch_differentiable(
                                                frame_positions, self.grid_size, self.frame_bounds, torch.exp(self.model.log_sigma)
                                            )
                                            test_frames2.append(frame_grid.flatten())
                                        else:
                                            uniform_grid = torch.ones(self.grid_size * self.grid_size, device=self.device) / (self.grid_size ** 2)
                                            test_frames2.append(uniform_grid)
                                    test_seq2 = torch.stack(test_frames2).unsqueeze(0)

                                if test_seq2 is not None:
                                    pred2 = self.model(test_seq2)[0].view(self.grid_size, self.grid_size)
                                    pred_diff = torch.mean(torch.abs(pred - pred2)).item()
                                    print(f"  Prediction difference between sequences: {pred_diff:.6f}")
                                    if pred_diff < 1e-6:
                                        print("  WARNING: Model produces identical predictions for different inputs!")
                                        print("  This suggests the model is ignoring temporal patterns.")

    def _analyze_temporal_signal(self):
        """Analyze if there's actually temporal signal worth learning"""
        if not self.frame_tweet_positions:
            return

        print("Analyzing temporal signal in the data:")

        # Check if consecutive frames are actually different
        frame_similarities = []
        non_empty_frames = [f for f in self.frame_tweet_positions if len(f) > 0]

        if len(non_empty_frames) < 2:
            print("  WARNING: Too few frames with tweets for temporal analysis")
            return

        # Sample a few frame pairs to check similarity
        frame_similarities = []
        for i in range(min(10, len(non_empty_frames) - 1)):
            frame1 = non_empty_frames[i]
            frame2 = non_empty_frames[i + 1]

            # Create density grids for comparison
            grid1 = self._create_density_grid_from_tweets(frame1, self.grid_size, self.frame_bounds)
            grid2 = self._create_density_grid_from_tweets(frame2, self.grid_size, self.frame_bounds)

            # Calculate similarity (lower = more different)
            similarity = np.mean(np.abs(grid1 - grid2))
            frame_similarities.append(similarity)

        avg_similarity = np.mean(frame_similarities)
        print(f"  Average frame-to-frame difference: {avg_similarity:.6f}")

        # Check temporal autocorrelation
        tweet_counts = [len(f) for f in self.frame_tweet_positions]
        if len(tweet_counts) > 1:
            # Simple lag-1 correlation
            count_corr = np.corrcoef(tweet_counts[:-1], tweet_counts[1:])[0, 1]
            print(f"  Tweet count autocorrelation (lag-1): {count_corr:.3f}")

            if abs(count_corr) < 0.1:
                print("  INFO: Low temporal correlation - LSTM may correctly ignore time patterns")
            elif count_corr > 0.3:
                print("  INFO: Strong positive correlation - temporal patterns exist")
            elif count_corr < -0.3:
                print("  INFO: Strong negative correlation - alternating patterns exist")

        # Additional analysis: Check tweet position stability vs randomness
        if len(non_empty_frames) >= 3:
            position_movements = []
            for i in range(min(5, len(non_empty_frames) - 1)):
                frame1 = non_empty_frames[i]
                frame2 = non_empty_frames[i + 1]

                if len(frame1) > 0 and len(frame2) > 0:
                    # Calculate center of mass for each frame
                    center1 = np.mean(frame1, axis=0)
                    center2 = np.mean(frame2, axis=0)
                    movement = np.linalg.norm(center2 - center1)
                    position_movements.append(movement)

            if position_movements:
                avg_movement = np.mean(position_movements)
                print(f"  Average center-of-mass movement between frames: {avg_movement:.6f}")


    def update_state(self, current_data: np.ndarray, current_times: np.ndarray) -> None:
        """Update model state with new data - LSTM uses all historical data"""
        # For this implementation, we don't update state during prediction
        # The model is trained once and makes predictions based on the learned patterns
        pass

    def predict_density(self, test_times: np.ndarray, grid_size: int = 100) -> np.ndarray:
        """Predict density grids for future time periods"""
        if self.model is None or self.train_positions is None:
            print("Model not trained, returning uniform distribution")
            uniform_density = np.ones((grid_size, grid_size)) / (grid_size ** 2)
            return np.array([uniform_density] * len(test_times))

        # Create recent history sequence for prediction
        density_grids, _ = self._prepare_time_series_data(
            self.train_positions, self.train_times, grid_size, self.train_weights
        )

        if len(density_grids) < self.sequence_length:
            print("Not enough historical data for prediction, returning uniform distribution")
            uniform_density = np.ones((grid_size, grid_size)) / (grid_size ** 2)
            return np.array([uniform_density] * len(test_times))

        # Use the most recent sequence for prediction
        recent_sequence = density_grids[-self.sequence_length:]
        recent_sequence_scaled = self.scaler.transform(recent_sequence)

        # Convert to tensor and add batch dimension
        input_tensor = torch.FloatTensor(recent_sequence_scaled).unsqueeze(0).to(self.device)

        predictions = []

        with torch.no_grad():
            for _ in range(len(test_times)):
                # Predict next frame
                prediction_scaled = self.model(input_tensor)

                # Inverse transform to original scale
                prediction = self.scaler.inverse_transform(prediction_scaled.cpu().numpy())

                # Reshape to grid and ensure valid probability distribution
                pred_grid = prediction.reshape(grid_size, grid_size)
                pred_grid = np.maximum(pred_grid, 0)  # Ensure non-negative
                pred_grid = pred_grid / pred_grid.sum()  # Normalize

                predictions.append(pred_grid)

                # Update input sequence for next prediction (sliding window)
                # Add prediction to sequence and remove oldest frame
                next_input = torch.cat([
                    input_tensor[:, 1:, :],  # Remove first frame
                    prediction_scaled.unsqueeze(1)  # Add prediction as last frame
                ], dim=1)
                input_tensor = next_input

        return np.array(predictions)

    def get_name(self) -> str:
        if self.model is not None and self.learn_sigma:
            learned_sigma = self.model.get_sigma()
            return f"LSTM Predictor (seq={self.sequence_length}, hidden={self.hidden_size}, σ_learned={learned_sigma:.4f})"
        else:
            return f"LSTM Predictor (seq={self.sequence_length}, hidden={self.hidden_size}, σ={self.gaussian_sigma})"