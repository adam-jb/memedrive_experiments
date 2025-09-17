#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from testing.framework import TestingFramework
from models.baseline import HistoricalAverageModel, RandomModel, WeeklyAverageModel, GaussianSmoothedHistoricalModel

def main():
    # Path to tweet data - use the larger 3.4M tweet dataset
    csv_path = '~/Desktop/memedrive_experiments/output_data/community_archive_good_faith_embeddings.csv'

    # Use larger sample to get more test tweets
    sample_size = 10_000  # Set to None to have all

    # Date window parameters (None = use all data). YYYY-mm-dd
    start_date = '2023-01-01'  # e.g., '2024-01-01'
    end_date =  '2024-12-31'    # e.g., '2024-12-31'

    # Forgiveness parameter - how many grid cells of tolerance to allow
    # 0 = harsh (exact match required), 3+ = forgiving (default generous)
    tolerance_radius = 4  # Very forgiving - predictions within 4 cells get credit

    print("Initializing testing framework...")
    framework = TestingFramework(csv_path, sample_size=sample_size,
                                start_date=start_date, end_date=end_date,
                                tolerance_radius=tolerance_radius)

    # Add baseline models
    framework.add_model(RandomModel())
    framework.add_model(HistoricalAverageModel(bandwidth=0.1))
    framework.add_model(GaussianSmoothedHistoricalModel(gaussian_bandwidth=0.05))
    #framework.add_model(WeeklyAverageModel(bandwidth=0.2))

    results = framework.run_evaluation(test_weeks=10)

    # Print results
    framework.print_results(results)

if __name__ == "__main__":
    main()