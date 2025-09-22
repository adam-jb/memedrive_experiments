#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from testing.framework import TestingFramework
from models.baseline import HistoricalAverageModel, RandomModel, GaussianSmoothedHistoricalModel
from models.drift_field import DriftFieldModel

def main():
    # Path to tweet data - use the larger 3.4M tweet dataset
    csv_path = '~/Desktop/memedrive_experiments/output_data/community_archive_good_faith_embeddings.csv'

    # Target topic: used to label experiment result files
    target_topic = 'good_faith'

    # Use larger sample to get more test tweets
    sample_size = 10_000  # Set to None to have all

    # Date window parameters (None = use all data). YYYY-mm-dd
    start_date = '2024-01-01'  # e.g., '2024-01-01'
    end_date =  '2024-12-31'    # e.g., '2024-12-31'

    # Models to animate (set to empty list to disable)
    animate_models = ['DriftField'] #['Gaussian Smoothed Historical']

    # Grid resolution (e.g., 100 for 100x100 grid)
    grid_size = 100

    # Frame duration in days (can be decimal, e.g., 0.5 for 12 hours)
    frame_duration_days = 1.0  # Default is 7 days (1 week)

    print("Initializing testing framework...")
    framework = TestingFramework(csv_path, sample_size=sample_size,
                                start_date=start_date, end_date=end_date,
                                animate_models=animate_models,
                                grid_size=grid_size,
                                target_topic=target_topic,
                                frame_duration_days=frame_duration_days)

    # Random model has FDS of 1
    framework.add_model(RandomModel())

    # Our baseline model
    framework.add_model(HistoricalAverageModel(bandwidth=0.1))

    # Modelling we're hoping beats baseline
    framework.add_model(DriftFieldModel())


    results = framework.run_evaluation(test_weeks=10)

    # Print results
    framework.print_results(results)

if __name__ == "__main__":
    main()