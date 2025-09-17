#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from testing.framework import TestingFramework
from models.baseline import HistoricalAverageModel, RandomModel, WeeklyAverageModel

def main():
    # Path to tweet data
    csv_path = '~/Desktop/memedrive_experiments/output_data/tweet_good_faith_ratings.csv'

    # Use larger sample to get more test tweets
    sample_size = 10_000

    print("Initializing testing framework...")
    framework = TestingFramework(csv_path, sample_size=sample_size)

    # Add baseline models
    framework.add_model(RandomModel())
    framework.add_model(HistoricalAverageModel(bandwidth=0.1))
    framework.add_model(WeeklyAverageModel(bandwidth=0.2))

    results = framework.run_evaluation(test_weeks=50)

    # Print results
    framework.print_results(results)

if __name__ == "__main__":
    main()