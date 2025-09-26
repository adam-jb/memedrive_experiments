
import pandas as pd
import numpy as np
from openai import OpenAI
from datetime import datetime
import json
from pathlib import Path
import time
from tqdm import tqdm
import logging
import os
import argparse
import dotenv
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading

dotenv.load_dotenv()

# Get environment variables for different models
openai_key = os.getenv("OPENAI", 'missing')
claude_key = os.getenv("ANTHROPIC_API_KEY", 'missing')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

# Define lens configurations
LENS_CONFIGS = {
    'good_faith': {
        'dimensions': ['sincerity', 'charity'],
        'scale': '1-7',
        'default_value': 4,
        'prompt_template': """
You are a careful rater of social media discourse quality

Rate the below tweet on 3 dimensions (1-7 scale):

1. SINCERITY: Authentic belief vs trolling/performing
   - Low (1-3): Trolling, bait, saying things for effect, bad faith questions
   - High (6-7): Genuine beliefs, real curiosity, authentic expression

2. CHARITY: How they interpret others
   - Low (1-3): Assumes worst, strawmans, takes least charitable reading
   - High (6-7): Steelmans arguments, assumes good intentions, generous interpretation

We're only interested in sincerity and charity. Where neutral say 4. Eg other things like humour not relevant.

Tweet: "{tweet_text}"

Return ONLY two numbers separated by commas: sincerity,charity
Eg 1,6
Or 4,3
""",
        'composite_formula': lambda dims: dims.mean(axis=1)
    },
    'excitement_directedness': {
        'dimensions': ['excitement', 'directedness'],
        'scale': '0-10',
        'default_value': 5,
        'prompt_template': """
You are a careful rater of social media discourse characteristics

Rate the below tweet on 2 dimensions (0-10 scale):

1. EXCITEMENT: Energy, enthusiasm, emotional intensity, passion level
   - Low (0-3): Calm, measured, low energy, dispassionate
   - Medium (4-6): Moderate energy, some enthusiasm
   - High (7-10): Very excited, passionate, high energy, intense emotion

2. DIRECTEDNESS: Focus orientation
   - Inward (0-3): Self-focused, internal community, personal concerns, introspective
   - Mixed (4-6): Balance of internal and external focus
   - Outward (7-10): External world focused, public issues, broad audience

Default to 5 for both dimensions when unclear or ambiguous.

Tweet: "{tweet_text}"

Return ONLY two numbers separated by commas: excitement,directedness
Eg 3,8
Or 7,2
Or 5,5
""",
        'composite_formula': lambda dims: dims.mean(axis=1)
    }
}

def rate_tweet_with_openai(tweet_text, lens_config, client):
    """Rate tweet using OpenAI with specified lens configuration"""
    prompt = lens_config['prompt_template'].format(tweet_text=tweet_text)
    expected_count = len(lens_config['dimensions'])
    default_value = lens_config['default_value']

    try:
        response = client.responses.create(
            model="gpt-5-mini",
            input=prompt,
            reasoning={"effort": "minimal"},
            text={"verbosity": "low"},
        )

        rating_text = response.output[1].content[0].text.strip()

        # Try to extract numbers from the response, handling various formats
        try:
            # First try direct comma split
            parts = rating_text.split(',')
            ratings = []
            for part in parts:
                # Extract first number found in each part
                import re
                numbers = re.findall(r'\b\d+(?:\.\d+)?\b', part.strip())
                if numbers:
                    ratings.append(float(numbers[0]))

            # If we didn't get enough ratings, pad with defaults
            while len(ratings) < expected_count:
                ratings.append(default_value)

            # Take only what we need
            ratings = ratings[:expected_count]

            if len(ratings) != expected_count:
                raise ValueError(f"Could not extract {expected_count} ratings")

            return ratings

        except Exception as parse_error:
            # If parsing fails completely, return neutral values
            logging.warning(f"Could not parse OpenAI response '{rating_text}', using defaults: {parse_error}")
            return [default_value] * expected_count

    except Exception as e:
        logging.warning(f"OpenAI API error, using default values: {e}")
        # Return default values instead of None to avoid downstream errors
        default_value = lens_config.get('default_value', 4)
        return [default_value] * expected_count


def rate_tweet_with_claude(tweet_text, lens_config):
    """Rate tweet using Claude with specified lens configuration"""
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=claude_key)

        prompt = lens_config['prompt_template'].format(tweet_text=tweet_text)
        expected_count = len(lens_config['dimensions'])
        default_value = lens_config['default_value']

        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=50,
            temperature=0.1,
            messages=[{"role": "user", "content": prompt}]
        )

        rating_text = message.content[0].text.strip()

        # Try to extract numbers from the response, handling various formats
        try:
            # First try direct comma split
            parts = rating_text.split(',')
            ratings = []
            for part in parts:
                # Extract first number found in each part
                import re
                numbers = re.findall(r'\b\d+(?:\.\d+)?\b', part.strip())
                if numbers:
                    ratings.append(float(numbers[0]))

            # If we didn't get enough ratings, pad with defaults
            while len(ratings) < expected_count:
                ratings.append(default_value)

            # Take only what we need
            ratings = ratings[:expected_count]

            if len(ratings) != expected_count:
                raise ValueError(f"Could not extract {expected_count} ratings")

            return ratings

        except Exception as parse_error:
            # If parsing fails completely, return neutral values
            logging.warning(f"Could not parse Claude response '{rating_text}', using defaults: {parse_error}")
            return [default_value] * expected_count

    except Exception as e:
        logging.warning(f"Claude API error, using default values: {e}")
        # Return default values instead of None to avoid downstream errors
        default_value = lens_config.get('default_value', 4)
        return [default_value] * expected_count


def rate_tweet(tweet_text, lens_config, model_type='openai', openai_client=None):
    """Rate tweet using specified model and lens configuration"""
    if model_type == 'openai':
        if openai_client is None:
            raise ValueError("OpenAI client required for OpenAI model")
        return rate_tweet_with_openai(tweet_text, lens_config, openai_client)
    elif model_type == 'claude':
        return rate_tweet_with_claude(tweet_text, lens_config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# Legacy function for backwards compatibility
def rate_3d_comprehensive(tweet_text, client):
    """Legacy function - use rate_tweet instead"""
    return rate_tweet(tweet_text, LENS_CONFIGS['good_faith'], 'openai', client)


def rate_tweet_with_index(tweet_data):
    """Wrapper to rate tweet while preserving index for concurrent processing"""
    idx, tweet_text, lens_config, model_type, openai_client = tweet_data

    # Skip if tweet is too short or None
    dimensions = lens_config['dimensions']
    if pd.isna(tweet_text) or len(str(tweet_text).strip()) < 10:
        result = {'index': idx, 'error': 'Tweet too short or empty'}
        for dim in dimensions:
            result[dim] = None
        return result

    # Rate the tweet
    ratings = rate_tweet(tweet_text, lens_config, model_type, openai_client)

    result = {'index': idx}
    for i, dim in enumerate(dimensions):
        result[dim] = ratings[i]
    result['error'] = None if ratings[0] is not None else 'Rating failed'

    return result


def process_tweets_concurrent(tweets_df, lens_config, model_type='openai', openai_client=None,
                             max_workers=5, delay_between_requests=0.1):
    """
    Process tweets with concurrent requests while maintaining order
    """
    results = []
    dimensions = lens_config['dimensions']

    # Prepare all tweet data for processing
    tweet_data_list = []
    for idx, row in tweets_df.iterrows():
        tweet_data = (idx, row['full_text'], lens_config, model_type, openai_client)
        tweet_data_list.append(tweet_data)

    print(f"Processing {len(tweet_data_list)} tweets with {max_workers} concurrent workers...")

    # Process in batches to maintain some rate limiting
    batch_size = max_workers * 10  # Process in larger chunks

    for batch_start in tqdm(range(0, len(tweet_data_list), batch_size), desc="Processing batches"):
        batch_end = min(batch_start + batch_size, len(tweet_data_list))
        batch_data = tweet_data_list[batch_start:batch_end]

        # Process batch concurrently
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_index = {executor.submit(rate_tweet_with_index, data): i
                              for i, data in enumerate(batch_data)}

            batch_results = [None] * len(batch_data)

            # Collect results in order
            for future in future_to_index:
                batch_index = future_to_index[future]
                try:
                    result = future.result()
                    batch_results[batch_index] = result
                except Exception as e:
                    logging.error(f"Error processing tweet: {e}")
                    # Create error result
                    idx = batch_data[batch_index][0]
                    error_result = {'index': idx, 'error': f'Processing failed: {str(e)}'}
                    for dim in dimensions:
                        error_result[dim] = None
                    batch_results[batch_index] = error_result

        results.extend(batch_results)

        # Rate limiting between batches
        if batch_end < len(tweet_data_list):
            time.sleep(delay_between_requests * max_workers)

        # Save intermediate results every 1000 tweets
        if len(results) % 1000 == 0:
            intermediate_df = pd.DataFrame(results)
            lens_name = '_'.join(dimensions)
            intermediate_df.to_csv(
                f'~/Desktop/memedrive_experiments/output_data/intermediate_ratings_{lens_name}_{len(results)}.csv',
                index=False
            )
            logging.info(f"Saved intermediate results at {len(results)} tweets")

    return pd.DataFrame(results)


def process_tweets_batch(tweets_df, lens_config, model_type='openai', openai_client=None, batch_size=100):
    """
    Process tweets in batches with rate limiting
    """
    results = []
    dimensions = lens_config['dimensions']
    default_value = lens_config['default_value']

    for i in tqdm(range(0, len(tweets_df), batch_size), desc="Processing batches"):
        batch = tweets_df.iloc[i:i+batch_size]

        for idx, row in batch.iterrows():
            tweet_text = row['full_text']

            # Skip if tweet is too short or None
            if pd.isna(tweet_text) or len(str(tweet_text).strip()) < 10:
                result = {'index': idx, 'error': 'Tweet too short or empty'}
                for dim in dimensions:
                    result[dim] = None
                results.append(result)
                continue

            # Rate the tweet
            ratings = rate_tweet(tweet_text, lens_config, model_type, openai_client)

            result = {'index': idx}
            for i, dim in enumerate(dimensions):
                result[dim] = ratings[i]
            result['error'] = None if ratings[0] is not None else 'Rating failed'
            results.append(result)

            # Rate limiting
            sleep_time = 0.5 if model_type == 'openai' else 1.0  # Claude needs more delay
            time.sleep(sleep_time)

        # Save intermediate results every 1000 tweets
        if (i + batch_size) % 1000 == 0:
            intermediate_df = pd.DataFrame(results)
            lens_name = '_'.join(dimensions)
            intermediate_df.to_csv(
                f'~/Desktop/memedrive_experiments/output_data/intermediate_ratings_{lens_name}_{i+batch_size}.csv',
                index=False
            )
            logging.info(f"Saved intermediate results at {i+batch_size} tweets")

    return pd.DataFrame(results)


def main():
    """Main execution function with argument parsing"""
    parser = argparse.ArgumentParser(description='Rate tweets using different lens configurations')
    parser.add_argument('--lens_type', choices=list(LENS_CONFIGS.keys()),
                       default='good_faith', help='Lens configuration to use')
    parser.add_argument('--model_type', choices=['openai', 'claude'],
                       default='openai', help='Model to use for rating')
    parser.add_argument('--sample_size', type=int, default=25_000,
                       help='Number of tweets to sample and rate')
    parser.add_argument('--concurrent', action='store_true',
                       help='Use concurrent processing for faster requests')
    parser.add_argument('--max_workers', type=int, default=5,
                       help='Maximum number of concurrent workers (default: 5)')

    args = parser.parse_args()

    # Get lens configuration
    lens_config = LENS_CONFIGS[args.lens_type]
    dimensions = lens_config['dimensions']
    lens_name = args.lens_type

    # Set up model client
    openai_client = None
    if args.model_type == 'openai':
        if openai_key == 'missing':
            raise ValueError('OpenAI API key missing - set OPENAI environment variable')
        openai_client = OpenAI(api_key=openai_key)
    elif args.model_type == 'claude':
        if claude_key == 'missing':
            raise ValueError('Claude API key missing - set ANTHROPIC_API_KEY environment variable')

    # Set paths
    input_path = Path('~/Desktop/memedrive_experiments/input_data/community_archive.parquet').expanduser()
    output_dir = Path('~/Desktop/memedrive_experiments/output_data').expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate timestamp for file naming
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    logging.info(f"Starting tweet rating process at {timestamp}")
    logging.info(f"Using lens: {lens_name}, model: {args.model_type}")

    # Load the data
    logging.info(f"Loading data from {input_path}")
    df = pd.read_parquet(input_path)
    logging.info(f"Loaded {len(df)} total tweets")

    # Sample N random tweets
    sample_size = min(args.sample_size, len(df))
    sample_df = df.sample(n=sample_size, random_state=42)
    logging.info(f"Sampled {sample_size} tweets")

    # Log the row IDs (indices) of chosen tweets
    chosen_indices = sample_df.index.tolist()
    indices_file = output_dir / f'chosen_tweet_indices_{lens_name}_{timestamp}.json'
    with open(indices_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'lens_type': lens_name,
            'model_type': args.model_type,
            'sample_size': sample_size,
            'indices': chosen_indices
        }, f, indent=2)
    logging.info(f"Saved chosen tweet indices to {indices_file}")

    # Process the tweets
    if args.concurrent:
        logging.info(f"Starting concurrent rating process with {args.max_workers} workers...")
        ratings_df = process_tweets_concurrent(sample_df, lens_config, args.model_type, openai_client,
                                             max_workers=args.max_workers)
    else:
        logging.info("Starting sequential rating process...")
        ratings_df = process_tweets_batch(sample_df, lens_config, args.model_type, openai_client)

    # Merge ratings with original tweet data
    final_df = sample_df.reset_index(drop=False)
    final_df = final_df.rename(columns={'index': 'original_index'})
    final_df = pd.concat([final_df, ratings_df], axis=1)

    # Calculate composite score using lens-specific formula
    composite_score = lens_config['composite_formula'](final_df[dimensions])
    final_df[f'{lens_name}_score'] = composite_score

    # Save CSV with lens-specific naming
    csv_file = output_dir / f'tweet_{lens_name}_ratings.csv'
    final_df.to_csv(csv_file, index=False)
    logging.info(f"Saved CSV version to {csv_file}")

    # Generate summary statistics
    mean_scores = {dim: final_df[dim].mean() for dim in dimensions}
    mean_scores[f'{lens_name}_overall'] = final_df[f'{lens_name}_score'].mean()

    score_distributions = {dim: final_df[dim].value_counts().sort_index().to_dict()
                          for dim in dimensions}

    summary_stats = {
        'timestamp': timestamp,
        'lens_type': lens_name,
        'model_type': args.model_type,
        'total_tweets_rated': len(final_df),
        'successful_ratings': final_df[dimensions[0]].notna().sum(),
        'failed_ratings': final_df[dimensions[0]].isna().sum(),
        'mean_scores': mean_scores,
        'score_distributions': score_distributions
    }

    summary_file = output_dir / f'rating_summary_{lens_name}_{timestamp}.json'
    with open(summary_file, 'w') as f:
        json.dump(summary_stats, f, indent=2, default=str)
    logging.info(f"Saved summary statistics to {summary_file}")

    # Print summary
    print("\n" + "="*50)
    print("RATING COMPLETE")
    print("="*50)
    print(f"Lens: {lens_name}, Model: {args.model_type}")
    print(f"Total tweets rated: {summary_stats['total_tweets_rated']}")
    print(f"Successful ratings: {summary_stats['successful_ratings']}")
    print(f"Failed ratings: {summary_stats['failed_ratings']}")
    print(f"\nMean Scores:")
    for dim, score in summary_stats['mean_scores'].items():
        if score:
            print(f"  {dim}: {score:.2f}")

    return final_df

if __name__ == "__main__":
    # Run the main process
    results = main()
