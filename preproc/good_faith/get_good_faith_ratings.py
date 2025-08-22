
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

# Get an environment variable
openai_key = os.getenv("OPENAI", 'missing')
if openai_key == 'missing':
    raise ValueError('openai key missing')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)


client = OpenAI(api_key = openai_key)

def rate_3d_comprehensive(tweet_text, client):
    """
    Rate tweet on Sincerity × Charity × Constructiveness
    """

    prompt = f"""
    You are a careful rater of social media discourse quality

    Rate the below tweet on 3 dimensions (1-7 scale):

    1. SINCERITY: Authentic belief vs trolling/performing
       - Low (1-3): Trolling, bait, saying things for effect, bad faith questions
       - High (6-7): Genuine beliefs, real curiosity, authentic expression

    2. CHARITY: How they interpret others
       - Low (1-3): Assumes worst, strawmans, takes least charitable reading
       - High (6-7): Steelmans arguments, assumes good intentions, generous interpretation

    3. CONSTRUCTIVENESS: Impact on discourse
       - Low (1-3): Tears down, increases heat, derails, pure criticism
       - High (6-7): Builds understanding, offers solutions, advances conversation

    Tweet: "{tweet_text}"

    Return ONLY three numbers separated by commas: sincerity,charity,constructiveness
    Eg 1,6,3
    Or 4,3,3
    """

    try:
        # response = client.chat.completions.create(
        #     model="gpt-5-chat-latest",
        #     messages=[
        #         {"role": "system", "content": "You are a careful rater of social media discourse quality."},
        #         {"role": "user", "content": prompt}
        #     ],
        #     temperature=0.05,  # Lower temperature for more consistent ratings
        #     max_completion_tokens=20
        # )
        response = client.responses.create(
            model="gpt-5-mini",
            input=prompt,
            reasoning={ "effort": "minimal" }, # can be high,medium,low,minimal
            text={ "verbosity": "low" }, # can be high,medium,low
        )

        # ### Is the returned text really this level of embedding I cant believe I just canno
        print('response output', response.output[1].content[0].text)
        rating_text = response.output[1].content[0].text.strip()

        # Parse the response
        ratings = [float(x.strip()) for x in rating_text.split(',')]

        if len(ratings) != 3:
            raise ValueError(f"Expected 3 ratings, got {len(ratings)}")

        # Validate ratings are in range
        for r in ratings:
            if r < 1 or r > 7:
                raise ValueError(f"Rating {r} out of range [1,7]")

        return ratings

    except Exception as e:
        logging.error(f"Error rating tweet: {e}")
        logging.error(f"Tweet text: {tweet_text[:100]}...")
        return [None, None, None]


def process_tweets_batch(tweets_df, client, batch_size=100):
    """
    Process tweets in batches with rate limiting
    """
    results = []

    for i in tqdm(range(0, len(tweets_df), batch_size), desc="Processing batches"):
        batch = tweets_df.iloc[i:i+batch_size]

        for idx, row in batch.iterrows():
            tweet_text = row['full_text']

            # Skip if tweet is too short or None
            if pd.isna(tweet_text) or len(str(tweet_text).strip()) < 10:
                results.append({
                    'index': idx,
                    'sincerity': None,
                    'charity': None,
                    'constructiveness': None,
                    'error': 'Tweet too short or empty'
                })
                continue

            # Rate the tweet
            ratings = rate_3d_comprehensive(tweet_text, client)

            results.append({
                'index': idx,
                'sincerity': ratings[0],
                'charity': ratings[1],
                'constructiveness': ratings[2],
                'error': None if ratings[0] is not None else 'Rating failed'
            })

            # Rate limiting (adjust as needed for GPT-5)
            time.sleep(0.5)  # 10 requests per second max

        # Save intermediate results every 1000 tweets
        if (i + batch_size) % 1000 == 0:
            intermediate_df = pd.DataFrame(results)
            intermediate_df.to_csv(
                f'~/Desktop/memedrive_experiments/output_data/intermediate_ratings_{i+batch_size}.csv',
                index=False
            )
            logging.info(f"Saved intermediate results at {i+batch_size} tweets")

    return pd.DataFrame(results)


def main():
    """
    Main execution function
    """
    # Set paths
    input_path = Path('~/Desktop/memedrive_experiments/input_data/community_archive.parquet').expanduser()
    output_dir = Path('~/Desktop/memedrive_experiments/output_data').expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate timestamp for file naming
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    logging.info(f"Starting tweet rating process at {timestamp}")

    # Load the data
    logging.info(f"Loading data from {input_path}")
    df = pd.read_parquet(input_path)
    logging.info(f"Loaded {len(df)} total tweets")

    # Sample N random tweets
    TWEET_COUNT = 25_000
    sample_size = min(TWEET_COUNT, len(df))  # In case there are fewer than 10k tweets
    sample_df = df.sample(n=sample_size, random_state=42)
    logging.info(f"Sampled {sample_size} tweets")

    # Log the row IDs (indices) of chosen tweets
    chosen_indices = sample_df.index.tolist()
    indices_file = output_dir / f'chosen_tweet_indices_{timestamp}.json'
    with open(indices_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'sample_size': sample_size,
            'indices': chosen_indices
        }, f, indent=2)
    logging.info(f"Saved chosen tweet indices to {indices_file}")

    # Process the tweets
    logging.info("Starting rating process...")
    ratings_df = process_tweets_batch(sample_df, client)

    # Merge ratings with original tweet data
    final_df = sample_df.reset_index(drop=False)
    final_df = final_df.rename(columns={'index': 'original_index'})
    final_df = pd.concat([final_df, ratings_df], axis=1)

    # Calculate composite good faith score
    final_df['good_faith_score'] = final_df[['sincerity', 'charity', 'constructiveness']].mean(axis=1)

    # Also save as CSV for easier inspection
    csv_file = output_dir / f'tweet_good_faith_ratings.csv'
    final_df.to_csv(csv_file, index=False)
    logging.info(f"Saved CSV version to {csv_file}")

    # Generate summary statistics
    summary_stats = {
        'timestamp': timestamp,
        'total_tweets_rated': len(final_df),
        'successful_ratings': final_df['sincerity'].notna().sum(),
        'failed_ratings': final_df['sincerity'].isna().sum(),
        'mean_scores': {
            'sincerity': final_df['sincerity'].mean(),
            'charity': final_df['charity'].mean(),
            'constructiveness': final_df['constructiveness'].mean(),
            'good_faith_overall': final_df['good_faith_score'].mean()
        },
        'score_distributions': {
            'sincerity': final_df['sincerity'].value_counts().sort_index().to_dict(),
            'charity': final_df['charity'].value_counts().sort_index().to_dict(),
            'constructiveness': final_df['constructiveness'].value_counts().sort_index().to_dict()
        }
    }

    summary_file = output_dir / f'rating_summary_{timestamp}.json'
    with open(summary_file, 'w') as f:
        json.dump(summary_stats, f, indent=2, default=str)
    logging.info(f"Saved summary statistics to {summary_file}")

    # Print summary
    print("\n" + "="*50)
    print("RATING COMPLETE")
    print("="*50)
    print(f"Total tweets rated: {summary_stats['total_tweets_rated']}")
    print(f"Successful ratings: {summary_stats['successful_ratings']}")
    print(f"Failed ratings: {summary_stats['failed_ratings']}")
    print("\nMean Scores:")
    for dim, score in summary_stats['mean_scores'].items():
        if score:
            print(f"  {dim}: {score:.2f}")

    return final_df

if __name__ == "__main__":
    # Run the main process
    results = main()
