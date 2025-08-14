# pip install pandas sentence-transformers fastparquet
import pandas as pd
import numpy as np
import time
from sentence_transformers import SentenceTransformer
import torch
import os
from multiprocessing import cpu_count

print('torch.cuda.device_count():', torch.cuda.device_count())

def check_hardware():
    """See what's available on your M1 Pro"""
    print(f"PyTorch version: {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"MPS built: {torch.backends.mps.is_built()}")
    print(f"CPU cores: {cpu_count()}")

    if torch.backends.mps.is_available():
        print("Using MPS (Metal Performance Shaders) - your M1 GPU")
        return 'mps'
    else:
        print("MPS not available, falling back to CPU")
        return 'cpu'

check_hardware()

t1 = time.time()
df = pd.read_parquet('~/Desktop/memedrive_experiments/input_data/community_archive.parquet')
print(f'seconds to read community archive file: {time.time() - t1}')



def mps_embeddings_chunked(df, batch_size=128, chunk_size=200_000):
   """Harness your M1 Pro GPU via Metal Performance Shaders with chunked saves"""
   device = 'mps' if torch.backends.mps.is_available() else 'cpu'
   print('Device used is: ', device)

   # Ensure output directory exists
   output_dir = os.path.expanduser('~/Desktop/memedrive_experiments/output_data/community_archive_embeddings')
   os.makedirs(output_dir, exist_ok=True)

   model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device=device)
   tweets = df['full_text'].tolist()
   total_tweets = len(tweets)

   print(f"Processing {total_tweets} tweets in chunks of {chunk_size}")

   for chunk_idx in range(0, total_tweets, chunk_size):
       chunk_start = chunk_idx
       chunk_end = min(chunk_idx + chunk_size, total_tweets)
       chunk_tweets = tweets[chunk_start:chunk_end]

       a = chunk_idx//chunk_size + 1
       b = (total_tweets + chunk_size - 1)//chunk_size
       chunk_count = 29 * a / b
       print('chunk_count', chunk_count)

       ## Hack to ensure we start with chunk 11
       if chunk_count < 10.9:
           continue

       print(f"\nProcessing chunk {chunk_count}")
       print(f"Tweets {chunk_start} to {chunk_end-1}")

       t1 = time.time()
       embeddings = model.encode(chunk_tweets,
                                batch_size=batch_size,
                                show_progress_bar=True,
                                convert_to_numpy=True)
       elapsed = time.time() - t1
       print(f"Chunk processing: {elapsed:.2f}s ({len(chunk_tweets)/elapsed:.0f} tweets/sec)")

       # Save chunk
       chunk_filename = f'embeddings_chunk_{chunk_idx//chunk_size:04d}.npz'
       chunk_path = os.path.join(output_dir, chunk_filename)

       t_save = time.time()
       np.savez_compressed(chunk_path,
                          embeddings=embeddings,
                          start_idx=chunk_start,
                          end_idx=chunk_end-1)
       print(f"Saved {chunk_filename} in {time.time() - t_save:.2f}s")


def concatenate_embeddings(output_dir):
   """Concatenate all embedding chunks back into single array"""
   output_dir = os.path.expanduser(output_dir)
   chunk_files = sorted([f for f in os.listdir(output_dir) if f.startswith('embeddings_chunk_')])

   if not chunk_files:
       print("No chunk files found")
       return None

   print(f"Found {len(chunk_files)} chunk files")
   embeddings_list = []

   for chunk_file in chunk_files:
       chunk_path = os.path.join(output_dir, chunk_file)
       data = np.load(chunk_path)
       embeddings_list.append(data['embeddings'])
       print(f"Loaded {chunk_file}: shape {data['embeddings'].shape}")

   full_embeddings = np.vstack(embeddings_list)
   print(f"Concatenated embeddings shape: {full_embeddings.shape}")
   return full_embeddings

# Run the chunked embedding process
# mps_embeddings_chunked(df, chunk_size=200_000)

# Concatenate all chunks if you need the full matrix
embedded_df = concatenate_embeddings('~/Desktop/memedrive_experiments/output_data/community_archive_embeddings')
np.savez_compressed(os.path.expanduser('~/Desktop/memedrive_experiments/output_data/community_archive_embeddings.npz'), embeddings=embedded_df)
