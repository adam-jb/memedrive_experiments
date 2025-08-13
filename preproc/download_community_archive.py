
# Downloads from azure
import pandas as pd

sas_url = 'https://digitalsrstorageaccount.blob.core.windows.net/test/network.parquet?sp=r&st=2025-08-13T15:15:06Z&se=2025-08-13T23:30:06Z&spr=https&sv=2024-11-04&sr=b&sig=%2B34YMRRc42ol4DFjPtN8K6imyi5kHSnk1c2uYzkm8CA%3D'
df = pd.read_parquet(sas_url)
df.to_parquet('~/Desktop/memedrive_experiments/input_data/community_archive.parquet')
