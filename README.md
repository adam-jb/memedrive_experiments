
## Next up

Plot be interactive

Clusters over time

See if can make algo to identify 'new ideas' and their paths

See if can predict and test changes in idea fronts (similar to how one predicts the weather)

See if can find 'semantic basins' (ie attractors for certain topics) and what their topology is, algorithmically.

Look for basins of particular things, such as 'high trust', and how people move towards or away from these




## Ideas
View movement of 'the conversation' inc topics as per embeddings

Can see what started 'change in topic' or 'movement'? Will beed a way to quantify what I mean by these

Want a good way to quantify how much tweets shift 'the discourse' (eg do they bring in new ideas, etc)
This is predicated on one or two tweets having some outsized 'tipping point' effect: it may be that that's not how ideas tend to move around and ideas tend to more often spread only when lots of people make lots of small contributions

Then consider if I can make predictions as to what will become big
It's possible to use this dataset to *both* train and test


## Preproc pipeline, in order it was run

preproc/Network Analysis.html: export of notebook which gets all tweets from all community archive on 13th August 2025. And concatenates them to one big parquet file. Stores the file in Azure blob storage temporarily.

preproc/download_community_archive.py: downloads the community archive parquet file from Azure with a temporary SAS token.

preproc/embed_community_archive.py: gets embeddings for all tweets based on the full text of the tweet (not images or usernames). Uses sentence-transformers/all-mpnet-base-v2' for embeddings.

preproc/umap_embeddings.py: reduce all embeddings to 2d.

preproc/plot_sample_embeddings.py: plot 10k random samples of the 2d embeddings

preproc/plot_animation.py: plots animation of weekly tweets in 2d embedding space


## Main blockers to immense success

Getting big and comprehensive data on discourse (eg even a sample of lots of Twitter data v expensive to do via the official API)
