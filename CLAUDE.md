

## Things to do next

- Clear what we’re trying to predict inc accuracy (be it specific locations or some cloud probabilistic thing)
- Set up to train and test on that (eg that landscape or cloud formation or whatever, for the next day - or next N days, but better to start small probably)
- Each model has to take training data and make predictions in the same way
- Have folder for models, and separate file for each model
- Start with v basic ‘model’ which is cloud learned from long history trends 

Files of interest for this:
- README.md for overview of project
- preproc/good_faith/README.md: deep overview of methods we might use
- csv_path = Path('~/Desktop/memedrive_experiments/output_data/tweet_good_faith_ratings.csv').expanduser() for the tweets, with 2d dimensions based on how 'good faith' they are
- embeddings_path = Path('~/Desktop/memedrive_experiments/output_data/community_archive_embeddings.npz').expanduser() for the full embedings of tweets



## Answers to questions from Claude

  Prediction Target:
  1. What exactly should we predict? Tweet locations in 2D good-faith space for the next day? Density clouds/probability
  distributions? Basin formation/dissolution events?
  A: tweet locations in 2d space over the next week (would do day, but week gives you more data), however this should be done probabilistically, so you are essentially predicting the kernel density of future tweets. 
  2. What's the prediction horizon? Next day, next week, or flexible N-day forecasting?
  a: Lets say week for now, and can edit to N-day later
  3. What constitutes prediction accuracy? Euclidean distance for specific locations, or probabilistic accuracy for cloud formations?
  a: Probabilistic accuracy

  Data & Scope:
  4. Should we use the full 2.2M tweet dataset or start with a smaller sample for faster iteration?
a: start with a sample, with ability to turn sample size up to the full dataset

  5. Do you want to predict individual tweet positions or aggregate patterns (like basin locations/strengths)?
a: probabilistic: we're almost predicting the 'field density' of the tweets, and if tweets tend to pop up in high density areas, then the model is doing good, and if low density its doign less well. The model should be rewarded for precision (ie gets more reward for being correct if area predicted was narrower, similar conceptually to a brier score in forecasting)

  6. Should we focus on the 2D good-faith dimensions (sincerity/charity) or also consider the full 768D embeddings space?
  a: start with 2d good faith dimensions

  Model Requirements:
  7. Do you want the baseline "cloud learned from long history trends" model to simply predict tomorrow's tweet density based on historical averages at each location?
  a: yea should be v simple, so long as no data leakage

  8. Should each model output point predictions, probability distributions, or both?
  a: probability distributions

  9. How should models handle the temporal decay of tweet influence you mentioned?
  a: this should be part of the modelling processes: its oart of their learning - to find the best approach to this

  Evaluation:
  10. How should we measure model performance? Mean squared error for locations, KL divergence for distributions, or basin detection accuracy?
  a: see above for the brier-score like ideas

  11. What constitutes a "good enough" baseline to beat? Random placement, historical average, or something else?
  a: historical avg





