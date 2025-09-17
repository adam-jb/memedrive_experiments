
## Ways of working

I prize simplicity: all your code should be extremely simple.

Where I ask for things, implement tho. do not be 'helpful' and try and add additional features: DO NOT DO THIS.

For NAs, do not just remove the rows, but find out the cause of these: work hard to do this. I will accept no hacks.


## Overall project which we're in the middle of doing

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

these are the columns and dtypes in tweet_good_faith_ratings:
original_index               int64
id                           int64
full_text                   object
created_at                  object
retweet_count                int64
favorite_count               int64
in_reply_to_status_id      float64
in_reply_to_screen_name     object
screen_name                 object
username                    object
datetime                    object
index                        int64
sincerity                  float64
charity                    float64
constructiveness           float64
error                       object
good_faith_score           float64

the column 'datetime' is in this format: 2024-01-07 20:49:34+00:00



## recent things done

- edit evaluate_model() in framework.py so it takes weighted mean across all weeks, by count of tweets for each week, not simple mean
- add parameter to run_test.py which sets the date window for your input data




## TO DO NEXT (in this order)
- add option of using 'field based' approach to probability density, rather than grid of probability densities. Need ideas to do this...
- understand why baselining has no predictive power: I think it should so thats a bad sign
- get better understanding of how precision-weighted scoring works, with worked examples written for clarity and drawn spatial diagrams, labelled
- make clearer in file names that testing framework is set up for good faith embeddings
- make clear (written instructions) how to adapt the testing framework for other embeddings of tweets
- map out all files, categorise them, and make clear where things might want to change


## Files setting out approaches to ultimately move to new model framework 
- Approaches described in preproc/good_faith/READMD.md




## summary of claude investigation into why the scores are so low, even though visually the density looks about right
```
Looking at the evaluation results, I can now explain why the scores are low:

  Root Cause: Precision-Weighted Score Penalizes Confidence Mismatch

  Key Findings:

  1. Random Uniform vs Historical Average:
    - Random: PWS = 0.0001 (terrible but consistent)
    - Historical: PWS = 0.0212 (200x better but still low)
  2. Historical Average Analysis:
    - Top 10% overlap: 0.7-1.0 (excellent spatial alignment!)
    - Predicted max density: ~0.26 (confident)
    - True max density: ~0.18-0.24 (similar confidence)
    - Entropy: 2.18 vs 2.35 (similar concentration levels)

  Why Scores Are Still Low Despite Good Performance:

  The Precision-Weighted Score formula severely penalizes any mismatch:
  score = sum(pred_weight × (2 × pred × true - pred²))
  where pred_weight = pred + 1e-8

  The penalty comes from the -pred² term:
  - When pred > true, the negative term dominates
  - Even small overconfidence gets heavily penalized
  - Perfect alignment only occurs when pred = true exactly

  Evidence:
  - Historical Average has excellent spatial overlap (0.7-1.0)
  - Historical Average has reasonable confidence levels (~0.26 vs ~0.20)
  - But small density mismatches get amplified by the quadratic penalty

  The Scoring Is Working Correctly

  The low scores actually indicate the PWS metric is very strict - it demands both:
  1. Correct spatial placement ✅ (Historical Average achieves this)
  2. Exact density matching ❌ (Hard to achieve perfectly)

  Historical Average is performing 200x better than random, which shows it has genuine predictive power. The absolute scores are low because PWS is designed to be a demanding metric.
```





## Claude overview of how sliding window works

```
    - For each test week, the model trains on all data up to that week (expanding window)
    - Line 268-277: train_mask = all_df['datetime'] < week_data['datetime'].min()
    - This ensures no data leakage - models never see future data

  Current Flow:
  Data: Jan 2023 ────────────── Dec 2024
             ↑                    ↑
       Training Data         Test Period
                            (last 20 weeks)

  For each test week:
  Week 1: Train on [Jan 2023 → Week 1 start] → Predict Week 1
  Week 2: Train on [Jan 2023 → Week 2 start] → Predict Week 2
  ...
  Week 20: Train on [Jan 2023 → Week 20 start] → Predict Week 20

  So the sliding window is working correctly - each week gets an expanding training window that includes all historical data up to (but not including) that prediction week.
```


## Claude overview of Precision-Weighted Score formula:

  score = np.sum(precision_weights * (2 * pred_norm * true_norm - pred_norm**2))

  Where precision_weights = pred_norm + 1e-8

  Minimum Score:
  - Approaches -∞ (negative infinity) when model predicts high probability where no tweets occur
  - Worst case: predict 100% in empty regions → large negative penalty

  Maximum Score:
  - Approaches 1 when prediction perfectly matches truth
  - Best case: predict exactly where tweets occur with high confidence
  - If pred_norm = true_norm everywhere, score ≈ 1

  Typical Range:
  - Random/poor models: -0.5 to 0.1
  - Good models: 0.3 to 0.8
  - Perfect model: ≈ 1.0

  The score rewards confident correct predictions heavily but severely penalizes confident wrong predictions, making it much more sensitive than traditional accuracy metrics.



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





