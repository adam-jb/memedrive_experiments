
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

- add option of using 'field based' approach to probability density, rather than grid of probability densities. Need ideas to do this...
want field probability approach instead, where field can flex in harshness to fit needs of customers

one musing on this: is there a way to have the densities be more than just collections of circles (or close to circles) eg a way to allow predictions with any shape of field? Or could one way be to make predictions as a dense grid (Eg 500x500) and then put
  gausians around the tweets themselves? 

implemented the field density approach to eval metric, calling it Field Density Score (FDS):
- have the 'trained' model return an N*N grid, each of which is the relative probability of a tweet appearing there in the next time unit (week atm)
- the average probability of these should be 1, so the mean probability of all cells is 1
- the final metric could be the weighted average of the square each tweet from the next week appears on, with the weighting being the importance of that tweet
- so a final metric of 1 should be equivalent to randomness; 0 means that the model is perfectly wrong (the most wrong it could be); and the maximum is huge: a proportion of how much better than randomness it is
- start with a probabilistic grid
This is easier to understand than more complicated things, though may be insensitive to modelling improvements which are valuable for customers: might make bespoke versions for customers, which zoom in on particular areas. 

Add parameter in main runner, which if set leads to making an animation with 2 frames for each prediction: one with the predicted density, and then a 2nd with the same density and the actual tweets overlaid. Should set this for each model, so can decdie which get animated. Animations should save in 'image_outputs' folder



## TO DO NOW in order

implement 'drift field' modelling approach, in new script under models/ folder

vague-ish details below of how this might look

involves a mix of momentum (physics-based) of things, with uncertainty of that momentum, plus extra source of uncertainty of not knowing where tweets are going to be bc they are just hard to predict. The underlying theory is that there is some conservation of momentum between tweets. 

IMPORTANT: Might want to a way to look at tweets day to day, so we're predicting with more temporal granularity: could improve the performance of the model if it can harness that level of detail

want to optimise how diffuse the predicted probability density is such that it maximises FDS .

1) Drift-field change (directly tests “discourse begets discourse”)
could this work if we dont have actual replies, only spatial closeness? Yes
It's related to Optical Transport. They both give you “arrows showing how density moves between snapshots.” Intuition of the difference:
- OT with smoothing = “Imagine you’re solving a logistics problem: every unit of density must be trucked from somewhere at time t to somewhere at time t+1 at minimal cost.”
- Density-flow drift field = “Imagine you’re watching clouds and measuring how they drift frame-to-frame. You don’t worry about exact conservation; you just see local motion.”
OT conserves all mass, so mass will be preserved between t and t+1: all the calculations are global and mass is conserved globally.

Drift-field change (think GPT made this up) is better for noise, doesnt preserve mass globally, goes calculations on local levels (ie considers sections of the overall space sequentally and, I think, independently). Noise is controlled via local regularization rather than a globally set 'tau'

I think drift-field change is more suited to our use case than Optical Transport.




## Notes for using the model with customers

Assuming we get a well calibrated model wihch is sensitive to inputs, our Field Density Score metric can be used to clearly tell the expected impact of how many tweets of a given sentiment their social media campaign will lead to, inc the movements in the broader 'weather fronts' (which could be simulated N steps into the future via stepwise Monte Carlo simulations of the model). Would want to link the expected increase in tweets to meaning for the customer (ie it is an indicator of broader sentiment, where we treat the movements in tweets as representative of movement in broader un-tweeted sentiment). For the latter claim would want to do a quick lit review to check how strongly that holds up (I imagine it holds up a bit, but not perfectly): if can quantify this extra source of uncertainty then that would be anaytically great, but may not add much value to customers (product might already be good enough).

Add option to make bespoke versions for customers, which zoom in on particular areas. to implement: make an optional param which lets the users zoom in on a particular topic, defined as an N-dimensional box, and we only consider tweets in that box when evaluating (tho the training data can still use the full canvas, events outside the box could have predictive power so good to give the model access to this). 





## TO DO ONE DAY
- make clearer in file names that testing framework is set up for good faith embeddings
- make clear (written instructions) how to adapt the testing framework for other embeddings of tweets
- map out all files, categorise them, and make clear where things might want to change

allow predictions to be made in embeddings of over 2 dimensions (this may already be possible - am not sure)




## Files setting out approaches to ultimately move to new model framework 
- Approaches described in preproc/good_faith/READMD.md





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



## Answers to questions from Claude earlier in dev process

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





