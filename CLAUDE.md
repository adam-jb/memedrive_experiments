

## Process thoughts for me (Adam)
Could use some kind of methodological brain trust, esp to help me make good choices of model (particularly around favouring simplicity)


## To run experiments

May need to edit some params in the model files (Eg n_calls for the drift_field.py Bayes optimiser). Then set params and run `python run_tests.py`





## Ways of working

I prize simplicity: all your code should be extremely simple. Always the most simple possible while retaining all functionality.

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



## The most important things to get this show on the road
1. Convert data to a meaningful set of dimensions (the 2d good faith ones aren’t so good: there needs to be signal for the models to find)
2. Make 'weight' account for each tweet in calculate_fds_score() (testing/framework.py) [there is _fds_loss() in lstm_predictor.py too, which should really read the FDS func from framework, so we only have to define the formula once, or have the FDS func in a separate file to be sourced by all others]
3. Adapt code to allow predictions to be made in embeddings of over 2 dimensions (this may already be possible - am not sure)


Other investigations:
1. look for basins, clusters and what I might count as memeplexes

Things which might help:
1. Downloading and using the 1bn+ tweets dataset
2. Make resolution higher


## Observations on Drift Field
1. It performs about the same as historical avg when trained on 100k sample and 200 bayes iters. However looking at the image, it distributes the predictions very differently to the baseline (clustering it too centrally). Makes me wonder if LSTM better
2. Might do better with more granular data (ie daily not weekl: contagions can spread very fast so a week might not be granular enough to model the 'physics' of the system)
3. Looked for if there are any scaling laws between sample size and quality of predictions [A: as of 22nd Sept 2025 I dont think there are: its max performance equals the baseline].




## On deep neural network

original instruction: time to create a new model for the framework! The idea is to use a deep neural network which trains on n-dimensional images of what the 'perfect' probability density distribution would be for that frame (based on its points and all previous points). The model be a time series model, which predicts the next frame from all the frames so far. What do you need to ask to clarify this? 

earlier thought which was implements: Can treat the learnable sigma parameter (how large the gaussian over each tweet is) as a regularisation term: larger gaussians around tweets means the model fits less strongly to areas. That's good. We need the actual 'reward signal' to be the right balance between this regulariser and what is actually being predicted (the location of specific tweets as per the FDS). One way to do this is have the FDS, or a version of FDS, be what is optimised for during LSTM training. 





## Notes for using the model with customers

Assuming we get a well calibrated model wihch is sensitive to inputs, our Field Density Score metric can be used to clearly tell the expected impact of how many tweets of a given sentiment their social media campaign will lead to, inc the movements in the broader 'weather fronts' (which could be simulated N steps into the future via stepwise Monte Carlo simulations of the model). Would want to link the expected increase in tweets to meaning for the customer (ie it is an indicator of broader sentiment, where we treat the movements in tweets as representative of movement in broader un-tweeted sentiment). For the latter claim would want to do a quick lit review to check how strongly that holds up (I imagine it holds up a bit, but not perfectly): if can quantify this extra source of uncertainty then that would be anaytically great, but may not add much value to customers (product might already be good enough).

Add option to make bespoke versions for customers, which zoom in on particular areas. to implement: make an optional param which lets the users zoom in on a particular topic, defined as an N-dimensional box, and we only consider tweets in that box when evaluating (tho the training data can still use the full canvas, events outside the box could have predictive power so good to give the model access to this). 





## Info on scripts

  Visualization/Animation and/or preprocessing:
  - preproc/network_analysis.html: .html extraction of a python notebook (originally run in microsoft fabric) which downloads all the data from the community archive and stores them in azure blob storage. Could adapt this to download freah copies of the community archive
  - preproc/download_community_archive.py: one-use-only code to download preprocessed community archive data from azure blob storage with a SAS token. Won't work now the SAS token has run out. 
  - preproc/embed_community_archive.py: use sentence-transformers/all-mpnet-base-v2 model to make embeddings of all tweets in the community archive, using locally run model (uses GPU of M1 Pro). Embeddings are based on the main text of the tweet (text only - ignores if replying to someone or not, and any images) Saves them all as a large matrix, with the row indices mirroring those in the main community archive parquet file
  - preproc/umap_embeddings.py: train UMAP model on a sample of community archive embeddings, then apply to all of them, to get 2d representation of the embeddings
  - plotly_interactive.py: makes an interactive plotly chart of all 2d embeddings, with a slider to move between weeks. and shows the tweets on hover. Makes a 500mb html file.
  - plot_animation.py: makes gif with frame for each week of 2d embeddings. Same data as shown in plotly_interactive.py but simpler representation.
  - preproc/datamapplot_animation.py: similar to other plots but using datamapplot py library


  Good Faith Processing:
  - preproc/good_faith/get_good_faith_ratings.py: get GPT to rate how 'good faith' tweets are according to 3 criteria (thus giving us 3d information for the sample tweets). At time of writing the criteria (which I think have plenty of room to improve) are sincerity, charity and constructiveness (issues with these 3 is they leave out humour, and the lack of context of isolated tweets make this hard to judge)
  - preproc/good_faith/create_transformation_matrix.py: both learns and applies a transformation matrix to convert the embeddings matrix of all tweets to 3d, where the 3 dimensions map to the good faith ratings as judged by GPT. It's essentially a linear model, predicting each of the 3 dimensions, with all 768 embeddings as features
  - preproc/good_faith/3d_to_2d_umap.py - train UMAP on a sample of tweets, and apply to all tweets, to convert from 3d to 2d (input data is on 3 dimensions of 'good faith' and output is 2d). If the good faith ratings were obtained in 2d in the first place this script could be skipped
  - create_community_archive_good_faith_embeddings.py: horizontally concat 2d good faith rankings and community_archive.parquet, to make file for main modelling: community_archive_good_faith_embeddings.csv


  Good Faith visualisations: 
  - preproc/create_tweet_timeseries.py: makes the json file used by preproc/good_faith/explorer.html (thats the only file it needs). Makes this from a single file: output_data/community_archive_good_faith_embeddings.csv (could edit to take embeddings other than good faith). The created file (tweet_timeseries.json) is a newer version of tweet_timeseries_ORIGINAL_LEGACY.json, which includes a separate entry for each tweet a number of days after it happened, to model the 'fade out' process, whereas at time of writing (21st sept 2025) the newer one just has the current date of tweet and then it vanishes the next date. The initial weight of the tweet is also calculated with log10, as per below function, which I expect underweights the impact of larger tweets. So: room to improve this script to fix that!
  ```python
  def calculate_tweet_size(favorite_count, retweet_count):
    """Calculate tweet size based on engagement metrics"""
    # Using logarithmic scaling similar to the original
    engagement = favorite_count + retweet_count
    if engagement == 0:
        return 1.0
    return np.log10(engagement + 1) * 1.5 + 1.0
  ```
  - preproc/good_faith/explorer.html: could use this as the base for more general tweet explorer. Uses WebGL to render lots of points very fast. Nice design imo. Could make easier to host by loading the tooltip info from azure cosmos db or similar. Reads from a single file: tweet_timeseries.json - for this to not have CORS issue need to host a py server to view locally: `python3 -m http.server`

  Predicting future tweets:
  - run_tests.py - main test runner. Set params for the models you want to test here, and then run to get all your results. Reads data from community_archive_good_faith_embeddings.csv. Expected models to be defined in a consistent way
  - models/baseline.py - simple baseline models, inc fully random (FDS=1), and baseline of taking density of all tweets in the training data, and applying nothing more than that. 
  - models/drift_field.py - drift field implementation
  - testing/framework.py - testing framework with FDS evaluation. Defines the base model class TweetPredictor, which all models inherit. Is called by run_tests.py and does lots of the heavy lifting of the testing process.




## Datasets
Community Archive = tweets made by several hundred prominent tweeters in the specific areas of twitter (eg 'TPOT')




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





## Thoughts on memeplexes

A memeplex is a cluster of memes which are mutually reinforcing

Trumpism is such a cluster. With a strong meme at the centre (Trump himself) and many other memes which reinforce Trump to their own benefit.

Trumpism was originally smaller, however other memes (ie public figures) decided (often correctly) that their own memes could grow if they aligned with Trump. So they support Trump, tapping into that energy while adding to the energy around Trump himself

To replicate such a dynamic, one need a central meme (or, leader) who is popular and memetic enough that influential others feel that its in their self interest to support your leading meme. 

I expect this would need to be done over a long timeline, in order to build the brand. With the dynamics of modern media, that timeline could be shorter than it used to be, however I think Trump and Farage have both benefitted from being public figures for over a decade now. In other words, their memes took a while to reach sufficient strength that many others would start trying to align themselves with them, and thus enter their memeplex

As well as a strongly memetic leader, one also needs enough influencers (of all kinds) who will be receptive to their message. I think this might partially compensate for weakness of memetic strength of the leader, but a better way of thinking of this is as a 2nd angle of attack: go for memeticity of leadership (most important thing) then make the memes easy to support for the people you want to support them. If such an ecosystem of candidates already exists, then great! If it doesn't and you're playing the long game, you may want to invest in 'seeding' potential influencers: lots of low-cost, low-chance-of-success, high-reward bets. Such a seeding process might look similar to a startup incubator, but for ideologically aligned and promising influencers. 

"Influencers" here means anyone with any form of public following, not just people who's main income comes from social media accounts.



# More info on deep neural net

LSTM Tweet Prediction Model Summary

  Model Architecture

  Input Processing:
  - Time Series Conversion: Tweet data → sequence of density grids over time frames
  - Spatial Representation: Each frame is a 2D grid (default 100×100) representing tweet density in good-faith space (sincerity × charity)
  - Sequence Learning: LSTM processes last N frames (default 7) to predict next frame's density

  Neural Network Structure:
  Input: [batch, sequence_length, grid_size²]
     ↓
  LSTM: [hidden_size=64, num_layers=2] + dropout
     ↓
  Dense: [hidden_size → grid_size²]
     ↓
  Softmax: Valid probability distribution
     ↓
  Output: [grid_size, grid_size] density prediction

  Key Innovation: FDS-Optimized Training

  Traditional approach: Train on synthetic density targets using MSE loss
  Our approach: Train directly on actual tweet positions using Field Density Score loss

  Loss Function:
  FDS_loss = -mean(log(predicted_probability_at_actual_tweet_locations))

  Benefits:
  - Optimizes exactly what we measure (tweet prediction accuracy)
  - No proxy targets or misaligned objectives
  - Learnable sigma finds optimal balance between precision and generalization

  Learnable Sigma Regularization

  Concept: Gaussian width around tweets becomes trainable parameter
  - Larger σ → smoother predictions → less overfitting → potentially lower precision
  - Smaller σ → sharper predictions → more overfitting → potentially better precision
  - Learned σ → optimal balance for FDS performance

  Recommended Hyperparameters

  Core Architecture:
  sequence_length=7        # 7 historical frames
  hidden_size=64          # LSTM hidden units  
  num_layers=2            # LSTM depth
  grid_size=100           # 100×100 spatial resolution

  Training:
  epochs=30               # Training iterations
  batch_size=32           # Batch size
  learning_rate=0.001     # Adam optimizer
  gaussian_sigma=0.05     # Initial σ (will be learned)
  learn_sigma=True        # Enable σ optimization

  Temporal:
  frame_duration_days=1.0  # Daily predictions (can be <1 for hourly)

  Why This Works

  1. Direct Optimization: Loss function = evaluation metric (FDS)
  2. Temporal Patterns: LSTM captures how tweet density evolves over time
  3. Spatial Awareness: Grid representation preserves 2D relationships
  4. Adaptive Regularization: Sigma learns dataset-specific optimal smoothness
  5. End-to-End: All parameters optimize for final prediction task

  Result: Model that learns to predict where tweets will actually appear, not just fit smooth density surfaces.



# Thoughts on drift field

implement 'drift field' modelling approach, in new script under models/ folder

vague-ish details below of how this might look

involves a mix of momentum (physics-based) of things, with uncertainty of that momentum, plus extra source of uncertainty of not knowing where tweets are going to be bc they are just hard to predict. The underlying theory is that there is some conservation of momentum between tweets. 

IMPORTANT: Might want to a way to look at tweets day to day, so we're predicting with more temporal granularity: could improve the performance of the model if it can harness that level of detail

want to optimise how diffuse the predicted probability density is such that it maximises FDS .

important that the parameters which can be learned include the decay of a tweets impact

important that some kind of 'movement' or 'momentum' is factored in for the general 'flow' of tweets in a given area, even though all tweets are static

1) Drift-field change (directly tests “discourse begets discourse”)
could this work if we dont have actual replies, only spatial closeness? Yes
It's related to Optical Transport. They both give you “arrows showing how density moves between snapshots.” Intuition of the difference:
- OT with smoothing = “Imagine you’re solving a logistics problem: every unit of density must be trucked from somewhere at time t to somewhere at time t+1 at minimal cost.”
- Density-flow drift field = “Imagine you’re watching clouds and measuring how they drift frame-to-frame. You don’t worry about exact conservation; you just see local motion.”
OT conserves all mass, so mass will be preserved between t and t+1: all the calculations are global and mass is conserved globally.

Drift-field change (think GPT made this up) is better for noise, doesnt preserve mass globally, goes calculations on local levels (ie considers sections of the overall space sequentally and, I think, independently). Noise is controlled via local regularization rather than a globally set 'tau'

I think drift-field change is more suited to our use case than Optical Transport.


## On the sliding window in drift field method
as i understand it, sliding window is best bc we only train up to history_window anyway and dont use older data, so sliding window size should be a a function of that. so we learn params for drift field with that, and we also apply said sliding window at test 



### Drift field params (may be out of date given the pace of development!): can add more to improve model down the line (this is simpler approach)
LEARNABLE_PARAMETERS = {
      # Temporal
      'history_window': [7],                      # How many past timesteps to use for learning momentum patterns
      'temporal_decay': [0.05, 0.1, 0.15, 0.2, 0.3],  # How much total density fades away each timestep (prevents infinite accumulation)

      # Movement & Flow
      'drift_scale': [0.5, 1.0, 1.5, 2.0, 2.5],       # How many grid cells density moves per timestep when following momentum
      'momentum_weight': [0.7],                   # What fraction of moving density follows learned patterns vs spreads randomly
      'density_persistence': [0.1, 0.3, 0.5, 0.7, 0.9],  # What fraction of density stays in the same grid cell vs moves somewhere

      # Spatial Processing
      'correlation_window_size': [3],             # Size of neighborhood (3x3) used to detect where density flowed from
      'local_region_size': [12],                  # Size of spatial chunks for independent momentum calculations

      # Final Prediction
      'diffusion_strength': [0.3],                # How much Gaussian blur to apply to final prediction (uncertainty spreading)

      # Tweet Importance in FDS
      'retweet_importance_weight': [1.5],         # How much more retweeted tweets count in the Field Density Score metric
  }


## Ways to improve drift field method
Learn params with bayesian optimisation rather than grid search: should learn better params in such cases



## Ways to make the drift field method more complicated
  - Adaptive history window parameters (min/max/adaptation rate)
  - Non-linear temporal weighting curves
  - Tweet importance weighting (retweets, likes, etc.)
  - Spatial adaptation parameters for different density regions



## Thoughts on getting to adequate level of predictive modelling
Need to increase frequency  of monitoring (that’s all on eval side), then factor in more granular tweets with timestamps and learnable decay factors (that’s all on predictive modelling side)



 ## Claude's Questions about Drift Field Method:

  1. Momentum Calculation
  - How do you calculate "momentum" from tweet positions? Is it the velocity vector of density centroids between time steps?
  - Do individual tweets have momentum, or does the overall density field have momentum?
  A: I think... tweets dont have momentum, so it's more the chain of tweets that has momentum (eg tweets consistently moving in a particular path, not necessarily linear, and with lots of noise, but there being some kind of signal in the noise)

  2. Temporal Granularity

  - You mentioned predicting day-to-day vs week-to-week. Would the drift field model:
    - Train on daily snapshots to learn finer-grained movement patterns?
    A: daily seems good (with some temporal persistence of effects of tweets, which can be a learned parameter) In future we might do it even more granularly than daily (eg hourly or similar), given how fast thought contagions can move on twitter (Eg information cascades)
    - Predict daily movements then aggregate to weekly predictions?
    A: Can we have a paramenter to switch between daily and weekly on this? (With other options to add later)
    - Or work directly at whatever time scale we specify?

  3. Local vs Global Processing

  - When you say "calculations on local levels" - do you mean:
    - Divide the 2D space into regions and calculate drift for each region independently?
    - Use sliding windows across the spatial grid?
    - Process each grid cell based only on its immediate neighbors?
    A: not sure, might be ok to ignore this

  4. "Arrows of Movement"

  - What exactly would these arrows represent?
    - Velocity vectors showing how density is flowing from one region to another?
    - Gradient fields pointing toward where tweets are likely to appear next?
    - Something else?
    A: the arrows would be in service of predicting the field density of where tweets are likley to appear next. However the velocity vectors might be good in the service of this. Thoughts?

  5. Input/Output Structure

  - Input: Would it take a sequence of historical density snapshots (e.g., last N weeks)?
  - Output: A predicted density grid for the next time period, or also velocity/drift vectors?

  6. "Discourse begets discourse"

  - How would spatial closeness proxy for discourse relationships without actual reply chains?
  - Is the idea that tweets in similar good-faith regions influence each other's future likelihood?

  The physics analogy of "watching clouds drift" is compelling - are you envisioning something like fluid dynamics applied to tweet density fields?
