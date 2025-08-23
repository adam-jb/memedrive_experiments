
Experiments on looking at 'good faith' in embeddings space

Relies on 'preproc/embed_community_archive.py' being run already

## Files

First two can combine to BE A GENERALISABLE TOOL: reducing embedding dimensions to a few dimensions concerning something of interest (eg Good Faith, or a certain topic)

get_good_faith_ratings.py gets LLM-based assessments of how good faith the tweets are.
- Uses 3 dimensions to define what 'good faith' is ('sincerity', 'charity', 'constructiveness') rated on scale of 1-7 by an AI. Arrived at these via discussion with Claude, which could refer to if open up the discussion of how to define Good Faith again: https://claude.ai/public/artifacts/e74e73b9-dd6f-41c3-bc5c-1ae9bf638755

create_transformation_matrix.py
 - Use linear regression to make a matrix to transform embeddings to 3d which are the ‘good faith’ where the 3 dimensions sincerity, charity, constructiveness.
 - Apply transformation matrix to all 5.5m tweet embeddings
 - Tested (code in older commits) ridge regression, CCA (3d and 1d) and random forest. ridge and 1d CCA got same result as linear regression; 3d CCA and RF did worse. So sticking with linear regression. Correlation between predicted and actual on 5k test set:
 ```
 sincerity: 0.6741
 charity: 0.6311
 constructiveness: 0.7000
```


explorer_preproc.py
 - get_fadeout_multiplier() can be changed to set the rate at which tweets 'degrade' over time in the plot
 - August 22nd: uses the first 2 dimensions from the 3d good faith, as UMAP not run at time of trying
 - Has the SECONDARY use of creating a merged csv of community archive tweets & their good faith dimensions, as it was doing that anyway
 - has HACK TO LIMIT FILE SIZE. As without that we'd be at a 5gb json output. Use this to determine how much data you generate visuals for


## Unused files
3d_to_2d_umap.py
 - So can visualise: goes from 3d of good faith to 2d
 - 23rd Aug 2025: Dont need because (1) unsure how useful 3d data is, and (2) its much nice to plot the original axes so can be clear what they mean


## explorer.html
To handle CORS to load local json files into local explore.html
```
python -m http.server 8000
```

 IF NEED A WAY TO have all the data load and be interactive (maybe start with just coords, and tooltip be key/value store which could run locally for starters)
   Good way to do this is likely to be IndexedDb and break tooltip files into separate ones for each day. Then it loads them whenever date changes (after a wait so not trapped by animations), as people can wait the 100-300ms it takes to load the 1mb file. Structure for this could look like:
```
   tweets/
   ├── render_data/
   │   ├── 2023-01-01.json     (~50KB, ~1000 tweets)
   │   ├── 2023-01-02.json
   │   └── ...
   └── full_data/
       ├── 2023-01-01.json     (~500KB, full tweet data)
       ├── 2023-01-02.json
       └── ...
```
Claude thinks Redis would be overkill for this. Use IndexedDb. See 2nd half of convo with Claude on 22nd August 2025 'tweet visualisation loading issue' for more on how to do this - link to convo: https://claude.ai/share/0c9e5267-557c-4920-ab79-5dcc53963cd0



## Single thing being done right now
make good 'basin finding' tool:
 - want it to be standardised function I can apply all over the place
 - probably best for it to be a fusion of methods, then can see how well they concur. Could include any which have predictive power beyond some threshold... tho the basins have to be there to detect (could make some fake data which I know has basins to test this, which is smaller dataset so I can be sure the code works, while being same structure, col names, etc, as the main dataset); this might involve making my pipelines more unified, and functionalised (so can point at different files).
 want to ensure it can scale to different numbers of dimensions in embedding space
 >>try claude code?


make a func to make and store arbitrary embeddings data, with labels

base it on real labels of tweets from:
df = pd.read_parquet('~/Desktop/memedrive_experiments/input_data/community_archive.parquet')
[dont actually read the file but use the col names: datetime, full_text]

the fake datetimes should be better 1st jan 2023 and 28th feb 2023

and make fake separate embeddings pertaining to a sample of these

the embeddings should be in 2d space

ensure that there are genuine clusters & noise in the data, and that the clusters dont move around much over time

make a final df which has the df fields and the embedding coords, where all embeding fields start with 'e' so can find them automatically if they scale

save output file without timestamp in ~/Desktop/memedrive_experiments/output_data/basin_finder/






Approaches to bring in:
 - [TO CHANGE: bring in Drift-field change instead which is similar but more suited to this use case] Optimal Transport with entropic smoothing: converts space to grid and computes minimum flow (energy) to go from snapshot t to t+1. This can be very 'overfitt-y', so the entropic smoothing makes it fit less strongly, which is better when there is more noise. Increase this regularisation parameter 'epsilon' as noise increases (I probably want it to be very high indeed)

 - Graph metastability with time decay: what metric does this give me for basins, and is it overall or per basin, or per basin per timestamp? A: it can be per basin and per time stamp (the latter if I rerun the algo multiple times with different sliding windows, which sounds a good idea)

 - Kernel Density Estimation: get the density of all tweets, accounting for relative importance and fading over time.
  Uses temporal decay function: `w = engagement_weight * exp(-(t - s)/τ)` where t = time now, s = time posted, and tau is a constant. I can set tau, or have the model learn a value for tau which gives it the most predictive power. gpt5 suggests giving only a few options for tau for an ML model to choose from, to prevent overfitting.
  Is there a tau-like parameter for spatial decay too? (eg width of contribution of each tweet to overall density) Yes: the bandwidth parameter (h) of KDE.

- Cluster tracking + barriers/escapes
```
Nice thing about this is it tells a discrete story: actual clusters you can name/plot, with entry/exit dynamics, barriers, and lifetimes. Great for stakeholder comms and dashboards.

Idea
Find clusters in each time bin (DBSCAN or mean-shift on the (x,y) tweets that are temporally nearby). Track clusters over time (Hungarian matching). For each tracked cluster (= candidate basin), compute:
Center drift toward local density core,
Escape rate (how fast points leave),
Inward-move fraction: % of child tweets closer to cluster center than their parents (vs null).
Barrier (density drop along easiest exit path).
Psuudocode:
# 0) per time bin, select tweets in a rolling window with temporal weights
# 1) cluster
# 2) compute cluster stats (eg centre drift, )
# 3) track clusters across time via IoU / nearest centers (Hungarian) [i think we'd need a measure of how much the clusters are changing between time stamps: large changes mean they are more likely to be a result of noise]
# 4) escape rates: follow member tweets across future bins
# 5) barriers via density-weighted shortest paths

Likely isue with cluster tracking
- Cluster instability; mitigate via stability selection across bandwidths and seeds.
```

- Spatio-temporal Hawkes (self-excitation) on regions

- Stochastic Lyapunov analysis




GPT thoughts on parameter specifying: 'the value at which strength peaks: that’s the “natural” value for that feature the basin.'


Other ideas
```

Optimal Transport with entropic smoothing: converts space to grid and computes minimum flow (energy) to go from snapshot t to t+1. This can be very 'overfitt-y', so the entropic smoothing makes it fit less strongly, which is better when there is more noise. Increase this regularisation parameter 'epsilon' as noise increases (I probably want it to be very high indeed)


1) Drift-field change (directly tests “discourse begets discourse”)
could this work if we dont have actual replies, only spatial closeness? Yes
It's related to Optical Transport. They both give you “arrows showing how density moves between snapshots.” Intuition of the difference:
- OT with smoothing = “Imagine you’re solving a logistics problem: every unit of density must be trucked from somewhere at time t to somewhere at time t+1 at minimal cost.”
- Density-flow drift field = “Imagine you’re watching clouds and measuring how they drift frame-to-frame. You don’t worry about exact conservation; you just see local motion.”
OT conserves all mass, so mass will be preserved between t and t+1: all the calculations are global and mass is conserved globally.

Drift-field change (think GPT made this up) is better for noise, doesnt preserve mass globally, goes calculations on local levels (ie considers sections of the overall space sequentally and, I think, independently). Noise is controlled via local regularization rather than a globally set 'tau'

I think drift-field change is more suited to our use case than Optical Transport.



7) Another approach for graph metastability using markov stability
**involves binning space and finding transition probailities: my sense is the data are too sparse and this too crude to be much good for this use case**
Build a kNN graph over tweets with temporal weights (decay) and conversation edges. Run diffusion maps (or Markov stability / PCCA+) to find metastable sets—regions where a random walk gets “trapped.”
What to compute
Conductance / normalized cut of sets (lower = stronger basin).
Spectral gaps: large gap ⇒ clear metastability.
Mean return / first passage times within sets (escape difficulty).
Track sets over time to test persistence.
Why it’s good
Captures non-convex, multi-basin structure without strong parametric assumptions.
Pitfalls
Scaling to 2.2M nodes: use sampling, landmark diffusion, or cluster-then-graph.

What it adds (vs your current time-decayed graph idea)
Clear basin strength via conductance, mean first-passage time (MFPT), and Markov Stability at multiple Markov times.
Natural per-basin, per-time scores by using sliding windows or exponential time weights.

to ask: what are conductance, mean first-passage time (MFPT), and Markov Stability ?
conductance = Measures how “leaky” a set of nodes is in the graph.
Low conductance → a random walker starting in this set rarely leaves → strong basin.
High conductance → the set is loosely connected → not a stable basin.

Mean first passage time: Imagine a random walker starting inside the basin. MFPT = average number of steps before it leaves the basin.
Longer MFPT → basin is “deep” or “sticky.”
Short MFPT → easy to escape → weak basin.

 Markov Stability: Treats a random walk as a Markov process over the graph.
 train markov model by: Estimate transition probabilities between bins based on the drift of density between snapshots, giving us a row-stochastic transition matrix
Measures how long a set of nodes retains probability mass under diffusion.
Large spectral gap or high Markov stability → a metastable set = strong candidate basin.

on spectral gap: In this context, “spectral” refers to eigenvalues and eigenvectors of a matrix that represents your system, usually the graph Laplacian or transition matrix of a Markov chain. It’s called spectral because it’s literally the “spectrum” (set of eigenvalues).
The difference between the first (largest) and second largest eigenvalue of P (or between first non-zero eigenvalues of Laplacian).
Measures how slow a random walker escapes a cluster:
Big gap → slow escape → strong basin.
Small gap → fast escape → weak basin.


psuedocode:
```
# 1) edges by spatial + temporal proximity (no replies needed)
for (i, j) in approximate_knn(df.xy, k=K):
    w_spatial  = exp(-dist_xy(i,j)/sigma_xy)
    w_temporal = exp(-abs(df.t[i]-df.t[j])/tau_time)
    w = w_spatial * w_temporal * engagement_scale(i,j)
    if w > eps: add_edge(G, i, j, weight=w)

# 2) (option A) sliding-window graphs
for r in windows:
    G_r = subgraph_by_time(G, window=r, decay=None)
    communities = spectral_or_leiden(G_r, n_comms="auto")

    for C in communities:
        cond[C,r]  = conductance(G_r, C)
        mfpt[C,r]  = mean_first_passage_time(G_r, C)
        mstab[C,r] = markov_stability(G_r, C, markov_time_grid)

# 2) (option B) single graph with continuous exponential time weights
# same scoring; times already baked into edges

# 3) persistence across time
tracks = match_communities_over_time(communities, criterion="Jaccard_on_nodes")
for track in tracks:
    per_time_scores = { "cond": cond[track,:], "mfpt": mfpt[track,:], "mstab": mstab[track,:] }
    aggregate_scores(track, per_time_scores)
```



3) Spatio-temporal Hawkes (self-excitation) on regions
Idea
Discretize the plane into regions (adaptive cells around candidate basins). Fit a marked Hawkes process where past tweets in region A elevate intensity of future tweets in A with decay
ϕ
ϕ. Basins correspond to high self-excitation and low cross-excitation.
3) Spatio-temporal Hawkes on regions (self-excitation): hwo does it work?

“Self-exciting” = past events increase the probability of future events nearby in space and time.

first step: discretize space
Either a regular grid or adaptive cells around candidate KDE peaks.
Each cell is a  “region” where we count events.

the underlying “attractiveness” of regions changes over time, because basins can emerge, persist, and fade. This means you cannot assume stationarity over the full dataset. **So want to keep retraining models over a sliding window**

the temporal window length for retraining acts as a hyperparameter controlling temporal resolution vs stability

it gets you expected intensity of future tweets in that region. Check out the temporal persistence: Persistent high scores → stable basin

sparse regions may produce unreliable parameter estimates. Could use regularization or thresholding to discard very low-activity regions.




7) SDE + (stochastic) Lyapunov analysis
Tweets themselves are static once posted.
The “flow” happens in future tweets appearing nearby: a basin attracts new events, not moves existing ones.
So you want a density-flow of events over time, not displacement of points.

Treat local tweet density as a stochastic process:
density(x, t+Δt) = f(density(x, t)) + noise
“Drift” = tendency for density to increase near high-density points (self-excitation).
Noise = stochastic variation in new tweet arrivals.

This is still a valid way to quantify basin stability, if you reinterpret the “drift” as growth of density in a region rather than movement of points.

Need to ensure regularisation happens the appropriate amount.

Ensure you quantify basin stability probabilistically, e.g., fraction of simulations where density increases near the basin center.

Regularization should be tuned via predictive performance on next time step, not arbitrary smoothing.

```



For all approaches: compare the findings to randomly sampling from all my tweets (ignoring datetime they were posted), which should be just noise (while maintaining the overall shape of my dataset for good comparisons)





## More to do: prioritise by what is going to help customers achieve their memedriving options
TODO: consider dimensions I might look in where I'm likely to find basins & reduce to them. One might be 'topics' ie the subject of the idea (which may or may not involve reducing the embeddings down from 768d)

Figure out better dimensions for good faith: sincerity and charity are fine, but things can be low on both and just funny, which is also good for the discourse!
could do 'good faith' then add confidence based on how much info, eg 'hahaha' without context cant be interpreted either way

TODO: plot animation for 'good faith' within a specific low-level topic, where people are more likely to know each other and have communities form

TODO: look at v popular tweets: are there any conditions prior to them taking off which suggest high receptivity to that idea?

TODO: plot animation as weather system. Get weather-like formulae for this.

TODO: make some physics-inspired model for impact of a single meme, and explaining trajectory of a meme, and predicting future memespace (and counterfactuals if more memes were to be added beyond those expected, perhap via some kind of monte carlo simulation)

TODO (this is bigger so for later): explore deep learning methods, eg spatio-temporal transformer, ideally powered by larger 1.3bn tweet dataset




## On searching for basins

I want to search for basins in a 3d space, which is a time series of tweets over several years. I can say the datetime each tweet was made, retweets_count and popularity of tweet.

I want a python script which searches for basins and gives me info (quant) on whether they exist. Give me several good approaches - no code at this stage.

When you do make the code borrow these to read in the file: Saved merged file to /Users/adambricknail/Desktop/memedrive_experiments/output_data/community_archive_good_faith_embeddings.csv
which has a row for each tweet, importance by retweet and favourite, and 'y'=sincerity, 'x'=charity
```
df shape: (2196452, 12)
df cols: Index(['id', 'full_text', 'created_at', 'retweet_count', 'favorite_count',
    'in_reply_to_status_id', 'in_reply_to_screen_name', 'screen_name',
    'username', 'datetime', 'x', 'y'],
    dtype='object')
```


I expect that tweet's effects last for a certain amount of time before disapearing, likely with some kind of decay curve

One way to test if basins exist:
1. Instead of user trajectories, look at how discourse begets discourse: Do tweets in spacial region A tend to generate replies in region A? Real basin: tweets "pull" responses toward the basin center # Fake basin: responses are random relative to original position


basins can be of different levels of strength (with some minimum threshold before they count).

Lyapunov stability analysis determines whether a dynamical system's equilibrium point is stable under small perturbations. Think of it like testing whether a marble in a bowl will return to the bottom if you nudge it slightly. How it works:
- Find an "energy" function - something that measures how far you are from equilibrium, like height in the bowl
- Check if this "energy" always decreases - does the marble always roll downhill toward the bottom?
- If yes, the system is stable - perturbations get smaller over time
Q: will this work for noisy systems where a basin still exists under the noise, but weaker?
A: The classical test breaks down, but stochastic extensions exist that are way more complex and give probabilistic guarantees instead of certainty (claude offered to code up a stochastic_lyapunov_test() for me). Many systems are ‘stable enough’ rather than ‘truly stable’

I wouldn’t be surprised if basins aren’t often very small (to point basin may not be the best model - tho they may be!) If I need to make arbitrary cutoffs as to what counts as a basin, consider the cutoff could be considered a hyper parameter in an ‘end to end model’ (ie one which includes pre-proc stages) in predicting movement of discourse

What would constitute "real" basins:
1. Temporal stability - basins persist over time
2. Flow convergence - trajectories converge toward basin centers
3. Escape barriers - it takes "energy" to leave basins
4. Non-random structure - basins aren't just artifacts of clustering algorithm

To consider: The basin detection might be better framed as "persistent interaction patterns" rather than "gravitational attractors."



Reasons the search for ‘good faith’ basins might fail:
1. Semantic spaces might not have genuine dynamics (just static similarity)
2. User behavior might be too noisy to detect coherent flow
3. Faith dimensions might not be "physical" enough for dynamical analysis

See last few back and forth in Claude chat ‘semantic basins and idea evolution’ on predicting effects of a single tweet, and the effects of more coordinated tweets


A conservative force field is one where the work done moving a particle from point A to point B depends only on the endpoints, not on the path taken. Reasons memespace is probably non-conservative:
* External events inject energy (news cycles, political events)
* History matters (what was said before affects what comes next)
* Network effects (who's connected to whom)
* Algorithmic amplification
However in practice I imagine it’s not a binary classification of conversation force field or otherwise, but degrees of both (eg there may be some basin or pull which dynamically forms, while also a bunch of the above factors are also in play). A better analogy: storm systems - they form around events, persist, then dissipate


If memes need opposing views to survive (to coexist with), how do I factor that into my model? Would ideally find these memes, and other classes of memes (eg the ‘antimemes’ which dissipate the power of other memes)
