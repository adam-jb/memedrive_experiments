
Experiments on looking at 'good faith' in embeddings space

Relies on 'preproc/embed_community_archive.py' being run already

## Files

First two can combine to BE A GENERALISABLE TOOL: reducing embedding dimensions to a few dimensions concerning something of interest (eg Good Faith, or a certain topic)

get_good_faith_ratings.py gets LLM-based assessments of how good faith the tweets are.
- Uses 3 dimensions to define what 'good faith' is ('sincerity', 'charity', 'constructiveness') rated on scale of 1-7 by an AI. Arrived at these via discussion with Claude, which could refer to if open up the discussion of how to define Good Faith again: https://claude.ai/public/artifacts/e74e73b9-dd6f-41c3-bc5c-1ae9bf638755

create_transformation_matrix.py
 - Use Canonical correlation analysis to make a matrix to transform embeddings to 3d which are the ‘good faith’ where the 3 dimensions sincerity, charity, constructiveness.
 - Apply transformation matrix to all 5.5m tweet embeddings

TODO: check good faith dimensions looks right for the tweets (ie is the CCA model trained on 25k tweets generalising well to 5.5m)

TODO: plot animation as weather system

TODO: algorithmically look for 'good faith basins' (want a 'basin identification' tool which generalises)

TODO: plot in general 2d space, measuring good faith some other way




## On searching for basins

Things we might do when the above is done:
1. Instead of user trajectories, look at how discourse begets discourse Do tweets in faith region A tend to generate replies in region B? # Real basin: tweets "pull" responses toward the basin center # Fake basin: responses are random relative to original position

A conservative force field is one where the work done moving a particle from point A to point B depends only on the endpoints, not on the path taken. Reasons memespace is probably non-conservative:
* External events inject energy (news cycles, political events)
* History matters (what was said before affects what comes next)
* Network effects (who's connected to whom)
* Algorithmic amplification
However in practice I imagine it’s not a binary classification of conversation force field or otherwise, but degrees of both (eg there may be some basin or pull which dynamically forms, while also a bunch of the above factors are also in play). A better analogy: storm systems - they form around events, persist, then dissipate

basins can also be of different levels of strength (with some minimum threshold before they count).

Lyapunov stability analysis determines whether a dynamical system's equilibrium point is stable under small perturbations. Think of it like testing whether a marble in a bowl will return to the bottom if you nudge it slightly. How it works:
- Find an "energy" function - something that measures how far you are from equilibrium, like height in the bowl
- Check if this "energy" always decreases - does the marble always roll downhill toward the bottom?
- If yes, the system is stable - perturbations get smaller over time
Q: will this work for noisy systems where a basin still exists under the noise, but weaker?
A: The classical test breaks down, but stochastic extensions exist that are way more complex and give probabilistic guarantees instead of certainty (claude offered to code up a stochastic_lyapunov_test() for me). Many systems are ‘stable enough’ rather than ‘truly stable’

I wouldn’t be surprised if basins aren’t often very small (to point basin may not be the best model - tho they may be!) If I need to make arbitrary cutoffs as to what counts as a basin, consider the cutoff could be considered a hyper parameter in an ‘end to end model’ (ie one which includes pre-proc stages) in predicting movement of discourse

If memes need opposing views to survive (to coexist with), how do I factor that into my model? Would ideally find these memes, and other classes of memes (eg the ‘antimemes’ which dissipate the power of other memes)

What would constitute "real" basins:
1. Temporal stability - basins persist over time
2. Flow convergence - trajectories converge toward basin centers
3. Escape barriers - it takes "energy" to leave basins
4. Non-random structure - basins aren't just artifacts of clustering algorithm

Reasons the search for ‘good faith’ basins might fail:
1. Semantic spaces might not have genuine dynamics (just static similarity)
2. User behavior might be too noisy to detect coherent flow
3. Faith dimensions might not be "physical" enough for dynamical analysis

To consider: The basin detection might be better framed as "persistent interaction patterns" rather than "gravitational attractors."

See last few back and forth in Claude chat ‘semantic basins and idea evolution’ on predicting effects of a single tweet, and the effects of more coordinated tweets
