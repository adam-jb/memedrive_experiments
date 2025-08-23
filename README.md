
## Next up

Plot be interactive

Clusters over time

See if can make algo to identify 'new ideas' and their paths

See if can predict and test changes in idea fronts (similar to how one predicts the weather)

See if can find 'semantic basins' (ie attractors for certain topics) and what their topology is, algorithmically.

Look for basins of particular things, such as 'high trust', and how people move towards or away from these




## Ideas
View movement of 'the conversation' inc topics as per embeddings

Can see what started 'change in topic' or 'movement'? Will need a way to quantify what I mean by these

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

preproc/plotly_interactive.py: makes 500mb html file which animates all tweets and is interactive


## Main blockers to immense success

Getting big and comprehensive data on discourse (eg even a sample of lots of Twitter data v expensive to do via the official API)




## Placeholder code

From claude on looking for 'real' dynamic basins. Suggests using these metrics (not sure if it made those up or legit):
Depth - difference between rim potential and center potential
Volume - proportion of space the basin occupies
Coherence - how circular vs elongated the basin is
Stability - resistance to perturbations (Lyapunov stability)
Population - number of users in the basin
Combined strength score - multiplies all these together

Explanation from Claude, to figure out later:
DynamicBasinModel.fit_temporal_landscape()
Purpose: Main orchestrator that fits basin landscapes across all time windows
Steps:

Slide time window across dataset (e.g., 2-week windows with 1-week overlap)
For each window: extract coordinates, fit potential field, detect basins
Track basin identity across windows using genealogy system
Store complete landscape + basin metadata for each time point
Skip windows with insufficient data (<100 tweets)

Key Output: landscapes dict mapping timestamps to {potential_field, basins, basin_ids}

_detect_significant_basins()
Purpose: Find local minima in potential field and filter by strength threshold
Steps:

find_local_minima(potential_field) - identifies candidate basin centers
For each candidate: compute watershed region (basin catchment area)
Calculate strength using the 5-component metric above
Only keep basins where strength > min_strength_threshold
Package results with center coordinates, masks, strength scores

Key Filter: Eliminates weak/spurious clusters that aren't real discourse attractors

_match_basins_to_history()
Purpose: Track basin identity across consecutive time windows
Steps:

First time window: assign fresh IDs (basin_0_gen_0, basin_1_gen_0, etc.)
Subsequent windows: compute overlap between current basins and previous basins
compute_basin_overlap() - typically Jaccard similarity between basin masks
If overlap > 30%: inherit previous basin ID (continuing basin)
If no good match: assign new ID (basin birth)
Handles basin mergers/splits implicitly through overlap logic

Key Output: Genealogy mapping that preserves basin identity through transformations

analyze_basin_dynamics()
Purpose: Detect lifecycle events by comparing consecutive time windows
Events Detected:

Births: New basin IDs appearing in genealogy
Deaths: Basin IDs disappearing from genealogy
Strength changes: Continuing basins with >20% strength change
Mergers/Splits: Multiple basins mapping to one, or vice versa

Implementation: Set operations on basin ID collections + strength difference calculations
Key Output: Timestamped event log for all basin transformations

correlate_with_external_drivers()
Purpose: Test if basin events cluster temporally near external events
Driver Types:

Major news events
Viral content (high-engagement tweets)
Policy announcements
Platform algorithm changes

Statistical Test:

For each basin event, find minimum distance to any external event
Compare distribution of distances to null model (random timing)
Use Mann-Whitney U test to detect significant temporal clustering
Lag window (e.g., 7 days) allows for delayed effects

Key Output: P-values and effect sizes for each (basin_event_type, external_driver) pair

build_basin_prediction_model()
Purpose: Train ML model to forecast future basin dynamics
Feature Engineering:

Current state: basin strengths, counts, spatial distribution
Trends: strength changes, formation/dissolution rates over recent windows
External signals: recent news/viral events based on correlation analysis
Network topology: user interaction patterns, community structure
Discourse velocity: rate of movement through faith space

Targets to Predict:

Probability of new basin formation in next window
Dissolution risk for existing basins
Expected strength changes for continuing basins

Model Type: Multi-task learner (predicts multiple related targets simultaneously)

validate_basin_model()
Purpose: Comprehensive model evaluation with multiple validation approaches
Validation Types:

Time series CV: Train on past, predict future (respects temporal ordering)
Null comparison: Beat baseline of "always predict historical average"
Sensitivity analysis: How robust are results to parameter changes?
Task-specific accuracy: Separate metrics for formation, dissolution, strength prediction

Key Metrics:

Cross-validation scores across time splits
Improvement over null model baseline
Feature importance rankings
Prediction accuracy by forecast horizon

Output: Comprehensive validation report assessing model reliability and usefulness
Each function handles one piece of the pipeline: temporal fitting → basin detection → identity tracking → event analysis → external correlation → prediction → validation.

```python
def compute_basin_strength(coordinates, basin_center, basin_mask):
    """Multi-dimensional basin strength metric"""
    # Depth: how much potential energy difference from rim to center
    rim_potential = np.max(potential_values[basin_boundary])
    center_potential = potential_values[basin_center]
    depth = rim_potential - center_potential
    # Volume: how much of faith space the basin occupies
    volume = np.sum(basin_mask) / total_space_volume
    # Coherence: how "circular" vs "elongated" the basin is
    eigenvals = np.linalg.eigvals(np.cov(coordinates[basin_mask].T))
    coherence = np.min(eigenvals) / np.max(eigenvals)  # 1 = circular, 0 = linear
    # Stability: resistance to perturbations
    stability = measure_lyapunov_stability(coordinates[basin_mask])
    # Population: how many users actually occupy the basin
    population = np.sum(basin_mask)
    # Combined strength score
    strength = (depth * volume * coherence * stability * np.log(population))
    return strength
class DynamicBasinModel:
    def init(self, min_strength_threshold=0.1):
        self.min_strength = min_strength_threshold
        self.basin_history = {}
        self.external_drivers = {}
    def fit_temporal_landscape(self, trajectories, timestamps, window_size='2weeks'):
        """Fit time-varying potential with basin tracking"""
        landscapes = {}
        basin_genealogy = {}  # Track basin identity across time
        for t in time_windows(timestamps, window_size):
            # Fit potential in this window
            window_coords = get_window_data(trajectories, timestamps, t, window_size)
            if len(window_coords) < 100:
                landscapes[t] = None
                continue
            # Compute potential field
            potential_field = self._fit_potential_field(window_coords)
            # Detect basins above strength threshold
            basins = self._detect_significant_basins(potential_field, window_coords)
            # Track basin identity across time
            basin_genealogy[t] = self._match_basins_to_history(basins, basin_genealogy)
            landscapes[t] = {
                'potential_field': potential_field,
                'basins': basins,
                'basin_ids': basin_genealogy[t]
            }
        return landscapes, basin_genealogy
    def detectsignificant_basins(self, potential_field, coordinates):
        """Only count basins above strength threshold"""
        # Find all local minima in potential
        candidate_basins = find_local_minima(potential_field)
        significant_basins = []
        for basin_center in candidate_basins:
            # Compute basin properties
            basin_mask = self._get_basin_watershed(potential_field, basin_center)
            strength = compute_basin_strength(coordinates, basin_center, basin_mask)
            if strength > self.min_strength:
                significant_basins.append({
                    'center': basin_center,
                    'mask': basin_mask,
                    'strength': strength,
                    'properties': self._compute_basin_properties(coordinates, basin_mask)
                })
        return significant_basins
def matchbasins_to_history(self, current_basins, genealogy):
    """Track basin identity across time windows"""
    if not genealogy:  # First time step
        return {i: f"basin_{i}_gen_0" for i in range(len(current_basins))}
    previous_basins = genealogy[max(genealogy.keys())]
    basin_matches = {}
    for i, current_basin in enumerate(current_basins):
        best_match = None
        best_overlap = 0
        # Find best overlap with previous basins
        for prev_id, prev_basin in previous_basins.items():
            overlap = compute_basin_overlap(current_basin, prev_basin)
            if overlap > best_overlap and overlap > 0.3:  # 30% overlap threshold
                best_overlap = overlap
                best_match = prev_id
        if best_match:
            basin_matches[i] = best_match  # Continuing basin
        else:
            basinmatches[i] = f"basin{i}gen{len(genealogy)}"  # New basin
    return basin_matches
def analyze_basin_dynamics(landscapes, genealogy):
    """Comprehensive basin lifecycle analysis"""
    events = {
        'births': [],
        'deaths': [],
        'mergers': [],
        'splits': [],
        'strength_changes': []
    }
    for t1, t2 in consecutive_windows(landscapes.keys()):
        if landscapes[t1] is None or landscapes[t2] is None:
            continue
        # Detect basin births (new IDs appearing)
        new_ids = set(genealogy[t2].values()) - set(genealogy[t1].values())
        if new_ids:
            events['births'].append({
                'time': t2,
                'new_basins': new_ids,
                'trigger': identify_trigger(t1, t2)
            })
        # Detect basin deaths (IDs disappearing)
        lost_ids = set(genealogy[t1].values()) - set(genealogy[t2].values())
        if lost_ids:
            events['deaths'].append({
                'time': t2,
                'lost_basins': lost_ids,
                'cause': identify_dissolution_cause(t1, t2, lost_ids)
            })
        # Detect strength changes
        for basin_id in set(genealogy[t1].values()) & set(genealogy[t2].values()):
            strength_change = measure_strength_change(basin_id, t1, t2, landscapes)
            if abs(strength_change) > 0.2:  # Significant change
                events['strength_changes'].append({
                    'time': t2,
                    'basin_id': basin_id,
                    'strength_delta': strength_change
                })
    return events
def correlate_with_external_drivers(events, external_data):
    """Link basin dynamics to external events"""
    drivers = {
        'news_events': external_data['major_news'],
        'viral_content': external_data['viral_tweets'],
        'policy_changes': external_data['policy_announcements'],
        'platform_changes': external_data['algorithm_updates']
    }
    correlations = {}
    for event_type, event_list in events.items():
        correlations[event_type] = {}
        for driver_name, driver_events in drivers.items():
            # Test temporal correlation
            correlation = test_temporal_correlation(
                event_list,
                driver_events,
                max_lag=timedelta(days=7)
            )
            correlations[event_type][driver_name] = {
                'correlation': correlation,
                'p_value': correlation.p_value,
                'significant_pairs': correlation.significant_matches
            }
    return correlations
def build_basin_prediction_model(landscapes, events, correlations):
    """Predict future basin formation/dissolution"""
    # Features for prediction
    features = {
        'current_basin_strengths': extract_current_strengths(landscapes),
        'trend_directions': compute_strength_trends(landscapes),
        'external_signals': encode_external_drivers(correlations),
        'network_topology': measure_user_network_changes(),
        'discourse_velocity': compute_discourse_velocity_field()
    }
    # Targets to predict
    targets = {
        'new_basin_probability': predict_basin_formation_probability(),
        'dissolution_risk': predict_basin_dissolution_risk(),
        'strength_changes': predict_strength_evolution()
    }
    # Multi-task learning model
    model = MultiTaskBasinPredictor()
    model.fit(features, targets)
    return model
def validate_basin_model(model, test_data):
    """Comprehensive model validation"""
    # Cross-validation on time series
    cv_scores = time_series_cross_validate(model, test_data)
    # Null model comparison
    null_model = RandomBasinModel()
    null_scores = null_model.evaluate(test_data)
    # Sensitivity analysis
    sensitivity = test_parameter_sensitivity(model, test_data)
    # Prediction accuracy
    accuracy = {
        'basin_formation': test_formation_predictions(model, test_data),
        'dissolution_timing': test_dissolution_predictions(model, test_data),
        'strength_evolution': test_strength_predictions(model, test_data)
    }
    return {
        'cross_validation': cv_scores,
        'null_comparison': cv_scores - null_scores,
        'sensitivity': sensitivity,
        'prediction_accuracy': accuracy
    }
  ```
