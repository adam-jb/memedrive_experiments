#!/usr/bin/env python3
"""
Ultra Basin Analyzer - Forward Predictive Tuning + Probabilistic Continuation + Early Detection

Changes vs previous version:
 - Hyperparameter selection is done by forward-time predictive skill (default: continuation AUROC),
   NOT static silhouette. Forward chaining with rolling evaluation.
 - Consensus bootstraps use per-tweet sampling probabilities combining popularity and within-bin recency:
     w = (1 + log(1 + RT + α * Fav)) * exp(-λ * Δdays_to_bin_end)
 - Probabilistic continuation via soft next-bin matching:
     s_ij = exp(-d_ij / τ) * conf_i * conf_j * prior_j ; s_iØ = exp(-γ)
     p_ij = s_ij / (s_iØ + Σ_j s_ij); p_cont(i) = 1 - p_iØ
 - Expected next size: E[size_{t+1} | i] = Σ_j p_ij * size_j (null contributes 0).
 - Early detection ON by default: tuned over a small grid of (N, S, T-quantile, k@) with logistic ranking
   and reported precision@k and a kappa-like lift: (P@k - π) / (1 - π), where π is prevalence among candidates.
 - CSV outputs for tuning metrics and final basin summaries.

Requirements:
 - Python 3.8+
 - numpy, pandas, scikit-learn, scipy, hdbscan


## to run with fewer bootstraps to test
python cluster_tracking3.py --n_boot=5

"""

from __future__ import annotations

import argparse
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.cluster import AgglomerativeClustering
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    silhouette_score,  # kept for quick diagnostics only (not used for tuning)
    roc_auc_score,
    log_loss,
    f1_score,
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

try:
    import hdbscan
except Exception:
    hdbscan = None

# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("UltraBasin")

# ------------------------------------------------------------------------------
# Defaults / Config
# ------------------------------------------------------------------------------

INPUT_FILE = "~/Desktop/memedrive_experiments/output_data/basin_finder/dummy_tweet_embeddings.csv"
OUTPUT_DIR = "~/Desktop/memedrive_experiments/output_data/basin_finder/"

HDBSCAN_min_cluster_size_global = 10  # must be an int

DEFAULT_CONFIG: Dict[str, Any] = {
    # Temporal binning
    "bin_size_days": 7,  # default; tuner will explore grid below
    "bin_size_candidates": [5, 7, 14],

    # Consensus / bootstrapping
    "n_bootstrap": 30,
    "consensus_max_clusters": None,  # optional cap; else heuristic
    "min_cluster_frac": 0.01,  # adaptive per-bin minimum (as frac of bin points)

    # Engagement & Recency weighting (used in consensus bootstrapping)
    "engagement_alphas": [0.5, 1.0, 2.0],
    "recency_decay_mode": "grid",  # always grid in tuner
    # distance softmatch parameters
    "softmatch_gamma": 1.0,   # s_iØ = exp(-gamma)
    "softmatch_use_prior": True,  # prior_j = size_j / sum(size)
    "softmatch_temperature_mode": "median",  # τ from median pairwise distances
    "blend_soft_and_model_lambda": 0.0,  # 0.0 = soft only; (kept for future)

    # Tracking
    "max_distance_threshold": 1.5,  # for hard Hungarian acceptance (unchanged)
    "min_track_length": 2,

    # Early detection (ON by default; tuned on a tiny grid)
    "early_enabled": True,
    "early_N_candidates": [2, 3],
    "early_S_candidates": [1, 2],
    "early_Tq_candidates": [0.5, 0.6],  # quantile for sustained threshold
    "early_k_frac_candidates": [0.05, 0.1],  # k = frac * (#fledgling candidates)
    # fledgling threshold uses max(adaptive_min_cluster_size, bin_size_q25)

    # Objective & tie-breakers
    "objective_metric": "continuation_auroc",  # main
    # tie-break order if objective draws: brier (smaller), logloss (smaller), size_mae (smaller), early_kappa_like (larger)
    "train_split": 0.6,
    "random_state": 42,

    # Performance
    "verbose_timings": True,
    "max_combos": None,  # optionally cap the big grid; None = full grid
}

# ------------------------------------------------------------------------------
# Data classes
# ------------------------------------------------------------------------------

@dataclass
class ClusterSummary:
    bin_id: int
    cluster_local_id: int
    center: np.ndarray
    size: int
    members_idx: np.ndarray  # indices within the bin df
    probability: float = 1.0  # consensus confidence
    # Soft continuation (to be filled per t): will be computed during evaluation, not stored here


@dataclass
class Basin:
    basin_id: int
    timeline: List[ClusterSummary] = field(default_factory=list)
    probabilities: List[float] = field(default_factory=list)
    duration: int = 0
    avg_size: float = 0.0
    total_size: int = 0
    size_stability: float = 0.0
    trend_slope: float = 0.0
    trend_r2: float = 0.0
    predicted_direction: str = "unknown"
    confidence: float = 0.0


# ------------------------------------------------------------------------------
# Utility metrics (safe versions)
# ------------------------------------------------------------------------------

def safe_roc_auc(y_true: List[int], y_prob: List[float]) -> float:
    try:
        y = np.asarray(y_true)
        if len(np.unique(y)) < 2:
            return np.nan
        return float(roc_auc_score(y, np.asarray(y_prob)))
    except Exception:
        return np.nan

def safe_log_loss(y_true: List[int], y_prob: List[float]) -> float:
    try:
        y = np.asarray(y_true)
        p = np.clip(np.asarray(y_prob), 1e-6, 1.0 - 1e-6)
        if len(np.unique(y)) < 2:
            return np.nan
        return float(log_loss(y, p))
    except Exception:
        return np.nan

def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    if len(y_true) == 0 or k <= 0:
        return np.nan
    idx = np.argsort(-y_score)[:k]
    return float(np.mean(y_true[idx])) if k > 0 else np.nan

# ------------------------------------------------------------------------------
# Analyzer
# ------------------------------------------------------------------------------

class UltraBasinAnalyzer:
    def __init__(self, input_file: str, output_dir: str = "./", config: Optional[Dict[str, Any]] = None):
        if hdbscan is None:
            raise ImportError("hdbscan is required but not installed. Install via `pip install hdbscan`.")

        self.input_file = Path(input_file).expanduser()
        self.output_dir = Path(output_dir).expanduser()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.config = DEFAULT_CONFIG.copy()
        if config:
            self.config.update(config)

        self.df: Optional[pd.DataFrame] = None
        self.embedding_cols: List[str] = []
        self.random_state = int(self.config.get("random_state", 42))
        np.random.seed(self.random_state)

        self.timings: Dict[str, float] = {}

    # --------------------------- I/O & Preparation ------------------------- #

    def load_data(self):
        t0 = time.time()
        logger.info("Loading data from %s", self.input_file)
        df = pd.read_csv(self.input_file)
        if "datetime" not in df.columns:
            raise ValueError("Input CSV must contain a 'datetime' column.")
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime").reset_index(drop=True)
        self.df = df
        self.embedding_cols = [c for c in df.columns if c.startswith("e")]
        if not self.embedding_cols:
            raise ValueError("No embedding columns found (expected columns starting with 'e').")
        logger.info("Loaded %d rows / %d embedding dims", len(df), len(self.embedding_cols))
        self.timings["load_data"] = time.time() - t0

    def temporal_split(self):
        t0 = time.time()
        assert self.df is not None
        split_idx = int(len(self.df) * float(self.config.get("train_split", 0.6)))
        train_df = self.df.iloc[:split_idx].reset_index(drop=True)
        test_df = self.df.iloc[split_idx:].reset_index(drop=True)
        logger.info("Temporal split -> Train: %d rows, Test: %d rows", len(train_df), len(test_df))
        self.timings["temporal_split"] = time.time() - t0
        return train_df, test_df

    # --------------------------- Binning ----------------------------------- #

    def _create_bins(self, data: pd.DataFrame, bin_size_days: int) -> List[pd.DataFrame]:
        """
        Create temporal bins of fixed size; inject 'bin_id', 'bin_start', 'bin_end' columns into each bin df.
        """
        t0 = time.time()
        start_date = data["datetime"].min()
        end_date = data["datetime"].max()
        bins = []
        current = start_date
        bin_id = 0
        while current < end_date:
            nxt = current + pd.Timedelta(days=bin_size_days)
            mask = (data["datetime"] >= current) & (data["datetime"] < nxt)
            bin_df = data.loc[mask].reset_index(drop=True)
            if len(bin_df) > 0:
                bin_df = bin_df.copy()
                bin_df["bin_id"] = bin_id
                bin_df["bin_start"] = current
                bin_df["bin_end"] = nxt
                bins.append(bin_df)
                bin_id += 1
            current = nxt
        self.timings["_create_bins"] = time.time() - t0
        return bins

    # ---------------- Consensus clustering per bin with weighted bootstrap -- #

    def _consensus_clusters_for_bin(self, bin_df: pd.DataFrame, alpha: float, recency_lambda: float,
                                    min_cluster_size_global: int) -> List[ClusterSummary]:
        """
        Bootstrap + HDBSCAN consensus with popularity + recency sampling.
        """
        n_boot = int(self.config.get("n_bootstrap", 30))
        n = len(bin_df)
        if n == 0:
            return []

        X = bin_df[self.embedding_cols].values
        scaler = StandardScaler().fit(X)
        Xs = scaler.transform(X)

        # Popularity weight w_pop = 1 + log(1 + RT + α * Fav)
        rt = bin_df.get("retweet_count", pd.Series(0, index=bin_df.index)).fillna(0).astype(float).values
        fav = bin_df.get("favourite_count", pd.Series(0, index=bin_df.index)).fillna(0).astype(float).values
        w_pop = 1.0 + np.log1p(np.maximum(0.0, rt) + alpha * np.maximum(0.0, fav))

        # Recency decay within bin relative to bin_end
        dt = (pd.to_datetime(bin_df["bin_end"]).values.astype("datetime64[ns]") -
              pd.to_datetime(bin_df["datetime"]).values.astype("datetime64[ns]")).astype("timedelta64[s]").astype(np.float64) / 86400.0
        decay = np.exp(-recency_lambda * dt)

        probs = w_pop * decay
        if not np.isfinite(probs).all() or probs.sum() <= 0:
            probs = np.ones(n)
        probs = probs / probs.sum()

        rng = np.random.RandomState(self.random_state)
        coassign = np.zeros((n, n), dtype=float)
        total_runs = 0

        for _ in range(n_boot):
            logger.info(f"Boostrap: {_}")

            sample_idx = rng.choice(np.arange(n), size=n, replace=True, p=probs)
            X_sample = Xs[sample_idx]
            try:
                # Adaptive min cluster size per run (cap to sample size)
                min_cs_run = max(2, min(min_cluster_size_global, max(2, int(len(sample_idx) * 0.1))))
                clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cs_run)
                labels_sample = clusterer.fit_predict(X_sample)
            except Exception:
                continue
            total_runs += 1
            labels_full = np.full(n, -1, dtype=int)
            for j, orig_idx in enumerate(sample_idx):
                labels_full[orig_idx] = int(labels_sample[j])
            for lbl in np.unique(labels_full):
                if lbl == -1:
                    continue
                members = np.where(labels_full == lbl)[0]
                if members.size > 0:
                    coassign[np.ix_(members, members)] += 1.0

        if total_runs == 0:
            logger.warning("No successful bootstrap runs for bin %s", int(bin_df["bin_id"].iloc[0]))
            return []

        coassign = coassign / float(total_runs)
        dist = 1.0 - coassign

        #### IDEALLY WOULD AVOID THIS TAKES AGES. Replaced it with just one line:
        X_repr = Xs
        # print('Low-dim representation (SVD)')
        # try:
        #     M = dist - dist.mean(axis=0)[None, :]
        #     U, S, _ = np.linalg.svd(M, full_matrices=False)
        #     repr_dim = min(8, max(1, n - 1))
        #     X_repr = U[:, :repr_dim] * S[:repr_dim]
        # except Exception:
        #     X_repr = Xs


        k_hint = self.config.get("consensus_max_clusters")
        if k_hint is None:
            k = max(1, int(np.sqrt(n) / 3))
            k = min(k, n)
        else:
            k = max(1, min(int(k_hint), n))

        try:
            agg = AgglomerativeClustering(n_clusters=k)
            cons_labels = agg.fit_predict(X_repr)
        except Exception:
            cons_labels = np.zeros(n, dtype=int)

        clusters: List[ClusterSummary] = []
        for lbl in np.unique(cons_labels):
            members = np.where(cons_labels == lbl)[0]
            if members.size == 0:
                continue
            center = X[members].mean(axis=0)  # back in original X scale
            if members.size == 1:
                prob = 1.0
            else:
                sub = coassign[np.ix_(members, members)]
                prob = float(np.mean(sub))
            clusters.append(ClusterSummary(
                bin_id=int(bin_df["bin_id"].iloc[0]),
                cluster_local_id=len(clusters),
                center=center,
                size=int(members.size),
                members_idx=members,
                probability=prob
            ))

        print(f'Got clusters of len {len(clusters)}')

        return clusters

    # ---------------------- Tracking across bins ---------------------------- #

    def _track_clusters_probabilistic(self, all_clusters: List[ClusterSummary]) -> List[Basin]:
        """
        Hard tracking (Hungarian) for discrete trajectories, independent of softmatch.
        """
        # group by bin
        bins: Dict[int, List[ClusterSummary]] = {}
        for c in all_clusters:
            bins.setdefault(c.bin_id, []).append(c)
        sorted_bins = sorted(bins.keys())

        basins: List[Basin] = []
        next_basin_id = 0

        for i, bin_id in enumerate(sorted_bins):
            curr = bins[bin_id]
            if i == 0:
                for c in curr:
                    basins.append(Basin(basin_id=next_basin_id, timeline=[c], probabilities=[c.probability]))
                    next_basin_id += 1
                continue

            prev_bin_id = sorted_bins[i - 1]
            prev = bins[prev_bin_id]
            if not prev:
                for c in curr:
                    basins.append(Basin(basin_id=next_basin_id, timeline=[c], probabilities=[c.probability]))
                    next_basin_id += 1
                continue

            prev_centers = np.vstack([c.center for c in prev])
            prev_probs = np.array([c.probability for c in prev])
            curr_centers = np.vstack([c.center for c in curr])
            curr_probs = np.array([c.probability for c in curr])

            distances = cdist(prev_centers, curr_centers)
            prob_factor = np.outer(1.0 / (prev_probs + 1e-8), 1.0 / (curr_probs + 1e-8))
            cost = distances * prob_factor

            row_ind, col_ind = linear_sum_assignment(cost)
            matched_prev = set()
            matched_curr = set()

            for r, c_idx in zip(row_ind, col_ind):
                threshold = self.config.get("max_distance_threshold", 1.5)
                scale = 1.0 + (1.0 - prev_probs[r]) + (1.0 - curr_probs[c_idx])
                if distances[r, c_idx] <= threshold * scale:
                    # append curr[c_idx] to the basin whose last element is prev[r]
                    # find the basin with last cluster == prev[r]
                    for b in basins:
                        if b.timeline and b.timeline[-1] is prev[r]:
                            b.timeline.append(curr[c_idx])
                            b.probabilities.append(curr[c_idx].probability)
                            matched_prev.add(r)
                            matched_curr.add(c_idx)
                            break

            for idx_c, c in enumerate(curr):
                if idx_c not in matched_curr:
                    basins.append(Basin(basin_id=next_basin_id, timeline=[c], probabilities=[c.probability]))
                    next_basin_id += 1

        # finalize stats + filter
        final_basins: List[Basin] = []
        for b in basins:
            b.duration = len(b.timeline)
            b.avg_size = float(np.mean([t.size for t in b.timeline])) if b.duration > 0 else 0.0
            b.total_size = int(np.sum([t.size for t in b.timeline])) if b.duration > 0 else 0
            sizes = np.array([t.size for t in b.timeline], dtype=float) if b.duration > 0 else np.array([])
            if b.duration > 0 and np.mean(sizes) > 0:
                b.size_stability = float(1.0 - (np.std(sizes) / (np.mean(sizes) + 1e-10)))
            else:
                b.size_stability = 0.0

            b.confidence = float(np.mean(b.probabilities)) * min(1.0, b.duration / max(1, self.config.get("n_bootstrap", 30)))

            if b.duration >= 2:
                X = np.arange(b.duration).reshape(-1, 1)
                y = np.array([t.size for t in b.timeline], dtype=float)
                lr = LinearRegression().fit(X, y)
                slope = float(lr.coef_[0])
                preds = lr.predict(X)
                ss_res = np.sum((y - preds) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2) + 1e-10
                r2 = float(1.0 - ss_res / ss_tot)
                b.trend_slope = slope
                b.trend_r2 = r2
                if slope > 0 and b.confidence > 0.3:
                    b.predicted_direction = "strengthening"
                elif slope < 0 and b.confidence > 0.3:
                    b.predicted_direction = "weakening"
                else:
                    b.predicted_direction = "stable"
            else:
                b.trend_slope = 0.0
                b.trend_r2 = 0.0
                b.predicted_direction = "insufficient_data"

            if b.duration >= self.config.get("min_track_length", 2):
                final_basins.append(b)

        return final_basins

    # ---------------------- Soft next-bin matching -------------------------- #

    def _softmatch_probs(self, prev_clusters: List[ClusterSummary], next_clusters: List[ClusterSummary]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return:
          - p_matrix: shape [n_prev, n_next], probabilities of matching each prev cluster to each next cluster
          - p_null: shape [n_prev], probability of dying (no continuation) per prev cluster
        Uses s_ij = exp(-d_ij / τ) * conf_i * conf_j * prior_j
             s_iØ = exp(-γ)
        """
        if len(prev_clusters) == 0:
            return np.zeros((0, len(next_clusters))), np.zeros((0,))
        if len(next_clusters) == 0:
            return np.zeros((len(prev_clusters), 0)), np.ones((len(prev_clusters),))

        prev_centers = np.vstack([c.center for c in prev_clusters])
        next_centers = np.vstack([c.center for c in next_clusters])
        dists = cdist(prev_centers, next_centers)
        tau_mode = self.config.get("softmatch_temperature_mode", "median")
        if tau_mode == "median":
            tau = np.median(dists[np.isfinite(dists) & (dists > 0)]) if np.any(np.isfinite(dists) & (dists > 0)) else 1.0
        else:
            tau = 1.0
        tau = max(tau, 1e-6)

        prev_conf = np.array([c.probability for c in prev_clusters])
        next_conf = np.array([c.probability for c in next_clusters])
        gamma = float(self.config.get("softmatch_gamma", 1.0))

        if self.config.get("softmatch_use_prior", True):
            sizes = np.array([c.size for c in next_clusters], dtype=float)
            prior = sizes / (sizes.sum() + 1e-8)
        else:
            prior = np.ones(len(next_clusters)) / len(next_clusters)

        # Scores
        s = np.exp(-dists / tau) * (prev_conf[:, None]) * (next_conf[None, :]) * (prior[None, :])
        s_null = np.exp(-gamma) * np.ones((len(prev_clusters), 1))

        denom = s_null + s.sum(axis=1, keepdims=True)
        p_null = (s_null / denom).ravel()
        p = s / denom  # normalized per prev cluster
        return p, p_null

    # ---------------------- Feature builders -------------------------------- #

    def _bin_cluster_features(self, bins_clusters: Dict[int, List[ClusterSummary]],
                              bin_dfs: List[pd.DataFrame],
                              basins_by_key: Dict[Tuple[int, int], Tuple[int, int]]) -> Dict[Tuple[int, int], Dict[str, float]]:
        """
        Build features for cluster i at bin t.
        key = (bin_id, cluster_local_id)
        """
        features: Dict[Tuple[int, int], Dict[str, float]] = {}
        # precompute within-bin center matrices for density/shifts
        for bin_idx, clusters in bins_clusters.items():
            centers = np.vstack([c.center for c in clusters]) if clusters else np.zeros((0, 1))
            # nearest-center distance within same bin
            within_dists = cdist(centers, centers) if len(clusters) > 1 else np.zeros((len(clusters), len(clusters)))
            # set diag to inf to find nearest other
            if len(clusters) > 1:
                np.fill_diagonal(within_dists, np.inf)

            bin_df = bin_dfs[bin_idx]
            # compute average member popularity in THIS bin (uses α, λ of the tuned combo; already recomputed for this combo)
            rt = bin_df.get("retweet_count", pd.Series(0, index=bin_df.index)).fillna(0).to_numpy(float)
            fav = bin_df.get("favourite_count", pd.Series(0, index=bin_df.index)).fillna(0).to_numpy(float)
            # we don't know α, λ here anymore; they were applied during consensus sampling; for features we approximate with:
            # pop ~ 1 + log1p(rt + fav) (agnostic). Good enough as a proxy for ranking features.
            pop_proxy = 1.0 + np.log1p(np.maximum(0.0, rt) + np.maximum(0.0, fav))

            n_points = len(bin_df)
            n_clusters = len(clusters)
            bin_density = (n_clusters / max(1, n_points))

            for idx_c, c in enumerate(clusters):
                key = (bin_idx, c.cluster_local_id)
                # neighbor density: mean dist to nearest 5 clusters (or fewer)
                if n_clusters > 1:
                    row = within_dists[idx_c]
                    k = min(5, n_clusters - 1)
                    neigh_density = float(np.mean(np.sort(row)[:k]))
                else:
                    neigh_density = float("inf")
                avg_member_pop = float(np.mean(pop_proxy[c.members_idx])) if len(c.members_idx) > 0 else 1.0

                # center shift vs closest cluster in previous bin
                if (bin_idx - 1) in bins_clusters and len(bins_clusters[bin_idx - 1]) > 0:
                    prev_centers = np.vstack([pc.center for pc in bins_clusters[bin_idx - 1]])
                    shift = float(np.min(cdist(c.center[None, :], prev_centers)))
                else:
                    shift = float("inf")

                # age in basin + size trend_last2 from basins_by_key
                # basins_by_key maps (bin_id, cluster_local_id) -> (basin_id, position)
                age = 1.0
                trend_last2 = 0.0
                if key in basins_by_key:
                    basin_id, pos = basins_by_key[key]
                    # find sizes at pos and pos-1
                    # we don't have basins list here; but we can infer from neighbor existence in basins_by_key
                    # Simpler: age = pos + 1
                    age = float(pos + 1)
                    if pos >= 1:
                        # need size at t and t-1: can retrieve from bins_clusters
                        prev_key = (bin_idx - 1, None)
                        # find the cluster at bin_idx-1 with same basin pos-1:
                        # We don't have reverse index; approximate trend using shift and current size:
                        trend_last2 = 0.0
                # trend_last2 left as 0.0 to avoid heavy bookkeeping; age still informative.

                features[key] = {
                    "size_t": float(c.size),
                    "prob_t": float(c.probability),
                    "avg_member_pop_t": float(avg_member_pop),
                    "center_shift_t": float(shift if np.isfinite(shift) else 0.0),
                    "age_in_basin_t": float(age),
                    "bin_density_t": float(bin_density),
                    "neigh_density_t": float(neigh_density if np.isfinite(neigh_density) else 0.0),
                    "size_trend_last2": float(trend_last2),
                }
        return features

    # ---------------------- Forward-chaining evaluation --------------------- #

    def _evaluate_forward(self, bin_dfs: List[pd.DataFrame], all_clusters: List[ClusterSummary],
                          basins: List[Basin], bin_size_days: int,
                          alpha: float, recency_lambda: float,
                          early_grid: List[Tuple[int, int, float, float]]) -> Dict[str, Any]:
        """
        Build labels and compute metrics using SOFT continuation probabilities and expected sizes.
        Early detection tuned over a small grid; best early metrics are reported.
        """
        # group clusters by bin
        bins_clusters: Dict[int, List[ClusterSummary]] = {}
        for c in all_clusters:
            bins_clusters.setdefault(c.bin_id, []).append(c)
        sorted_bins = sorted(bins_clusters.keys())
        if len(sorted_bins) < 2:
            return {
                "continuation_auroc": np.nan, "continuation_brier": np.nan, "continuation_logloss": np.nan,
                "size_mae": np.nan, "size_rmse": np.nan, "direction_acc": np.nan, "direction_f1": np.nan,
                "early_p_at_k": np.nan, "early_kappa_like": np.nan,
                "best_early_N": None, "best_early_S": None, "best_early_Tq": None, "best_early_kfrac": None
            }

        # basins_by_key: (bin_id, cluster_local_id) -> (basin_id, position_in_basin)
        basins_by_key: Dict[Tuple[int, int], Tuple[int, int]] = {}
        for b in basins:
            for pos, c in enumerate(b.timeline):
                basins_by_key[(c.bin_id, c.cluster_local_id)] = (b.basin_id, pos)

        # features
        features_map = self._bin_cluster_features(bins_clusters, bin_dfs, basins_by_key)

        # Accumulators
        cont_y, cont_p = [], []
        size_y, size_pred = [], []
        dir_y, dir_pred = [], []

        # Precompute per pair soft matches
        soft_cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}  # t -> (p_matrix, p_null)

        for t_idx in range(len(sorted_bins) - 1):
            t = sorted_bins[t_idx]
            t_next = sorted_bins[t_idx + 1]
            prev_cl = bins_clusters.get(t, [])
            next_cl = bins_clusters.get(t_next, [])

            # Soft probs for this t -> t+1
            p_mat, p_null = self._softmatch_probs(prev_cl, next_cl)
            soft_cache[t] = (p_mat, p_null)

            # Continuation labels (ground truth): does basin of cluster at t have entry at t+1?
            # Also get actual next size (0 if none)
            key_next_present = set((c.bin_id, c.cluster_local_id) for c in next_cl)
            # Build mapping from basin id to set of bin_ids present for quick lookup
            basin_presence: Dict[int, set] = {}
            basin_size_by_key: Dict[Tuple[int, int], int] = {}
            for b in basins:
                bid = b.basin_id
                s = set()
                for c in b.timeline:
                    s.add(c.bin_id)
                    basin_size_by_key[(c.bin_id, c.cluster_local_id)] = c.size
                basin_presence[bid] = s

            # For each prev cluster i, compute labels and soft predictions
            for i, c_prev in enumerate(prev_cl):
                key_prev = (t, c_prev.cluster_local_id)
                # label: continuation
                if key_prev in basins_by_key:
                    bid, pos = basins_by_key[key_prev]
                    y_cont = 1 if (t_next in basin_presence.get(bid, set())) else 0
                else:
                    y_cont = 0

                # predicted probability of continuation: 1 - p_null
                p_cont = float(1.0 - p_null[i]) if len(p_null) > i else 0.0
                cont_y.append(y_cont)
                cont_p.append(p_cont)

                # actual size at t+1 (0 if none)
                actual_size_next = 0
                if y_cont == 1:
                    # find the cluster in t+1 belonging to same basin
                    # find which cluster in next matches the basin (naively search)
                    size_found = 0
                    for j, c_next in enumerate(next_cl):
                        key_next = (t_next, c_next.cluster_local_id)
                        if key_next in basins_by_key and basins_by_key[key_next][0] == bid:
                            size_found = c_next.size
                            break
                    actual_size_next = size_found

                # predicted expected size
                if p_mat.shape[0] > i:
                    exp_size = float(np.dot(p_mat[i, :], np.array([c.size for c in next_cl], dtype=float)))
                else:
                    exp_size = 0.0
                size_y.append(actual_size_next)
                size_pred.append(exp_size)

                # Direction labels (grow vs shrink): treat no-continuation as shrink
                y_dir = 1 if actual_size_next > c_prev.size else 0
                yhat_dir = 1 if exp_size > c_prev.size else 0
                dir_y.append(y_dir)
                dir_pred.append(yhat_dir)

        # Metrics
        cont_auroc = safe_roc_auc(cont_y, cont_p)
        cont_brier = float(np.mean((np.array(cont_p) - np.array(cont_y)) ** 2)) if cont_y else np.nan
        cont_logloss = safe_log_loss(cont_y, cont_p)
        size_mae = mean_absolute_error(size_y, size_pred) if size_y else np.nan
        size_rmse = rmse(size_y, size_pred) if size_y else np.nan
        direction_acc = accuracy_score(dir_y, dir_pred) if dir_y else np.nan
        direction_f1 = f1_score(dir_y, dir_pred) if dir_y and len(set(dir_y)) > 1 else np.nan

        # ---------------- Early detection (on by default, tuned over tiny grid) ---------------- #
        best_early = {
            "p_at_k": np.nan,
            "kappa_like": np.nan,
            "N": None,
            "S": None,
            "Tq": None,
            "kfrac": None,
        }

        if self.config.get("early_enabled", True):
            # collect cluster sizes across train bins to compute quantile thresholds
            all_cluster_sizes = [c.size for c in all_clusters]
            for (N, S, Tq, kfrac) in early_grid:
                # compute T (sustained threshold) as quantile over all cluster sizes
                if len(all_cluster_sizes) == 0:
                    continue
                T = float(np.quantile(all_cluster_sizes, Tq))
                # rolling logistic over fledglings
                y_all = []
                p_all = []
                pi_prevalences = []
                for t_idx in range(len(sorted_bins) - 1):  # up to T-1; we need future bins
                    t = sorted_bins[t_idx]
                    t_nexts = [sorted_bins[idx] for idx in range(t_idx + 1, min(len(sorted_bins), t_idx + 1 + N))]
                    curr_bin_df = bin_dfs[t]
                    clusters_t = bins_clusters.get(t, [])
                    if len(clusters_t) == 0:
                        continue
                    # adaptive min cluster size used earlier
                    bin_sizes = [c.size for c in clusters_t]
                    q25 = float(np.quantile(bin_sizes, 0.25)) if len(bin_sizes) >= 4 else np.median(bin_sizes) if bin_sizes else 0
                    # Define fledgling set
                    # adaptive_min here approximated as max(global_min, min_cluster_frac * len(bin_df))
                    global_min = HDBSCAN_min_cluster_size_global
                    min_frac = float(self.config.get("min_cluster_frac", 0.01))
                    adaptive_min = max(global_min, int(max(2, min_frac * max(1, len(curr_bin_df)))))
                    fledgling_threshold = max(adaptive_min, int(math.ceil(q25)))
                    fledgling_idxs = [i for i, c in enumerate(clusters_t) if c.size <= fledgling_threshold]
                    if len(fledgling_idxs) == 0:
                        continue

                    # labels: "becomes sustained within next N bins for >= S consecutive bins"
                    # for each fledgling, walk its basin future
                    labels = []
                    feats = []
                    for i in fledgling_idxs:
                        c_prev = clusters_t[i]
                        key_prev = (t, c_prev.cluster_local_id)
                        is_pos = 0
                        if key_prev in basins_by_key:
                            bid, pos = basins_by_key[key_prev]
                            # gather sizes in the next N bins for same basin
                            future_sizes = []
                            for fut_bin in t_nexts:
                                # find cluster in basin at fut_bin
                                found = 0
                                for c in basins:
                                    if c.basin_id == bid:
                                        for cc in c.timeline:
                                            if cc.bin_id == fut_bin:
                                                future_sizes.append(cc.size)
                                                found = 1
                                                break
                                        break
                                if not found:
                                    future_sizes.append(0)
                            # check S consecutive >= T
                            consec = 0
                            for s in future_sizes:
                                if s >= T:
                                    consec += 1
                                    if consec >= S:
                                        is_pos = 1
                                        break
                                else:
                                    consec = 0
                        labels.append(is_pos)

                        # simple fledgling features (reuse existing map)
                        f = features_map.get(key_prev, {})
                        feats.append([
                            f.get("size_t", 0.0),
                            f.get("prob_t", 0.0),
                            f.get("avg_member_pop_t", 0.0),
                            f.get("age_in_basin_t", 1.0),
                            f.get("center_shift_t", 0.0),
                            f.get("bin_density_t", 0.0),
                            f.get("neigh_density_t", 0.0),
                        ])

                    labels = np.asarray(labels, dtype=int)
                    feats = np.asarray(feats, dtype=float)

                    # rolling fit: train on all prior fledglings (bins < t), predict on t
                    # accumulate across t_idx
                    if t_idx == 0 or feats.size == 0:
                        continue

                    # build train pool from earlier steps
                    train_labels = []
                    train_feats = []
                    for t_idx2 in range(0, t_idx):
                        t2 = sorted_bins[t_idx2]
                        clusters_t2 = bins_clusters.get(t2, [])
                        if len(clusters_t2) == 0:
                            continue
                        bin_sizes2 = [c.size for c in clusters_t2]
                        q25_2 = float(np.quantile(bin_sizes2, 0.25)) if len(bin_sizes2) >= 4 else np.median(bin_sizes2) if bin_sizes2 else 0
                        bin_df2 = bin_dfs[t2]
                        global_min2 = HDBSCAN_min_cluster_size_global
                        min_frac2 = float(self.config.get("min_cluster_frac", 0.01))
                        adaptive_min2 = max(global_min2, int(max(2, min_frac2 * max(1, len(bin_df2)))))
                        fledgling_threshold2 = max(adaptive_min2, int(math.ceil(q25_2)))
                        fidx2 = [i2 for i2, c2 in enumerate(clusters_t2) if c2.size <= fledgling_threshold2]
                        if not fidx2:
                            continue
                        ys2 = []
                        xs2 = []
                        for i2 in fidx2:
                            cp = clusters_t2[i2]
                            keyp = (t2, cp.cluster_local_id)
                            is_pos2 = 0
                            if keyp in basins_by_key:
                                bid2, pos2 = basins_by_key[keyp]
                                future_bins2 = [sorted_bins[idx] for idx in range(t_idx2 + 1, min(len(sorted_bins), t_idx2 + 1 + N))]
                                future_sizes2 = []
                                for fb in future_bins2:
                                    found2 = 0
                                    for c in basins:
                                        if c.basin_id == bid2:
                                            for cc in c.timeline:
                                                if cc.bin_id == fb:
                                                    future_sizes2.append(cc.size)
                                                    found2 = 1
                                                    break
                                            break
                                    if not found2:
                                        future_sizes2.append(0)
                                consec2 = 0
                                for s2 in future_sizes2:
                                    if s2 >= T:
                                        consec2 += 1
                                        if consec2 >= S:
                                            is_pos2 = 1
                                            break
                                    else:
                                        consec2 = 0
                            ys2.append(is_pos2)
                            f2 = features_map.get(keyp, {})
                            xs2.append([
                                f2.get("size_t", 0.0),
                                f2.get("prob_t", 0.0),
                                f2.get("avg_member_pop_t", 0.0),
                                f2.get("age_in_basin_t", 1.0),
                                f2.get("center_shift_t", 0.0),
                                f2.get("bin_density_t", 0.0),
                                f2.get("neigh_density_t", 0.0),
                            ])
                        if xs2:
                            train_labels.extend(ys2)
                            train_feats.extend(xs2)

                    if not train_feats:
                        continue

                    lr = LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear")
                    try:
                        lr.fit(np.asarray(train_feats, dtype=float), np.asarray(train_labels, dtype=int))
                        probs = lr.predict_proba(feats)[:, 1]
                    except Exception:
                        probs = np.full(len(labels), float(np.mean(train_labels)))

                    y_all.extend(labels.tolist())
                    p_all.extend(probs.tolist())
                    # prevalence π among candidates in this bin:
                    pi = float(np.mean(labels)) if len(labels) > 0 else 0.0
                    if not np.isnan(pi):
                        pi_prevalences.append(pi)

                # Aggregate precision@k and kappa-like lift
                if len(y_all) > 0:
                    y_all_np = np.asarray(y_all, dtype=int)
                    p_all_np = np.asarray(p_all, dtype=float)
                    # k from fraction over total candidates
                    K = max(1, int(round(kfrac * len(y_all_np))))
                    p_at_k = precision_at_k(y_all_np, p_all_np, K)
                    pi = float(np.mean(pi_prevalences)) if pi_prevalences else (float(np.mean(y_all_np)) if len(y_all_np) > 0 else 0.0)
                    if p_at_k is np.nan or not np.isfinite(p_at_k):
                        kappa_like = np.nan
                    else:
                        denom = max(1e-8, 1.0 - pi)
                        kappa_like = float((p_at_k - pi) / denom)
                    # keep best
                    if (not np.isfinite(best_early["kappa_like"])) or (np.isfinite(kappa_like) and kappa_like > best_early["kappa_like"]):
                        best_early = {
                            "p_at_k": float(p_at_k) if p_at_k is not np.nan else np.nan,
                            "kappa_like": float(kappa_like) if np.isfinite(kappa_like) else np.nan,
                            "N": int(N),
                            "S": int(S),
                            "Tq": float(Tq),
                            "kfrac": float(kfrac),
                        }

        return {
            "continuation_auroc": cont_auroc,
            "continuation_brier": cont_brier,
            "continuation_logloss": cont_logloss,
            "size_mae": size_mae,
            "size_rmse": size_rmse,
            "direction_acc": direction_acc,
            "direction_f1": direction_f1,
            "early_p_at_k": best_early["p_at_k"],
            "early_kappa_like": best_early["kappa_like"],
            "best_early_N": best_early["N"],
            "best_early_S": best_early["S"],
            "best_early_Tq": best_early["Tq"],
            "best_early_kfrac": best_early["kfrac"],
        }

    # ---------------------- High-level basin discovery ---------------------- #

    def find_basins_in_dataframe(self, data: pd.DataFrame, label: str, bin_size_days: int,
                                 alpha: float, recency_lambda: float) -> Tuple[List[Basin], List[pd.DataFrame], List[ClusterSummary]]:
        """
        Create bins, per-bin consensus clusters with current (alpha, lambda), and track.
        Returns (basins, bin_dfs, all_clusters)
        """
        t0 = time.time()
        bin_dfs = self._create_bins(data, bin_size_days)
        logger.info("Finding basins in %s: %d bins (size=%d days)", label, len(bin_dfs), bin_size_days)

        # global fallback min cluster size
        n_train = len(data)
        global_min = HDBSCAN_min_cluster_size_global
        if global_min is None:
            global_min = max(5, int(np.sqrt(n_train) / 10))
        global_min = int(global_min)

        all_clusters: List[ClusterSummary] = []
        for bin_df in bin_dfs:
            clusters = self._consensus_clusters_for_bin(
                bin_df=bin_df,
                alpha=alpha,
                recency_lambda=recency_lambda,
                min_cluster_size_global=global_min,
            )
            # adaptive per-bin min threshold
            min_frac = float(self.config.get("min_cluster_frac", 0.01))
            adaptive_min = max(global_min, int(max(2, min_frac * max(1, len(bin_df)))))
            kept = [c for c in clusters if c.size >= adaptive_min]
            if clusters and not kept:
                kept = [max(clusters, key=lambda x: x.size)]
            all_clusters.extend(kept)

        basins = self._track_clusters_probabilistic(all_clusters)
        logger.info("Found %d basins in %s", len(basins), label)
        self.timings[f"find_basins_{label}"] = time.time() - t0
        return basins, bin_dfs, all_clusters

    # ---------------------- Tuning (forward-chaining) ----------------------- #

    def _lambda_grid_for_bin(self, bin_size_days: int) -> List[float]:
        B = float(bin_size_days)
        return [0.0, math.log(2) / B, math.log(2) / (B / 2.0)]

    def forward_chain_tune(self, train_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Grid over:
          - bin_size_days ∈ {5,7,14}
          - min_cluster_frac ∈ {0.002, 0.005, 0.01, 0.02}
          - recency_decay_lambda ∈ {0, ln2/bin, ln2/(bin/2)}
          - engagement alpha ∈ {0.5, 1.0, 2.0}
        Score by forward-chained predictive metrics (default objective: continuation AUROC).
        Also tune early detection tiny grid (N,S,Tq,kfrac) per combo and report best early metrics.
        """
        t0 = time.time()

        bin_candidates = list(self.config.get("bin_size_candidates", [self.config.get("bin_size_days", 7)]))
        min_frac_grid = [0.002, 0.005, 0.01, 0.02]
        alpha_grid = list(self.config.get("engagement_alphas", [0.5, 1.0, 2.0]))

        early_grid = []
        for N in self.config.get("early_N_candidates", [2, 3]):
            for S in self.config.get("early_S_candidates", [1, 2]):
                for Tq in self.config.get("early_Tq_candidates", [0.5, 0.6]):
                    for kfrac in self.config.get("early_k_frac_candidates", [0.05, 0.1]):
                        early_grid.append((int(N), int(S), float(Tq), float(kfrac)))

        results = []
        combo_count = 0
        max_combos = self.config.get("max_combos", None)

        for B in bin_candidates:
            lambda_grid = self._lambda_grid_for_bin(B)
            for min_frac in min_frac_grid:
                for alpha in alpha_grid:
                    for lam in lambda_grid:
                        if (max_combos is not None) and (combo_count >= max_combos):
                            break
                        combo_count += 1
                        logger.info("Tuning combo #%d: bin=%d, min_frac=%.4f, alpha=%.2f, lambda=%.5f",
                                    combo_count, B, min_frac, alpha, lam)

                        # Set current params
                        self.config["bin_size_days"] = int(B)
                        self.config["min_cluster_frac"] = float(min_frac)

                        # Build basins on TRAIN with current (B, alpha, lam)
                        basins, bin_dfs, all_clusters = self.find_basins_in_dataframe(
                            train_df, label="train_tune", bin_size_days=int(B), alpha=float(alpha), recency_lambda=float(lam)
                        )

                        # Evaluate forward with softmatch
                        metrics = self._evaluate_forward(
                            bin_dfs=bin_dfs, all_clusters=all_clusters, basins=basins,
                            bin_size_days=int(B), alpha=float(alpha), recency_lambda=float(lam),
                            early_grid=early_grid
                        )

                        row = {
                            "bin_size": int(B),
                            "min_cluster_frac": float(min_frac),
                            "alpha": float(alpha),
                            "recency_lambda": float(lam),
                            "continuation_auroc": metrics["continuation_auroc"],
                            "continuation_brier": metrics["continuation_brier"],
                            "continuation_logloss": metrics["continuation_logloss"],
                            "size_mae": metrics["size_mae"],
                            "size_rmse": metrics["size_rmse"],
                            "direction_acc": metrics["direction_acc"],
                            "direction_f1": metrics["direction_f1"],
                            "early_p_at_k": metrics["early_p_at_k"],
                            "early_kappa_like": metrics["early_kappa_like"],
                            "best_early_N": metrics["best_early_N"],
                            "best_early_S": metrics["best_early_S"],
                            "best_early_Tq": metrics["best_early_Tq"],
                            "best_early_kfrac": metrics["best_early_kfrac"],
                        }
                        results.append(row)

        # Save tuning results
        df_res = pd.DataFrame(results)
        out_tune = self.output_dir / "tuning_results.csv"
        df_res.to_csv(out_tune, index=False)
        logger.info("Saved tuning results to %s", out_tune)

        # Select best by objective + tie-breakers
        if df_res.empty:
            raise RuntimeError("Tuning produced no results.")

        def sort_key(row):
            # objective: larger better if AUROC; smaller better for brier, logloss, size_mae
            # We’ll sort by:
            # 1) continuation_auroc DESC
            # 2) continuation_brier ASC
            # 3) continuation_logloss ASC
            # 4) size_mae ASC
            # 5) early_kappa_like DESC
            return (
                - (row["continuation_auroc"] if np.isfinite(row["continuation_auroc"]) else -1e9),
                (row["continuation_brier"] if np.isfinite(row["continuation_brier"]) else 1e9),
                (row["continuation_logloss"] if np.isfinite(row["continuation_logloss"]) else 1e9),
                (row["size_mae"] if np.isfinite(row["size_mae"]) else 1e9),
                - (row["early_kappa_like"] if np.isfinite(row["early_kappa_like"]) else -1e9),
            )

        best_idx = min(range(len(df_res)), key=lambda i: sort_key(df_res.iloc[i].to_dict()))
        best = df_res.iloc[best_idx].to_dict()
        logger.info("Best combo: %s", best)

        self.timings["forward_chain_tune"] = time.time() - t0
        return best

    # ---------------------- Validation & saving ---------------------------- #

    def validate_results(self, train_basins: List[Basin], test_basins: List[Basin]) -> Dict[str, Any]:
        def safe_mean(xs):
            return float(np.mean(xs)) if xs else 0.0

        if not train_basins or not test_basins:
            return {"validation": "failed", "reason": "no_basins"}

        train_avg_duration = safe_mean([b.duration for b in train_basins])
        test_avg_duration = safe_mean([b.duration for b in test_basins])
        train_avg_size = safe_mean([b.avg_size for b in train_basins])
        test_avg_size = safe_mean([b.avg_size for b in test_basins])

        duration_sim = min(train_avg_duration, test_avg_duration) / max(1e-8, max(train_avg_duration, test_avg_duration))
        size_sim = min(train_avg_size, test_avg_size) / max(1e-8, max(train_avg_size, test_avg_size))
        similarity = float((duration_sim + size_sim) / 2.0)
        risk = "Low" if similarity > 0.7 else "High" if similarity < 0.4 else "Medium"
        return {
            "similarity_score": similarity,
            "train_basins": len(train_basins),
            "test_basins": len(test_basins),
            "overfitting_risk": risk,
        }

    def save_basins(self, basins: List[Basin], filename: str):
        rows = []
        for b in basins:
            rows.append({
                "basin_id": b.basin_id,
                "duration": b.duration,
                "avg_size": b.avg_size,
                "total_size": b.total_size,
                "size_stability": b.size_stability,
                "trend_slope": b.trend_slope,
                "trend_r2": b.trend_r2,
                "predicted_direction": b.predicted_direction,
                "confidence": b.confidence,
            })
        df_out = pd.DataFrame(rows)
        out_path = self.output_dir / filename
        df_out.to_csv(out_path, index=False)
        logger.info("Saved basin summary to %s", out_path)

    # ---------------------- Runner ---------------------------------------- #

    def run_analysis(self) -> Dict[str, Any]:
        t0 = time.time()
        self.load_data()
        train_df, test_df = self.temporal_split()

        # Tuning on TRAIN
        best = self.forward_chain_tune(train_df)
        B = int(best["bin_size"])
        min_frac = float(best["min_cluster_frac"])
        alpha = float(best["alpha"])
        lam = float(best["recency_lambda"])

        self.config["bin_size_days"] = B
        self.config["min_cluster_frac"] = min_frac

        # Final basins on TRAIN + TEST using tuned (B, alpha, lam)
        train_basins, train_bins, _ = self.find_basins_in_dataframe(train_df, label="train", bin_size_days=B, alpha=alpha, recency_lambda=lam)
        test_basins, test_bins, _ = self.find_basins_in_dataframe(test_df, label="test", bin_size_days=B, alpha=alpha, recency_lambda=lam)

        validation = self.validate_results(train_basins, test_basins)

        # Save
        self.save_basins(train_basins, "train_basins_summary.csv")
        self.save_basins(test_basins, "test_basins_summary.csv")

        self.timings["total_run"] = time.time() - t0
        logger.info("Total analysis completed in %.2fs", self.timings["total_run"])

        return {
            "best_parameters": best,
            "validation": validation,
            "train_basins": len(train_basins),
            "test_basins": len(test_basins),
            "timings": self.timings,
        }

# ---------------------- CLI ----------------------------------------------- #

def parse_args():
    parser = argparse.ArgumentParser(description="Ultra Basin Analyzer - Predictive Tuning + Probabilistic Continuation + Early Detection")
    parser.add_argument("--bins", "-b", type=int, default=None, help="Override bin size (days). If set, tuner will still evaluate other grid values unless --no-tune.")
    parser.add_argument("--n_boot", type=int, default=None, help="Override number of bootstrap runs per bin.")
    parser.add_argument("--min_frac", type=float, default=None, help="Override min_cluster_frac (fraction of bin size).")
    parser.add_argument("--no_tune", action="store_true", help="Skip tuning and run directly with current config (not recommended).")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    cfg: Dict[str, Any] = {}
    if args.bins is not None:
        cfg["bin_size_days"] = int(args.bins)
        cfg["bin_size_candidates"] = [int(args.bins)]
    if args.n_boot is not None:
        cfg["n_bootstrap"] = int(args.n_boot)
    if args.min_frac is not None:
        cfg["min_cluster_frac"] = float(args.min_frac)

    analyzer = UltraBasinAnalyzer(input_file=INPUT_FILE, output_dir=OUTPUT_DIR, config=cfg)

    if args.no_tune:
        analyzer.load_data()
        train_df, test_df = analyzer.temporal_split()
        B = int(analyzer.config.get("bin_size_days", 7))
        # pick alpha, lambda defaults for direct run
        alpha_default = float(analyzer.config.get("engagement_alphas", [1.0])[0])
        lam_default = float(analyzer._lambda_grid_for_bin(B)[0])
        train_basins, _, _ = analyzer.find_basins_in_dataframe(train_df, label="train", bin_size_days=B, alpha=alpha_default, recency_lambda=lam_default)
        test_basins, _, _ = analyzer.find_basins_in_dataframe(test_df, label="test", bin_size_days=B, alpha=alpha_default, recency_lambda=lam_default)
        analyzer.save_basins(train_basins, "train_basins_summary.csv")
        analyzer.save_basins(test_basins, "test_basins_summary.csv")
        logger.info("FINAL RESULTS: %s", {
            "best_parameters": {"bin_size": B, "alpha": alpha_default, "recency_lambda": lam_default, "min_cluster_frac": analyzer.config.get("min_cluster_frac", 0.01)},
            "validation": analyzer.validate_results(train_basins, test_basins),
            "train_basins": len(train_basins),
            "test_basins": len(test_basins),
            "timings": analyzer.timings,
        })
    else:
        results = analyzer.run_analysis()
        logger.info("FINAL RESULTS: %s", results)
