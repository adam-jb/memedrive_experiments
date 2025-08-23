#!/usr/bin/env python3
"""
Ultra Basin Analyzer - Full Script (HDBSCAN-only, probabilistic consensus, temporal tracking)

Features:
 - HDBSCAN-only clustering (no KMeans).
 - Bootstrap consensus clustering per temporal bin (produces cluster probability/confidence).
 - Adaptive min_cluster_size per bin (fraction of bin size) + global fallback/defaults.
 - Temporal tracking using Hungarian matching with probability-aware costs.
 - Basin trajectory modelling (linear trend), stability metrics, and predicted direction.
 - Detailed timing/logging for optimization, per-bin processing, consensus runs, tracking, and full run.
 - CSV export of basin summaries for train and test.
 - CLI usage.

Requirements:
 - Python 3.8+
 - numpy, pandas, scikit-learn, scipy, hdbscan (install hdbscan separately)
"""

from __future__ import annotations

import argparse
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.cluster import AgglomerativeClustering
from sklearn.linear_model import LinearRegression
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Optional: silence specific sklearn FutureWarnings (uncomment if desired)
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

try:
    import hdbscan
except Exception:
    hdbscan = None

# Logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("UltraBasin")

# ----------------------------- Config / Defaults ----------------------------- #

DEFAULT_CONFIG: Dict[str, Any] = {
    # Temporal & binning
    "bin_size_days": 3,
    "bin_size_candidates": [1, 2, 3, 5, 7],

    # Consensus / bootstrapping
    "n_bootstrap": 30,  # number of bootstrap clustering runs per bin
    "consensus_max_clusters": None,  # heuristic fallback for consensus clustering

    # HDBSCAN defaults
    "hdbscan_min_cluster_size_global": None,  # if None auto-calc from train size
    "min_cluster_frac": 0.05,  # adaptive min cluster size per bin = max(min_cluster_min, frac * bin_size)

    # Tracking
    "max_distance_threshold": 1.5,
    "min_track_length": 2,

    # Misc
    "engagement_weight": True,
    "train_split": 0.6,
    "random_state": 42,

    # Performance
    "verbose_timings": True,
}

# ----------------------------- Data classes --------------------------------- #

INPUT_FILE = "~/Desktop/memedrive_experiments/output_data/basin_finder/dummy_tweet_embeddings.csv"
OUTPUT_DIR = "~/Desktop/memedrive_experiments/output_data/basin_finder/"

@dataclass
class ClusterSummary:
    bin_id: int
    cluster_local_id: int
    center: np.ndarray
    size: int
    members_idx: np.ndarray  # indices in the bin (0..n_bin-1)
    probability: float = 1.0  # consensus confidence 0..1


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
    predicted_direction: str = "unknown"  # strengthening / weakening / stable / insufficient_data
    confidence: float = 0.0


# ----------------------------- Analyzer ------------------------------------ #


class UltraBasinAnalyzer:
    def __init__(self, input_file: str, output_dir: str = "./", config: Optional[Dict[str, Any]] = None):
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

        if hdbscan is None:
            raise ImportError("hdbscan is required but not installed. Install via `pip install hdbscan`.")

        # runtime info
        self.timings: Dict[str, float] = {}

    # --------------------------- I/O & Preparation ------------------------- #

    def load_data(self):
        t0 = time.time()
        logger.info("Loading data from %s", self.input_file)
        self.df = pd.read_csv(self.input_file)
        if "datetime" not in self.df.columns:
            raise ValueError("Input CSV must contain a 'datetime' column.")
        self.df["datetime"] = pd.to_datetime(self.df["datetime"])
        self.df = self.df.sort_values("datetime").reset_index(drop=True)
        # embedding columns are any columns that start with 'e'
        self.embedding_cols = [c for c in self.df.columns if c.startswith("e")]
        if not self.embedding_cols:
            raise ValueError("No embedding columns found (expected columns starting with 'e').")
        logger.info("Loaded %d rows and %d embedding dims", len(self.df), len(self.embedding_cols))
        self.timings["load_data"] = time.time() - t0

    def temporal_split(self):
        # split train/test by chronological order
        t0 = time.time()
        split_idx = int(len(self.df) * float(self.config.get("train_split", 0.6)))
        train_df = self.df.iloc[:split_idx].reset_index(drop=True)
        test_df = self.df.iloc[split_idx:].reset_index(drop=True)
        logger.info("Temporal split -> Train: %d rows, Test: %d rows", len(train_df), len(test_df))
        self.timings["temporal_split"] = time.time() - t0
        return train_df, test_df

    # ---------------------- Heuristics & diagnostics ------------------------ #

    def assess_data_quality(self, sample_fraction: float = 0.1) -> Dict[str, Any]:
        """
        Quick heuristics: effective rows, silhouette estimate on subsample, and recommended param budget.
        """
        t0 = time.time()
        df = self.df
        n = len(df)
        # effective rows are those where embedding columns are not NaN
        valid_mask = ~df[self.embedding_cols].isnull().any(axis=1)
        effective_n = int(valid_mask.sum())

        # subsample for silhouette estimate
        subsample_n = min(max(100, int(n * sample_fraction)), 2000)
        subst = df.loc[valid_mask].sample(n=subsample_n, random_state=self.random_state)
        X = subst[self.embedding_cols].values
        scaler = StandardScaler().fit(X)
        Xs = scaler.transform(X)

        sil = 0.0
        try:
            # quick agglomerative with small k to estimate separation
            k = min(8, max(2, subsample_n // 200))
            agg = AgglomerativeClustering(n_clusters=k)
            labels = agg.fit_predict(Xs)
            sil = float(silhouette_score(Xs, labels))
        except Exception:
            sil = 0.0

        param_budget = int(min(self.config.get("max_params_to_optimize", 3),
                               max(1, math.log2(max(2, effective_n)) * (0.5 + sil))))
        diag = {
            "n": n,
            "effective_n": effective_n,
            "silhouette_estimate": sil,
            "recommended_param_budget": param_budget,
        }
        self.timings["assess_data_quality"] = time.time() - t0
        logger.info("Data diagnostics: %s", diag)
        return diag

    # ---------------------- Parameter optimization ------------------------- #

    def optimize_parameters(self, train_df: pd.DataFrame):
        """
        Optimize bin_size among candidates using mean silhouette across bins.
        Also sets a global fallback hdbscan_min_cluster_size and kmeans_k heuristic
        (we don't use kmeans but keep a heuristic for fallback decisions).
        """
        t0 = time.time()
        diag = self.assess_data_quality()
        bin_candidates = self.config.get("bin_size_candidates", [self.config.get("bin_size_days")])
        best_bin = int(self.config.get("bin_size_days", 3))
        best_score = -np.inf

        logger.info("Optimizing bin_size among: %s", bin_candidates)
        for b in bin_candidates:
            score = self._evaluate_bin_size_on_sample(train_df, b)
            logger.info("Bin %s -> score %.4f", b, score)
            if score > best_score:
                best_score = score
                best_bin = int(b)

        # global fallback hdbscan min cluster size (for large-scale defaults)
        n_train = len(train_df)
        global_min = self.config.get("hdbscan_min_cluster_size_global")
        if global_min is None:
            global_min = max(5, int(np.sqrt(n_train) / 10))

        self.config["chosen_bin_size"] = best_bin
        self.config["hdbscan_min_cluster_size_global"] = int(global_min)
        logger.info("Optimized params: bin_size=%d, hdbscan_min_cluster_size_global=%d", best_bin, global_min)
        self.timings["optimize_parameters"] = time.time() - t0
        return {"bin_size": best_bin, "hdbscan_min_cluster_size_global": int(global_min)}

    def _evaluate_bin_size_on_sample(self, train_df: pd.DataFrame, bin_size: int) -> float:
        # create bins and compute mean silhouette (quick) across sufficiently large bins
        t0 = time.time()
        bins = self._create_bins(train_df, bin_size)
        scores = []
        for bin_df in bins:
            if len(bin_df) < 20:
                continue
            X = bin_df[self.embedding_cols].values
            scaler = StandardScaler().fit(X)
            Xs = scaler.transform(X)
            try:
                k = min(8, max(2, len(bin_df) // 50))
                # Use Agglomerative quick clustering to estimate silhouette
                agg = AgglomerativeClustering(n_clusters=k)
                labels = agg.fit_predict(Xs)
                s = silhouette_score(Xs, labels)
                scores.append(s)
            except Exception:
                continue
        elapsed = time.time() - t0
        # logger.debug("Bin size eval %d took %.2fs", bin_size, elapsed)
        return float(np.mean(scores)) if scores else -1.0

    # ---------------------- Binning ---------------------------------------- #

    def _create_bins(self, data: pd.DataFrame, bin_size_days: int) -> List[pd.DataFrame]:
        """
        Create temporal bins of size `bin_size_days`. Returns list of dataframes (each bin).
        """
        t0 = time.time()
        start_date = data["datetime"].min()
        end_date = data["datetime"].max()
        bins = []
        current = start_date
        while current < end_date:
            nxt = current + pd.Timedelta(days=bin_size_days)
            mask = (data["datetime"] >= current) & (data["datetime"] < nxt)
            bin_df = data.loc[mask].reset_index(drop=True)
            if len(bin_df) > 0:
                bins.append(bin_df)
            current = nxt
        self.timings["_create_bins"] = time.time() - t0
        return bins

    # ---------------------- Consensus clustering per bin ------------------- #

    def _consensus_clusters_for_bin(self, bin_df: pd.DataFrame, bin_id: int) -> List[ClusterSummary]:
        """
        Bootstrap clustering to create consensus clusters with probabilities.
        Returns list of ClusterSummary (members indices are 0..n_bin-1).
        """
        t0 = time.time()
        n_boot = int(self.config.get("n_bootstrap", 30))
        n = len(bin_df)
        if n == 0:
            return []

        X = bin_df[self.embedding_cols].values
        scaler = StandardScaler().fit(X)
        Xs = scaler.transform(X)

        # engagement weighting: sampling probabilities
        if self.config.get("engagement_weight") and ("retweet_count" in bin_df.columns or "favourite_count" in bin_df.columns):
            rt = bin_df.get("retweet_count", pd.Series(0, index=bin_df.index)).fillna(0).astype(float).values
            fav = bin_df.get("favourite_count", pd.Series(0, index=bin_df.index)).fillna(0).astype(float).values
            weights = rt + fav + 1.0
            probs = weights / np.sum(weights)
        else:
            probs = np.ones(n) / n

        rng = np.random.RandomState(self.random_state)

        # co-assignment counts matrix
        coassign = np.zeros((n, n), dtype=float)

        # run bootstrap clustering runs
        total_runs = 0
        for run in range(n_boot):
            # sample indices with replacement using probs
            sample_idx = rng.choice(np.arange(n), size=n, replace=True, p=probs)
            X_sample = Xs[sample_idx]

            try:
                # HDBSCAN with adaptive min_cluster_size for the sampled set:
                # we use the global fallback for HDBSCAN min cluster size here, but could adapt further
                min_cs_global = int(self.config.get("hdbscan_min_cluster_size_global", 10))
                # ensure min_cs not larger than sample size
                min_cs_run = max(2, min(min_cs_global, max(2, int(len(sample_idx) * 0.1))))
                clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cs_run)
                labels_sample = clusterer.fit_predict(X_sample)
            except Exception:
                # if HDBSCAN fails (rare), skip this run
                continue

            total_runs += 1
            # map sample labels back to original indices (last occurrence wins for duplicates)
            labels_full = np.full(n, -1, dtype=int)
            for j, orig_idx in enumerate(sample_idx):
                labels_full[orig_idx] = int(labels_sample[j])

            # increment coassign for members that share non-noise label
            for lbl in np.unique(labels_full):
                if lbl == -1:
                    continue
                members = np.where(labels_full == lbl)[0]
                if members.size > 0:
                    # increment rows for all pairs in members
                    coassign[np.ix_(members, members)] += 1.0

        if total_runs == 0:
            logger.warning("No successful bootstrap clustering runs for bin %d", bin_id)
            self.timings[f"consensus_bin_{bin_id}"] = time.time() - t0
            return []

        # normalize to 0..1 by number of runs
        coassign = coassign / float(max(1, total_runs))

        # create a distance-like representation: dist = 1 - coassign
        dist = 1.0 - coassign

        # low-d embedding of dist (SVD) to feed into AgglomerativeClustering
        try:
            # center the dist matrix for SVD stability
            M = dist - dist.mean(axis=0)[None, :]
            U, S, Vt = np.linalg.svd(M, full_matrices=False)
            repr_dim = min(8, max(1, n - 1))
            X_repr = U[:, :repr_dim] * S[:repr_dim]
        except Exception:
            # fallback: use original embeddings scaled
            X_repr = Xs

        # heuristic number of consensus clusters: sqrt(n)/3 or config override
        k_hint = self.config.get("consensus_max_clusters")
        if k_hint is None:
            k = max(1, int(np.sqrt(n) / 3))
            k = min(k, n)
        else:
            k = max(1, min(int(k_hint), n))

        # Agglomerative clustering on representation to get consensus clusters
        try:
            agg = AgglomerativeClustering(n_clusters=k)
            cons_labels = agg.fit_predict(X_repr)
        except Exception:
            # fallback: all-in-one cluster
            cons_labels = np.zeros(n, dtype=int)

        clusters: List[ClusterSummary] = []
        for lbl in np.unique(cons_labels):
            members = np.where(cons_labels == lbl)[0]
            if members.size == 0:
                continue
            center = X[members].mean(axis=0)
            if members.size == 1:
                prob = 1.0
            else:
                sub = coassign[np.ix_(members, members)]
                prob = float(np.mean(sub))
            clusters.append(ClusterSummary(bin_id=bin_id,
                                           cluster_local_id=len(clusters),
                                           center=center,
                                           size=int(members.size),
                                           members_idx=members,
                                           probability=prob))
        self.timings[f"consensus_bin_{bin_id}"] = time.time() - t0
        return clusters

    # ---------------------- Tracking across bins ---------------------------- #

    def _track_clusters_probabilistic(self, all_clusters: List[ClusterSummary]) -> List[Basin]:
        """
        Track consensus clusters across time bins with probability-aware Hungarian matching.
        """
        t0 = time.time()
        # group clusters by bin_id
        bins = {}
        for c in all_clusters:
            bins.setdefault(c.bin_id, []).append(c)
        sorted_bins = sorted(bins.keys())

        basins: List[Basin] = []
        next_basin_id = 0

        for i, bin_id in enumerate(sorted_bins):
            curr = bins[bin_id]
            if i == 0:
                # start a basin for each cluster in first bin
                for c in curr:
                    basins.append(Basin(basin_id=next_basin_id, timeline=[c], probabilities=[c.probability]))
                    next_basin_id += 1
                continue

            prev_active = [b for b in basins if len(b.timeline) == i]  # basins that had last event in previous bin
            # if no active prev basins, start new ones for all curr clusters
            if not prev_active:
                for c in curr:
                    basins.append(Basin(basin_id=next_basin_id, timeline=[c], probabilities=[c.probability]))
                    next_basin_id += 1
                continue

            prev_centers = np.vstack([b.timeline[-1].center for b in prev_active])
            prev_probs = np.array([b.timeline[-1].probability for b in prev_active])
            curr_centers = np.vstack([c.center for c in curr])
            curr_probs = np.array([c.probability for c in curr])

            distances = cdist(prev_centers, curr_centers)

            # cost: distance scaled by inverse probabilities (prefer matching high-prob clusters)
            prob_factor = np.outer(1.0 / (prev_probs + 1e-8), 1.0 / (curr_probs + 1e-8))
            cost = distances * prob_factor

            # Hungarian assignment
            row_ind, col_ind = linear_sum_assignment(cost)
            matched_prev = set()
            matched_curr = set()

            for r, c_idx in zip(row_ind, col_ind):
                # acceptance threshold: scaled distance threshold considering cluster confidences
                threshold = self.config.get("max_distance_threshold", 1.5)
                # scale threshold smaller for more confident clusters and larger for low-confidence
                scale = 1.0 + (1.0 - prev_probs[r]) + (1.0 - curr_probs[c_idx])
                if distances[r, c_idx] <= threshold * scale:
                    # attach current cluster to prev_active[r] basin
                    prev_active[r].timeline.append(curr[c_idx])
                    prev_active[r].probabilities.append(curr[c_idx].probability)
                    matched_prev.add(r)
                    matched_curr.add(c_idx)

            # start new basins for unmatched current clusters
            for idx_c, c in enumerate(curr):
                if idx_c not in matched_curr:
                    basins.append(Basin(basin_id=next_basin_id, timeline=[c], probabilities=[c.probability]))
                    next_basin_id += 1

            # note: unmatched prev basins simply stop receiving new timeline entries (they end)

        # finalize basins: compute stats and filter by min_track_length
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

            # aggregate confidence: mean of probabilities scaled by duration proportion to n_bootstrap
            b.confidence = float(np.mean(b.probabilities)) * min(1.0, b.duration / max(1, self.config.get("n_bootstrap", 30)))

            # trend analysis (linear regression on sizes vs time)
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

        self.timings["tracking"] = time.time() - t0
        return final_basins

    # ---------------------- High-level basin discovery ---------------------- #

    def find_basins_in_dataframe(self, data: pd.DataFrame, label: str = "") -> List[Basin]:
        """
        Top-level: create bins, run consensus clustering per bin, then track clusters across time to form basins.
        """
        t0 = time.time()
        bin_size = int(self.config.get("chosen_bin_size", self.config.get("bin_size_days", 3)))
        bins = self._create_bins(data, bin_size)
        logger.info("Finding basins in %s: created %d bins (bin_size=%d days)", label or "data", len(bins), bin_size)

        all_clusters: List[ClusterSummary] = []
        for i, bin_df in enumerate(bins):
            bin_start = time.time()
            # compute adaptive min_cluster_size for this bin (as fraction of bin size)
            min_frac = float(self.config.get("min_cluster_frac", 0.05))
            min_cluster_min = int(self.config.get("hdbscan_min_cluster_size_global", 5))
            adaptive_min = max(min_cluster_min, int(max(2, min_frac * max(1, len(bin_df)))))
            # store in config for use inside bootstrap if needed
            self.config["hdbscan_min_cluster_size_bin"] = adaptive_min

            # run consensus clustering (bootstrap)
            clusters = self._consensus_clusters_for_bin(bin_df, i)
            # filter tiny clusters based on adaptive_min
            filtered = [c for c in clusters if c.size >= adaptive_min]
            # If consensus produced clusters but all filtered out, consider lowering threshold: keep the largest cluster
            if clusters and not filtered:
                # keep largest consensus cluster even if below adaptive_min
                largest = max(clusters, key=lambda x: x.size)
                filtered = [largest]
            # append with bin id already embedded
            for c in filtered:
                # adjust cluster center back to original embedding scale if necessary (we used raw X mean)
                all_clusters.append(c)
            elapsed_bin = time.time() - bin_start
            logger.info("Processed bin %d/%d (n=%d) -> %d consensus clusters (kept %d) in %.2fs",
                        i + 1, len(bins), len(bin_df), len(clusters), len(filtered), elapsed_bin)

        basins = self._track_clusters_probabilistic(all_clusters)
        logger.info("Found %d basins in %s", len(basins), label or "data")
        self.timings[f"find_basins_{label or 'data'}"] = time.time() - t0
        return basins

    # ---------------------- Validation & saving ---------------------------- #

    def validate_results(self, train_basins: List[Basin], test_basins: List[Basin]) -> Dict[str, Any]:
        if not train_basins or not test_basins:
            return {"validation": "failed", "reason": "no_basins"}

        def safe_mean(xs):
            return float(np.mean(xs)) if xs else 0.0

        train_avg_duration = safe_mean([b.duration for b in train_basins])
        test_avg_duration = safe_mean([b.duration for b in test_basins])
        train_avg_size = safe_mean([b.avg_size for b in train_basins])
        test_avg_size = safe_mean([b.avg_size for b in test_basins])

        duration_sim = min(train_avg_duration, test_avg_duration) / max(1e-8, max(train_avg_duration, test_avg_duration))
        size_sim = min(train_avg_size, test_avg_size) / max(1e-8, max(train_avg_size, test_avg_size))
        similarity = float((duration_sim + size_sim) / 2.0)

        risk = "Low" if similarity > 0.7 else "High" if similarity < 0.4 else "Medium"
        validation = {
            "similarity_score": similarity,
            "train_basins": len(train_basins),
            "test_basins": len(test_basins),
            "overfitting_risk": risk,
        }
        logger.info("Validation: %s", validation)
        return validation

    def save_basins(self, basins: List[Basin], filename: str = "basins_summary.csv"):
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
        params = self.optimize_parameters(train_df)
        # set chosen bin size to config so find_basins picks it up
        self.config["chosen_bin_size"] = int(params["bin_size"])
        # find basins
        train_basins = self.find_basins_in_dataframe(train_df, label="train")
        test_basins = self.find_basins_in_dataframe(test_df, label="test")
        validation = self.validate_results(train_basins, test_basins)
        # save outputs
        self.save_basins(train_basins, "train_basins_summary.csv")
        self.save_basins(test_basins, "test_basins_summary.csv")
        self.timings["total_run"] = time.time() - t0
        logger.info("Total analysis completed in %.2fs", self.timings["total_run"])
        return {
            "parameters": params,
            "validation": validation,
            "train_basins": len(train_basins),
            "test_basins": len(test_basins),
            "timings": self.timings,
        }


# ---------------------- CLI & Execution ---------------------------------- #

def parse_args():
    parser = argparse.ArgumentParser(description="Ultra Basin Analyzer - Full Script (HDBSCAN-only)")
    #parser.add_argument("--input", "-i", required=True, help="Input CSV file path (must contain 'datetime' and embedding columns 'e*').")
    #parser.add_argument("--out", "-o", default="./output", help="Output directory for CSV summaries.")
    parser.add_argument("--bins", "-b", type=int, default=None, help="Override chosen bin size (days).")
    parser.add_argument("--n_boot", type=int, default=None, help="Override number of bootstrap runs per bin.")
    parser.add_argument("--min_frac", type=float, default=None, help="Override min_cluster_frac (fraction of bin size).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = {}
    if args.bins is not None:
        cfg["bin_size_days"] = args.bins
        cfg["bin_size_candidates"] = [args.bins]
    if args.n_boot is not None:
        cfg["n_bootstrap"] = int(args.n_boot)
    if args.min_frac is not None:
        cfg["min_cluster_frac"] = float(args.min_frac)

    analyzer = UltraBasinAnalyzer(input_file=INPUT_FILE, output_dir=OUTPUT_DIR, config=cfg)
    results = analyzer.run_analysis()

    logger.info("FINAL RESULTS: %s", results)
