"""Shared data/split/metric code.

Mirrors the authors' final table generator
(repo/code/scripts/generate_alternative_metrics_legacy.py + generate_final_tables.py)
so reproduced and improved models are scored on byte-identical test sets.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, precision_recall_fscore_support, r2_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "repo"
PROC = REPO / "data" / "processed"
RESULTS = ROOT / "results"
SEED = 42
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.70, 0.15, 0.15

DATASETS = ["rolling_mean", "cleaned_data", "era5_daily", "processed_seq", "hydrographic", "biotoxin", "cast"]
RANDOM_SPLIT = {"cast"}
DATE_COL = {
    "cleaned_data": "date", "rolling_mean": "date", "processed_seq": "Date", "cast": "Date",
    "era5_daily": "time", "biotoxin": "UTC DATE (YYYYMMDD)", "hydrographic": "UTC DATE (YYYYMMDD)",
}
DEEP_DATASETS = ["rolling_mean", "cleaned_data", "processed_seq", "hydrographic", "biotoxin"]

# Copied verbatim from LEGACY_DATASET_CONFIG in the authors' script.
LEGACY = {
    "biotoxin": {"target": "VALUE", "exclude": ["Date", "date", "time", "Time"]},
    "cast": {"target": "Bottom_D", "exclude": [
        "Date", "date", "time", "Time", "Lat_Dec", "Lat_Deg", "Lat_Min", "Lat_Hem",
        "Lon_Dec", "Lon_Deg", "Lon_Min", "Lon_Hem", "Rpt_Line", "St_Line", "Ac_Line",
        "Rpt_Sta", "St_Station", "Ac_Sta", "Sta_ID", "Sta_Code", "Orig_Sta_ID",
        "Cruise_ID", "Cast_ID", "DbSta_ID"]},
    # Paper Table 1 / S11: era5 benchmark file has 8 raw ERA5 predictors (no wind10 rolling columns).
    "era5_daily": {"target": "wind10", "exclude": ["Date", "date", "time", "Time"],
                   "only": ["latitude", "longitude", "number", "u10", "v10", "msl", "tcc", "ssr_flux"]},
    "cleaned_data": {"target": "G2chla", "exclude": ["Date", "date", "time", "Time"]},
    "rolling_mean": {"target": "G2chla", "exclude": ["Date", "date", "time", "Time"]},
    # Authors' published predictions show the tabular target is Target_G2chla (their column fallback).
    "processed_seq": {"target": "Target_G2chla", "exclude": ["Date", "date", "time", "Time"]},
    "hydrographic": {"target": "CHLOROPHYLL-a (µg l-1)", "exclude": [
        "STATION", "LATITUDE (degrees North)", "LONGITUDE (degrees East)",
        "DATUM", "UTC DATE (YYYYMMDD)", "UTC TIME (hhmmss)"]},
}


def load_tabular(ds, df=None):
    """Exact replica of prepare_legacy_tabular(): returns X, y, df, target_col."""
    df = pd.read_csv(PROC / ds / "clean.csv", low_memory=False) if df is None else df
    cfg = LEGACY[ds]
    target = cfg["target"]
    if target not in df.columns:  # authors fallback
        chl = [c for c in df.columns if "chlorophyll-a" in c.lower() or "chla" in c.lower()]
        target = chl[0] if chl else df.select_dtypes(include=[np.number]).columns[-1]
    excl = set(cfg["exclude"]) | {target}
    cols = cfg.get("only", df.columns)
    X = df[[c for c in cols if c not in excl]].select_dtypes(include=[np.number]).copy()
    y = df[target].copy()
    X = X.fillna(X.median(numeric_only=True))
    y = y.fillna(y.median())
    X = X.drop(columns=[c for c in X.columns if X[c].std() < 1e-10])
    if X.shape[1] > 1:
        corr = X.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_remove = set()
        for col in upper.columns:
            for f in upper.index[upper[col] > 0.95].tolist():
                if f not in to_remove:
                    to_remove.add(col)
        X = X.drop(columns=list(to_remove))
    return X, y, df, target


def stratified_random_split(y, test_size, val_size, seed, n_bins=10):
    y = pd.Series(y).reset_index(drop=True)
    idx = np.arange(len(y))
    bins = pd.qcut(y, q=n_bins, duplicates="drop").astype(str)
    tr, tmp = train_test_split(idx, test_size=test_size + val_size, random_state=seed, stratify=bins)
    va, te = train_test_split(tmp, test_size=1 - val_size / (test_size + val_size),
                              random_state=seed, stratify=bins.iloc[tmp])
    return tr, va, te


def time_split(n):
    a, b = int(n * TRAIN_RATIO), int(n * (TRAIN_RATIO + VAL_RATIO))
    idx = np.arange(n)
    return idx[:a], idx[a:b], idx[b:]


def tabular_splits(ds, df=None):
    """Returns (X, y, (tr, va, te), split_rule) in the authors' row order."""
    X, y, df, _ = load_tabular(ds, df)
    if ds not in RANDOM_SPLIT and DATE_COL[ds] in df.columns:
        order = df.sort_values(by=DATE_COL[ds]).index  # authors' default quicksort, unstable ties preserved
        X, y = X.loc[order].reset_index(drop=True), y.loc[order].reset_index(drop=True)
        return X, y, time_split(len(y)), "chronological-70/15/15"
    return X, y, stratified_random_split(y, TEST_RATIO, VAL_RATIO, SEED), "stratified-random-70/15/15"


def sequence_splits(ds):
    d = np.load(PROC / ds / "sequences.npz", allow_pickle=True)
    X, y = d["X"].astype(np.float64), d["y"].astype(np.float64)
    return X, y, time_split(len(y))


def bootstrap_ci_r2(y_true, y_pred, n_bootstrap=300, seed=42, max_samples=5000):
    """Replica of _bootstrap_ci_r2() in generate_final_tables.py (300 resamples, seed 42)."""
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    n = len(y_true)
    if n > max_samples:
        pool = rng.choice(n, size=max_samples, replace=False)
        y_true, y_pred, n = y_true[pool], y_pred[pool], max_samples
    s = [r2_score(y_true[i], y_pred[i]) for i in (rng.integers(0, n, n) for _ in range(n_bootstrap))]
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def metrics(y_true, y_pred, y_train):
    y_true, y_pred, y_train = (np.asarray(a, float) for a in (y_true, y_pred, y_train))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    thr = float(np.percentile(y_train, 90))
    p, r, f1, _ = precision_recall_fscore_support((y_true >= thr).astype(int), (y_pred >= thr).astype(int),
                                                  average="binary", zero_division=0)
    lo, hi = bootstrap_ci_r2(y_true, y_pred)
    return {"R2": float(r2_score(y_true, y_pred)), "CI_low": lo, "CI_high": hi,
            "MAE": float(mean_absolute_error(y_true, y_pred)), "RMSE": rmse,
            "NRMSE": rmse / float(np.ptp(y_train)), "event_P": float(p), "event_R": float(r),
            "event_F1": float(f1), "n_test": len(y_true)}
