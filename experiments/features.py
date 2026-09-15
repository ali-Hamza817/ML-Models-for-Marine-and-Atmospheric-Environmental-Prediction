"""Feature builders for the improved models. Every feature uses only predictors of the current row
and/or rows strictly before it (sequence window) — never the current or future target."""
import numpy as np
import pandas as pd

from common import DATE_COL, LEGACY, PROC, sequence_splits, tabular_splits


def target_channel(ds):
    df = pd.read_csv(PROC / ds / "clean.csv", nrows=5, low_memory=False)
    return list(df.select_dtypes(include=[np.number]).columns).index(LEGACY[ds]["target"])


def window_features(S, tc):
    """Summaries of the 30-step history window (same information the paper's LSTM receives)."""
    n, w, f = S.shape
    t = S[:, :, tc]
    steps = np.arange(w) - (w - 1) / 2
    slope = (S * steps[None, :, None]).sum(1) / (steps ** 2).sum()
    ewm = []
    for a in (0.5, 0.2, 0.05):
        wts = (1 - a) ** np.arange(w)[::-1]
        ewm.append((t * wts).sum(1) / wts.sum())
    return np.hstack([t, S[:, -1, :], S.mean(1), S.std(1), S.min(1), S.max(1), slope, np.stack(ewm, 1)])


def seq_design(ds, hybrid):
    """Rows aligned to sequence targets: window summaries (+ current-row tabular predictors if hybrid)."""
    S, y, split = sequence_splits(ds)
    F = window_features(S, target_channel(ds))
    if hybrid:
        X_tab, _, _, _ = tabular_splits(ds)
        F = np.hstack([F, X_tab.to_numpy(float)[30:30 + len(y)]])  # sequence i  <->  sorted row i+30
    return F, y, split


def era5_physics(X):
    """Derived from the same-row ERA5 predictors only."""
    X = X.copy()
    X["uv_speed"] = np.hypot(X["u10"], X["v10"])
    X["uv_dir_sin"] = X["v10"] / X["uv_speed"].clip(lower=1e-3)
    X["uv_dir_cos"] = X["u10"] / X["uv_speed"].clip(lower=1e-3)
    return X


def era5_context(ds="era5_daily"):
    """Previous-day ERA5 predictors (u10, v10, msl) at the same grid cell: a daily-mean wind vector
    hides intra-day direction changes, which the day-to-day vector change reveals. No target values are used."""
    X, y, split, rule = tabular_splits(ds)
    df = pd.read_csv(PROC / ds / "clean.csv")
    df = df.loc[df.sort_values(by=DATE_COL[ds]).index].reset_index(drop=True)
    df["t"] = pd.to_datetime(df["time"])
    key = df[["latitude", "longitude", "t", "u10", "v10", "msl"]].copy()
    out = era5_physics(X)
    prev = key.copy()
    prev["t"] = prev["t"] + pd.Timedelta(days=1)  # day t-1 values joined onto day t
    m = df[["latitude", "longitude", "t"]].merge(prev, on=["latitude", "longitude", "t"], how="left",
                                                 suffixes=("", "_prev"))
    assert len(m) == len(df), "duplicate (lat, lon, day) keys"
    for c in ["u10", "v10", "msl"]:
        out[f"{c}_prev"] = m[c].to_numpy()
    out["speed_prev"] = np.hypot(out["u10_prev"], out["v10_prev"])
    out["dvec_prev"] = np.hypot(out["u10"] - out["u10_prev"], out["v10"] - out["v10_prev"])
    day = df.groupby("t")
    out["msl_day_std"] = day["msl"].transform("std").to_numpy()  # same-day spatial pressure spread
    out["speed_day_mean"] = pd.Series(np.hypot(df.u10, df.v10)).groupby(df["t"]).transform("mean").to_numpy()
    return out, y, split, rule


def leakfree_df(ds):
    """rolling_mean / cleaned_data: the authors' G2chla_{mean,std,min,max,median,trend,pct_change}_{7,30}
    windows end at the current row, i.e. they contain the target. Recompute them (verified identical to the
    released columns when unshifted) from the target shifted by one row, so only past values enter."""
    df = pd.read_csv(PROC / ds / "clean.csv", low_memory=False)
    s = df["G2chla"].shift(1)
    for w in (7, 30):
        r = s.rolling(w)
        new = {"mean": r.mean(), "std": r.std(), "min": r.min(), "max": r.max(), "median": r.median(),
               "trend": r.apply(lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=True),
               "pct_change": s.pct_change(w, fill_method=None)}
        for k, v in new.items():
            v = v.replace([np.inf, -np.inf], np.nan)
            df[f"G2chla_{k}_{w}"] = v.fillna(v.mean())
    return df
