#!/usr/bin/env python3
"""
Recompute hydrographic traditional-ML results with a consistent split/baseline.

This script:
1) Uses a single, explicit stratified random split (seeded).
2) Evaluates all models and the train-mean baseline in the SAME space.
3) Bootstraps on the test set to provide r2_mean/r2_std/CI.
4) Overwrites ONLY the hydrographic rows in updated_detailed_results.csv.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import xgboost as xgb

REPO_ROOT = Path(__file__).resolve().parents[2]
CFG_PATHS = [
    REPO_ROOT / "configs" / "config.yaml",
    REPO_ROOT / "code" / "src" / "config.yaml",
]
BEST_PARAMS_PATH = REPO_ROOT / "logs" / "best_hyperparameters.csv"
UPDATED_RESULTS_PATH = REPO_ROOT / "outputs" / "tables" / "old tables" / "updated_detailed_results.csv"

DATASET = "hydrographic"


def load_config() -> Dict:
    for path in CFG_PATHS:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
    raise FileNotFoundError("config.yaml not found in expected locations.")


def load_best_params(dataset: str, model_name: str) -> Dict:
    if not BEST_PARAMS_PATH.exists():
        return {}
    df = pd.read_csv(BEST_PARAMS_PATH)
    row = df[(df["Dataset"] == dataset) & (df["Model"] == model_name)]
    if row.empty:
        return {}
    raw = row.iloc[0]["Best_Parameters"]
    try:
        params = json.loads(raw)
    except Exception:
        return {}
    params.pop("random_state", None)
    return params


def get_target_col(cfg: Dict, df: pd.DataFrame) -> str:
    target_col = cfg.get("datasets", {}).get(DATASET, {}).get("target_col")
    if target_col in df.columns:
        return target_col
    # Fallback to a chlorophyll column if naming differs.
    candidates = [c for c in df.columns if "chlorophyll" in c.lower() or "chla" in c.lower()]
    if candidates:
        return candidates[0]
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if not len(numeric_cols):
        raise ValueError("No numeric target column found.")
    return numeric_cols[-1]


def infer_date_col(df: pd.DataFrame) -> str | None:
    for col in df.columns:
        lower = col.lower()
        if "date" in lower or "time" in lower:
            return col
    return None


def prepare_features(df: pd.DataFrame, target_col: str, date_col: str | None) -> pd.DataFrame:
    exclude = {target_col}
    if date_col and date_col in df.columns:
        exclude.add(date_col)
    # Drop any obvious time/date columns to avoid leakage via ordinal dates.
    for col in df.columns:
        if col.lower() in {"date", "time"}:
            exclude.add(col)
    X = df[[c for c in df.columns if c not in exclude]].select_dtypes(include=[np.number]).copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.mean(numeric_only=True))
    return X


def stratified_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float,
    seed: int,
    n_bins: int = 10,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    y_series = pd.Series(y)
    try:
        bins = pd.qcut(y_series, q=n_bins, duplicates="drop")
        stratify = bins.astype(str)
        return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=stratify)
    except Exception:
        return train_test_split(X, y, test_size=test_size, random_state=seed)


def _xgb_params() -> Tuple[Dict, bool]:
    params: Dict = {}
    use_gpu = os.getenv("XGB_USE_GPU", "0") == "1"
    if use_gpu:
        params["tree_method"] = "gpu_hist"
        params["predictor"] = "gpu_predictor"
        params["gpu_id"] = 0
    return params, use_gpu


def build_models(seed: int) -> Dict[str, object]:
    rf_params = load_best_params(DATASET, "RandomForest")
    if not rf_params:
        rf_params = {
            "n_estimators": 300,
            "max_depth": 20,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
        }
    rf_params["n_estimators"] = int(min(500, rf_params.get("n_estimators", 300)))

    xgb_params, use_gpu = _xgb_params()
    xgb_model_params = {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "random_state": seed,
        "verbosity": 0,
        **xgb_params,
    }

    models: Dict[str, object] = {
        "rf": RandomForestRegressor(random_state=seed, **rf_params),
        "xgb": xgb.XGBRegressor(**xgb_model_params),
        "svr": SVR(C=10.0, gamma="scale", epsilon=0.05),
        "ridge": Ridge(alpha=1.0, random_state=seed),
        "lasso": Lasso(alpha=0.001, random_state=seed, max_iter=5000),
    }

    if use_gpu:
        print("XGB GPU enabled via XGB_USE_GPU=1")

    return models


def bootstrap_r2(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        scores.append(r2_score(y_true[idx], y_pred[idx]))
    scores_arr = np.array(scores, dtype=float)
    return {
        "r2_mean": float(np.mean(scores_arr)),
        "r2_std": float(np.std(scores_arr)),
        "ci_lower": float(np.percentile(scores_arr, 2.5)),
        "ci_upper": float(np.percentile(scores_arr, 97.5)),
    }


def format_r2(mean: float, std: float) -> str:
    return f"{mean:.3f}\u00b1{std:.3f}"


def recompute(seed: int = 42) -> pd.DataFrame:
    cfg = load_config()
    global_cfg = cfg.get("global", {})
    test_ratio = float(global_cfg.get("test_ratio", 0.15))
    n_bootstrap = int(global_cfg.get("bootstrap_samples", 1000))

    data_path = REPO_ROOT / "data" / "processed" / DATASET / "clean.csv"
    df = pd.read_csv(data_path)

    target_col = get_target_col(cfg, df)
    date_col = infer_date_col(df)
    if date_col and date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    X = prepare_features(df, target_col, date_col)
    y = df[target_col].astype(float)
    mask = ~(X.isna().any(axis=1) | y.isna())
    X = X.loc[mask].reset_index(drop=True)
    y = y.loc[mask].reset_index(drop=True)

    X_train, X_test, y_train, y_test = stratified_split(X, y, test_size=test_ratio, seed=seed)

    # Standardize features for linear/SVR models (and keep consistent for all models).
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results: List[Dict[str, float | str]] = []

    # Baseline: train mean in raw space.
    baseline_pred = np.full(shape=len(y_test), fill_value=float(np.mean(y_train)), dtype=float)
    baseline_stats = bootstrap_r2(y_test.to_numpy(), baseline_pred, n_bootstrap=n_bootstrap, seed=seed)
    results.append(
        {
            "dataset": DATASET,
            "model": "mean",
            **baseline_stats,
            "r2_formatted": format_r2(baseline_stats["r2_mean"], baseline_stats["r2_std"]),
        }
    )

    models = build_models(seed=seed)
    for name, model in models.items():
        try:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        except Exception as exc:
            # GPU failures fall back to CPU for XGB.
            if name == "xgb" and os.getenv("XGB_USE_GPU", "0") == "1":
                print(f"XGB GPU failed, fallback to CPU: {exc}")
                cpu_model = xgb.XGBRegressor(
                    n_estimators=300,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    random_state=seed,
                    verbosity=0,
                )
                cpu_model.fit(X_train_scaled, y_train)
                y_pred = cpu_model.predict(X_test_scaled)
            else:
                raise

        stats = bootstrap_r2(y_test.to_numpy(), np.asarray(y_pred, dtype=float), n_bootstrap=n_bootstrap, seed=seed)
        results.append(
            {
                "dataset": DATASET,
                "model": name,
                **stats,
                "r2_formatted": format_r2(stats["r2_mean"], stats["r2_std"]),
            }
        )

    # Ridge/Lasso baseline rows should exist alongside mean for downstream scripts.
    # We map ridge/lasso results if present; otherwise we add placeholders.
    models_present = {row["model"] for row in results}
    for needed in ("ridge", "lasso"):
        if needed not in models_present:
            results.append(
                {
                    "dataset": DATASET,
                    "model": needed,
                    "r2_mean": np.nan,
                    "r2_std": np.nan,
                    "ci_lower": np.nan,
                    "ci_upper": np.nan,
                    "r2_formatted": "N/A",
                }
            )

    out_df = pd.DataFrame(results)
    # Ensure deterministic ordering for hydrographic rows.
    order = ["rf", "xgb", "svr", "mean", "ridge", "lasso"]
    out_df["model"] = pd.Categorical(out_df["model"], categories=order, ordered=True)
    out_df = out_df.sort_values("model").reset_index(drop=True)
    return out_df


def update_updated_results(hydro_df: pd.DataFrame) -> None:
    if not UPDATED_RESULTS_PATH.exists():
        raise FileNotFoundError(f"Missing updated results file: {UPDATED_RESULTS_PATH}")
    updated = pd.read_csv(UPDATED_RESULTS_PATH)
    updated = updated[updated["dataset"] != DATASET]
    combined = pd.concat([updated, hydro_df], ignore_index=True)
    combined.to_csv(UPDATED_RESULTS_PATH, index=False)
    print(f"Updated hydrographic rows in: {UPDATED_RESULTS_PATH}")


def main() -> None:
    seed = 42
    hydro_df = recompute(seed=seed)
    print("Recomputed hydrographic results:")
    print(hydro_df.to_string(index=False))
    update_updated_results(hydro_df)


if __name__ == "__main__":
    main()

