#!/usr/bin/env python3
"""
Hydrographic baseline diagnostic.
Checks baseline R2 using training-mean and compares with trained models.
Outputs train/test target distributions and a summary table.
"""

from pathlib import Path
import os

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import xgboost as xgb
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_config():
    cfg_path = REPO_ROOT / "configs" / "config.yaml"
    if not cfg_path.exists():
        cfg_path = REPO_ROOT / "code" / "src" / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_target_col(cfg, dataset):
    target_col = cfg.get("datasets", {}).get(dataset, {}).get("target_col")
    return target_col


def infer_date_col(df):
    for col in df.columns:
        if "date" in col.lower() or "time" in col.lower():
            return col
    return None


def stratified_split(X, y, test_size=0.2, seed=42, n_bins=10):
    y_series = pd.Series(y)
    try:
        bins = pd.qcut(y_series, q=n_bins, duplicates="drop")
        return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=bins)
    except Exception:
        return train_test_split(X, y, test_size=test_size, random_state=seed)

def _xgb_params():
    params = {}
    use_gpu = os.getenv("XGB_USE_GPU", "0") == "1"
    if use_gpu:
        params["tree_method"] = "gpu_hist"
        params["predictor"] = "gpu_predictor"
        params["gpu_id"] = 0
    return params, use_gpu


def main():
    dataset = "hydrographic"
    cfg = load_config()
    test_ratio = float(cfg.get("global", {}).get("test_ratio", 0.15))
    data_path = REPO_ROOT / "data" / "processed" / dataset / "clean.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"missing data: {data_path}")

    df = pd.read_csv(data_path)
    target_col = get_target_col(cfg, dataset)
    if not target_col or target_col not in df.columns:
        # fallback by name
        candidates = [c for c in df.columns if "chlorophyll" in c.lower() or "chla" in c.lower()]
        target_col = candidates[0] if candidates else df.select_dtypes(include=[np.number]).columns[-1]

    date_col = infer_date_col(df)
    if date_col and date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    X = df.drop(columns=[target_col]).select_dtypes(include=[np.number])
    y = df[target_col].values

    X_train, X_test, y_train, y_test = stratified_split(X, y, test_size=test_ratio, seed=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Baseline: training mean
    baseline_pred = np.full_like(y_test, fill_value=np.mean(y_train), dtype=float)
    baseline_r2 = r2_score(y_test, baseline_pred)

    results = [{
        "dataset": dataset,
        "model": "train_mean",
        "r2": baseline_r2,
        "baseline_r2": baseline_r2,
        "delta_r2": 0.0,
        "n_train": len(y_train),
        "n_test": len(y_test),
    }]

    xgb_params, use_gpu = _xgb_params()
    if use_gpu:
        print("XGB GPU enabled via XGB_USE_GPU=1")

    models = {
        "rf": RandomForestRegressor(
            n_estimators=300,
            max_depth=20,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features="sqrt",
            random_state=42,
        ),
        "svr": SVR(C=10.0, gamma="scale", epsilon=0.05),
        "xgb": xgb.XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            verbosity=0,
            **xgb_params,
        ),
    }

    for model_name, model in models.items():
        try:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        except Exception as exc:
            if model_name == "xgb" and use_gpu:
                print(f"XGB GPU failed, fallback to CPU: {exc}")
                cpu_model = xgb.XGBRegressor(
                    n_estimators=300,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    random_state=42,
                    verbosity=0,
                )
                cpu_model.fit(X_train_scaled, y_train)
                y_pred = cpu_model.predict(X_test_scaled)
            else:
                raise

        r2 = r2_score(y_test, y_pred)
        results.append({
            "dataset": dataset,
            "model": model_name,
            "r2": r2,
            "baseline_r2": baseline_r2,
            "delta_r2": r2 - baseline_r2,
            "n_train": len(y_train),
            "n_test": len(y_test),
        })

    out_dir = REPO_ROOT / "outputs" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "hydrographic_baseline_diagnostics.csv"
    pd.DataFrame(results).to_csv(out_path, index=False)

    # Plot target distribution
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(y_train, bins=30, alpha=0.7, label="train")
    axes[0].hist(y_test, bins=30, alpha=0.7, label="test")
    axes[0].set_title("Target distribution (hist)")
    axes[0].legend()

    axes[1].boxplot([y_train, y_test], labels=["train", "test"])
    axes[1].set_title("Target distribution (box)")

    fig.tight_layout()
    fig_dir = REPO_ROOT / "outputs" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_path = fig_dir / "hydrographic_target_distribution.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved diagnostics: {out_path}")
    print(f"Saved figure: {fig_path}")


if __name__ == "__main__":
    main()
