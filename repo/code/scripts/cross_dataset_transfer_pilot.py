#!/usr/bin/env python3
"""
Cross-dataset transfer pilot (Chl-a): train on one dataset, test on another.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_dataset(name, target_col, max_samples=None, seed=42):
    path = REPO_ROOT / "data" / "processed" / name / "clean.csv"
    df = pd.read_csv(path)
    if max_samples and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=seed)
    if target_col not in df.columns:
        candidates = [c for c in df.columns if "chla" in c.lower()]
        target = candidates[0]
    else:
        target = target_col
    X = df.drop(columns=[target]).select_dtypes(include=[np.number])
    y = df[target].values
    return X, y


def evaluate_transfer(train_ds, test_ds, target_col="G2chla", max_samples=None, seed=42, n_estimators=100):
    X_train, y_train = load_dataset(train_ds, target_col, max_samples=max_samples, seed=seed)
    X_test, y_test = load_dataset(test_ds, target_col, max_samples=max_samples, seed=seed)

    common_cols = [c for c in X_train.columns if c in X_test.columns]
    X_train = X_train[common_cols]
    X_test = X_test[common_cols]

    model = RandomForestRegressor(n_estimators=n_estimators, random_state=seed)
    model.fit(X_train.values, y_train)
    y_pred = model.predict(X_test.values)
    return {
        "train_dataset": train_ds,
        "test_dataset": test_ds,
        "n_features": len(common_cols),
        "r2": r2_score(y_test, y_pred),
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": mean_squared_error(y_test, y_pred, squared=False),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Cross-dataset transfer pilot")
    parser.add_argument("--max-samples", type=int, default=5000)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pairs = [
        ("cleaned_data", "rolling_mean"),
        ("rolling_mean", "cleaned_data"),
    ]
    results = [
        evaluate_transfer(a, b, max_samples=args.max_samples, seed=args.seed, n_estimators=args.n_estimators)
        for a, b in pairs
    ]
    out_dir = REPO_ROOT / "outputs" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "transfer_pilot.csv"
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
