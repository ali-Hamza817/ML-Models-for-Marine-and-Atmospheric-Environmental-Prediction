#!/usr/bin/env python3
"""
ERA5 downsampling ablation to test sample size effects.
"""

import argparse
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SIZES = [100000, 50000, 10000, 5000, 1000]


def load_best_params(dataset, model):
    params_path = REPO_ROOT / "logs" / "best_hyperparameters.csv"
    if not params_path.exists():
        return {}
    df = pd.read_csv(params_path)
    row = df[(df["Dataset"] == dataset) & (df["Model"].str.lower() == model.lower())]
    if row.empty:
        return {}
    raw = row.iloc[0]["Best_Parameters"]
    try:
        return json.loads(raw)
    except Exception:
        return {}


def time_split(df, date_col, train_ratio=0.7, val_ratio=0.15):
    df_sorted = df.sort_values(date_col).reset_index(drop=True)
    n = len(df_sorted)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    train = df_sorted.iloc[:train_end]
    val = df_sorted.iloc[train_end:val_end]
    test = df_sorted.iloc[val_end:]
    return train, val, test


def run_one(df, target_col, date_col, params, seed):
    train, val, test = time_split(df, date_col)
    X_train = train.drop(columns=[target_col]).select_dtypes(include=[np.number]).values
    y_train = train[target_col].values
    X_test = test.drop(columns=[target_col]).select_dtypes(include=[np.number]).values
    y_test = test[target_col].values

    model = RandomForestRegressor(random_state=seed, **params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return r2_score(y_test, y_pred)


def main():
    parser = argparse.ArgumentParser(description="ERA5 downsampling ablation")
    parser.add_argument("--sizes", nargs="+", type=int, default=DEFAULT_SIZES)
    parser.add_argument("--model", default="rf")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_path = REPO_ROOT / "data" / "processed" / "era5_daily" / "clean.csv"
    df = pd.read_csv(data_path)
    target_col = "wind10" if "wind10" in df.columns else df.select_dtypes(include=[np.number]).columns[-1]
    date_col = "time" if "time" in df.columns else None
    if not date_col:
        date_candidates = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
        date_col = date_candidates[0] if date_candidates else None
    if not date_col:
        raise ValueError("No date column found for time split")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    params = load_best_params("era5_daily", "RandomForest")
    if not params:
        params = {"n_estimators": 200, "max_depth": 15, "min_samples_split": 5, "min_samples_leaf": 2}
    if "random_state" in params:
        params.pop("random_state")

    results = []

    requested_sizes = [min(int(s), len(df)) for s in args.sizes]
    if len(df) not in requested_sizes:
        requested_sizes = [len(df)] + requested_sizes

    for size in requested_sizes:
        size = min(size, len(df))
        r2_scores = []
        for rep in range(args.repeats):
            sample = df.sample(n=size, random_state=int(args.seed + rep))
            r2 = run_one(sample, target_col, date_col, params, seed=int(args.seed + rep))
            r2_scores.append(r2)
        results.append({
            "dataset": "era5_daily",
            "size": size,
            "r2_mean": float(np.mean(r2_scores)),
            "r2_std": float(np.std(r2_scores)),
            "repeats": args.repeats,
            "seed": int(args.seed),
        })

    results = sorted(results, key=lambda r: r["size"], reverse=True)

    out_dir = REPO_ROOT / "outputs" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "era5_downsample_ablation.csv"
    pd.DataFrame(results).to_csv(out_path, index=False)

    # Plot
    fig_dir = REPO_ROOT / "outputs" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    sizes = [r["size"] for r in results]
    means = [r["r2_mean"] for r in results]
    stds = [r["r2_std"] for r in results]
    color_main = "#1f4e79"
    color_band = "#4e79a7"

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.set_facecolor("#f6f8fb")
    ax.grid(True, which="major", color="#d9dee7", linewidth=0.9, alpha=0.9)
    ax.grid(True, which="minor", color="#e7ebf2", linewidth=0.6, alpha=0.7)

    sizes_arr = np.asarray(sizes, dtype=float)
    means_arr = np.asarray(means, dtype=float)
    stds_arr = np.asarray(stds, dtype=float)
    lower = means_arr - stds_arr
    upper = means_arr + stds_arr

    ax.fill_between(sizes_arr, lower, upper, color=color_band, alpha=0.18, linewidth=0)
    ax.plot(
        sizes_arr,
        means_arr,
        "-o",
        color=color_main,
        linewidth=2.6,
        markersize=6.8,
        markerfacecolor="white",
        markeredgewidth=2.0,
    )

    # Annotate key points without cluttering the curve.
    # Rightmost two points (50k, 100k) shift downward to avoid legend overlap.
    large_sizes = sorted(sizes_arr)[-2:]  # 50k and 100k
    for x_val, y_val in zip(sizes_arr, means_arr):
        if x_val in large_sizes:
            y_offset, va = -14, "top"
        else:
            y_offset, va = 7, "bottom"
        ax.annotate(
            f"{y_val:.3f}",
            xy=(x_val, y_val),
            xytext=(0, y_offset),
            textcoords="offset points",
            ha="center",
            va=va,
            fontsize=9,
            color="#1f2933",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.9),
        )

    ax.set_xscale("log")
    ax.set_xlabel("Sample size (log scale)", fontsize=12, color="#111827")
    ax.set_ylabel("R\u00b2 on held-out test", fontsize=12, color="#111827")
    ax.tick_params(axis="both", labelsize=10.5, colors="#111827")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Use a compact set of readable log ticks.
    tick_values = [1000, 5000, 10000, 50000, 100000]
    ax.set_xticks(tick_values)

    def fmt_k(x_val, _pos):
        if x_val >= 1000:
            return f"{int(x_val/1000)}k"
        return f"{int(x_val)}"

    ax.xaxis.set_major_formatter(FuncFormatter(fmt_k))
    ax.set_ylim(min(lower) - 0.03, max(upper) + 0.015)

    ax.text(
        0.02,
        0.96,
        "Shaded band: +/-1 std over repeats",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.5,
        color="#334155",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#e5e7eb", alpha=0.95),
    )

    fig.tight_layout()
    for ext in ["png", "pdf", "tiff"]:
        fig_path = fig_dir / f"era5_downsample_curve.{ext}"
        fig.savefig(fig_path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Saved ablation table: {out_path}")
    print(f"Saved ablation figure(s): {fig_dir / 'era5_downsample_curve.*'}")


if __name__ == "__main__":
    main()
