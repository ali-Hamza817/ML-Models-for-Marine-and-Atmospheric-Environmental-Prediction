#!/usr/bin/env python3
"""
Generate dataset split summary with date ranges.
"""

from pathlib import Path
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_config():
    cfg_path = REPO_ROOT / "configs" / "config.yaml"
    if not cfg_path.exists():
        cfg_path = REPO_ROOT / "code" / "src" / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def infer_date_col(df, meta):
    if meta and "date_col" in meta:
        return meta["date_col"]
    for col in df.columns:
        if "date" in col.lower() or "time" in col.lower():
            return col
    return None


def main():
    cfg = load_config()
    train_ratio = cfg.get("global", {}).get("train_ratio", 0.7)
    val_ratio = cfg.get("global", {}).get("val_ratio", 0.15)

    rows = []
    for dataset, meta in cfg.get("datasets", {}).items():
        data_path = REPO_ROOT / "data" / "processed" / dataset / "clean.csv"
        if not data_path.exists():
            continue
        df = pd.read_csv(data_path)
        date_col = infer_date_col(df, meta)
        if not date_col or date_col not in df.columns:
            rows.append({
                "Dataset": dataset,
                "Split Rule": "random",
                "Train Range": "N/A",
                "Val Range": "N/A",
                "Test Range": "N/A",
                "Period": "N/A",
            })
            continue

        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col]).sort_values(date_col)
        if df.empty:
            continue

        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        train = df.iloc[:train_end]
        val = df.iloc[train_end:val_end]
        test = df.iloc[val_end:]

        rows.append({
            "Dataset": dataset,
            "Split Rule": "time-ordered",
            "Train Range": f"{train[date_col].min().date()} to {train[date_col].max().date()}",
            "Val Range": f"{val[date_col].min().date()} to {val[date_col].max().date()}",
            "Test Range": f"{test[date_col].min().date()} to {test[date_col].max().date()}",
            "Period": f"{df[date_col].min().date()} to {df[date_col].max().date()}",
        })

    out_dir = REPO_ROOT / "outputs" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "dataset_split_summary.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
