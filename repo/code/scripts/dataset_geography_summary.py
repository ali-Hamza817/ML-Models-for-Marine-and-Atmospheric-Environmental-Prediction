#!/usr/bin/env python3
"""
Generate geographic summary (lat/lon bounds) for each dataset.
"""

from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]


LAT_KEYS = ["lat", "latitude"]
LON_KEYS = ["lon", "longitude"]


def find_lat_lon_columns(df):
    lat_candidates = []
    lon_candidates = []
    for col in df.columns:
        low = col.lower()
        if any(k in low for k in LAT_KEYS):
            lat_candidates.append(col)
        if any(k in low for k in LON_KEYS):
            lon_candidates.append(col)

    lat_col = lat_candidates[0] if lat_candidates else None
    lon_col = lon_candidates[0] if lon_candidates else None
    return lat_col, lon_col


def main():
    data_dir = REPO_ROOT / "data" / "processed"
    rows = []
    for dataset_dir in sorted([p for p in data_dir.iterdir() if p.is_dir()]):
        dataset = dataset_dir.name
        data_path = dataset_dir / "clean.csv"
        if not data_path.exists():
            continue
        df = pd.read_csv(data_path)
        lat_col, lon_col = find_lat_lon_columns(df)
        if lat_col and lon_col:
            lat_min = df[lat_col].min()
            lat_max = df[lat_col].max()
            lon_min = df[lon_col].min()
            lon_max = df[lon_col].max()
            rows.append({
                "Dataset": dataset,
                "Latitude Column": lat_col,
                "Longitude Column": lon_col,
                "Lat Min": lat_min,
                "Lat Max": lat_max,
                "Lon Min": lon_min,
                "Lon Max": lon_max,
            })
        else:
            rows.append({
                "Dataset": dataset,
                "Latitude Column": "N/A",
                "Longitude Column": "N/A",
                "Lat Min": "N/A",
                "Lat Max": "N/A",
                "Lon Min": "N/A",
                "Lon Max": "N/A",
            })

    out_dir = REPO_ROOT / "outputs" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "dataset_geography_summary.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
