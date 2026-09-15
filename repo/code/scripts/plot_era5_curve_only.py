#!/usr/bin/env python3
"""
Re-plot era5_downsample_curve from existing CSV results (no model retraining).
Fixes:
  1. y-axis label: R² (LaTeX) instead of "R2"
  2. Rightmost annotations (50k/100k) shifted downward to avoid legend overlap
  3. Wider ylim bottom margin so the shaded band is not clipped
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

REPO_ROOT = Path(__file__).resolve().parents[2]
CSV_PATH  = REPO_ROOT / "outputs" / "tables" / "era5_downsample_ablation.csv"
FIG_DIR   = REPO_ROOT / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV_PATH)

# Keep only the 5 canonical sizes (drop the full-dataset row if present)
KEEP = {1000, 5000, 10000, 50000, 100000}
df = df[df["size"].isin(KEEP)].sort_values("size")

sizes_arr = np.asarray(df["size"].values,    dtype=float)
means_arr = np.asarray(df["r2_mean"].values, dtype=float)
stds_arr  = np.asarray(df["r2_std"].values,  dtype=float)
lower     = means_arr - stds_arr
upper     = means_arr + stds_arr

color_main = "#1f4e79"
color_band = "#4e79a7"

fig, ax = plt.subplots(figsize=(7.0, 4.4))
ax.set_facecolor("#f6f8fb")
ax.grid(True, which="major", color="#d9dee7", linewidth=0.9, alpha=0.9)
ax.grid(True, which="minor", color="#e7ebf2", linewidth=0.6, alpha=0.7)

ax.fill_between(sizes_arr, lower, upper, color=color_band, alpha=0.18, linewidth=0)
ax.plot(
    sizes_arr, means_arr, "-o",
    color=color_main, linewidth=2.6, markersize=6.8,
    markerfacecolor="white", markeredgewidth=2.0,
)

# Annotations: rightmost two points go BELOW the marker to avoid legend overlap
large_sizes = sorted(sizes_arr)[-2:]  # 50k, 100k
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
        ha="center", va=va,
        fontsize=9, color="#1f2933",
        bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                  edgecolor="none", alpha=0.9),
    )

ax.set_xscale("log")
ax.set_xlabel("Sample size (log scale)", fontsize=12, color="#111827")
# FIX 1: proper R² label
ax.set_ylabel("R\u00b2 on held-out test", fontsize=12, color="#111827")
ax.tick_params(axis="both", labelsize=10.5, colors="#111827")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

tick_values = [1000, 5000, 10000, 50000, 100000]
ax.set_xticks(tick_values)
ax.xaxis.set_major_formatter(FuncFormatter(
    lambda x, _: f"{int(x/1000)}k" if x >= 1000 else str(int(x))
))

# FIX 3: wider bottom margin so shaded band isn't clipped
ax.set_ylim(min(lower) - 0.03, max(upper) + 0.015)

ax.text(
    0.02, 0.96,
    "Shaded band: +/-1 std over repeats",
    transform=ax.transAxes,
    ha="left", va="top", fontsize=9.5, color="#334155",
    bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
              edgecolor="#e5e7eb", alpha=0.95),
)

fig.tight_layout()
for ext in ["png", "pdf", "tiff"]:
    out = FIG_DIR / f"era5_downsample_curve.{ext}"
    fig.savefig(out, dpi=600, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out}")

plt.close(fig)
print("Done.")
