"""figures/results_chart.png: paper-reported vs improved test R² per dataset (from results/improved_summary.csv)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from common import RESULTS, ROOT  # noqa: E402

# Headline protocol = protocol of the paper's best model (tabular: train-only fit; deep: train+val final fit).
PROTO = {"rolling_mean": "train_only", "cleaned_data": "train_only", "era5_daily": "train_only", "cast": "train_only",
         "hydrographic": "train_val", "biotoxin": "train_val", "processed_seq": "train_val"}
INK, INK2, GRID, PAPER_C, OURS_C = "#0b0b0b", "#52514e", "#e1e0d9", "#86b6ef", "#1c5cab"

s = pd.read_csv(RESULTS / "improved_summary.csv").set_index("task")
rows = [(ds, s.loc[ds, "paper_R2"], s.loc[ds, f"R2_{p}"]) for ds, p in PROTO.items()][::-1]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=200)
for i, (ds, paper, ours) in enumerate(rows):
    ax.plot([paper, ours], [i, i], color="#c3c2b7", lw=2, zorder=1)
    ax.scatter(paper, i, s=70, color=PAPER_C, edgecolor="white", linewidth=2, zorder=2)
    ax.scatter(ours, i, s=70, color=OURS_C, edgecolor="white", linewidth=2, zorder=3)
    ax.text(max(paper, ours) + 0.02, i, f"{paper:.3f} → {ours:.3f}", va="center", color=INK2, fontsize=8)
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([d + (" *" if d == "processed_seq" else "") for d, _, _ in rows], color=INK)
ax.set_xlim(0, 1.12)
ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_xlabel("Test R² (higher is better)", color=INK2)
ax.grid(axis="x", color=GRID, lw=0.6)
ax.set_axisbelow(True)
for sp in ["top", "right", "left"]:
    ax.spines[sp].set_visible(False)
ax.spines["bottom"].set_color("#c3c2b7")
ax.tick_params(colors=INK2, length=0)
ax.scatter([], [], s=70, color=PAPER_C, label="Paper (reported)")
ax.scatter([], [], s=70, color=OURS_C, label="This work (same test rows)")
ax.legend(loc="upper left", frameon=False, fontsize=8, labelcolor=INK)
fig.tight_layout()
(ROOT / "figures").mkdir(exist_ok=True)
fig.savefig(ROOT / "figures" / "results_chart.png", facecolor="white")
