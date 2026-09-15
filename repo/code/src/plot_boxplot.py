import pandas as pd, matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "tables" / "perf_summary.csv")
plt.figure(figsize=(10,4))
df.boxplot(column="RMSE", by="dataset", rot=45)
plt.tight_layout()
plt.suptitle(""); plt.ylabel("RMSE")
plt.savefig(ROOT / "figures" / "fig3_boxplot.svg")
print("✓ boxplot saved ⇒ figures/fig3_boxplot.svg")
