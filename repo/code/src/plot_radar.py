import shap, pandas as pd, numpy as np, matplotlib.pyplot as plt
from pathlib import Path, PureWindowsPath
import joblib

ROOT  = Path(__file__).resolve().parents[1]
ds    = "cleaned_data"          # ←切换数据集
model = "xgb"

# ---------- load ----------
mdl = joblib.load(ROOT / "models" / ds / f"{model}.pkl")
df  = pd.read_csv(ROOT / "data_proc" / ds / "clean.csv")
tcol = df.columns[df.columns.str.contains("G2chla")][0]
X    = df.drop(columns=[tcol]).select_dtypes(include=[np.number])

# ---------- shap ----------
expl = shap.TreeExplainer(mdl.named_steps[model])
sv   = expl.shap_values(X)
mean_abs = np.abs(sv).mean(0)
topN = 10
idx  = np.argsort(mean_abs)[-topN:]
labels = X.columns[idx]
values = mean_abs[idx]

# ---------- radar ----------
angles = np.linspace(0, 2*np.pi, topN, endpoint=False)
values = np.concatenate((values, [values[0]]))
angles = np.concatenate((angles, [angles[0]]))

fig = plt.figure(figsize=(6,6))
ax  = fig.add_subplot(111, polar=True)
ax.plot(angles, values, linewidth=2)
ax.fill(angles, values, alpha=0.3)
ax.set_thetagrids(angles * 180/np.pi, labels, fontsize=8)
ax.set_title(f"Top‑{topN} SHAP importance ({ds})")
plt.tight_layout()
fig.savefig(ROOT / "figures" / f"fig4_radar_{ds}.svg")
print("✓ radar saved ⇒ figures/")
