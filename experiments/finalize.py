"""Build the final comparison tables from saved predictions.

Ensemble = Caruana greedy forward selection (with replacement, 30 steps) on VALIDATION predictions of
train-only fits. The resulting weights are applied unchanged to (a) train-only fits and (b) train+val refits.
The test split is only used to report metrics.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from common import RESULTS, REPO, bootstrap_ci_r2

PRED = RESULTS / "preds"
POOLS = {  # design prefixes allowed per task (same inputs as the paper's models; see README for each)
    "rolling_mean": ["rolling_mean__lgb", "rolling_mean__xgb", "rolling_mean__et", "rolling_mean__rf"],
    "cleaned_data": ["cleaned_data__lgb", "cleaned_data__xgb", "cleaned_data__et", "cleaned_data__rf"],
    "era5_daily": ["era5_daily__lgb", "era5_daily__xgb", "era5_daily__et", "era5_daily__rf", "era5_daily__context__"],
    "cast": ["cast__"],
    "processed_seq": ["processed_seq__"],
    "hydrographic": ["hydrographic__"],
    "biotoxin": ["biotoxin__"],
    "rolling_mean (leak-free features)": ["rolling_mean__leakfree__"],
    "cleaned_data (leak-free features)": ["cleaned_data__leakfree__"],
}
# Paper convention: tabular models are fit on train; the deep models' final fit uses train+val.
PAPER_BEST = {"rolling_mean": "XGB", "cleaned_data": "XGB", "era5_daily": "RF", "cast": "RF",
              "processed_seq": "LSTM", "hydrographic": "LSTM", "biotoxin": "LSTM"}


def load_pool(prefixes):
    c = {}
    for f in sorted(PRED.glob("*.npz")):
        name = f.stem
        if any(name.startswith(p) for p in prefixes) and ("leakfree" in name) == any("leakfree" in p for p in prefixes):
            c[name] = dict(np.load(f))
    return c


def caruana(val_preds, y_val, steps=30):
    names = list(val_preds)
    counts = dict.fromkeys(names, 0)
    cur = np.zeros_like(y_val)
    for k in range(1, steps + 1):
        best = max(names, key=lambda n: r2_score(y_val, (cur * (k - 1) + val_preds[n]) / k))
        counts[best] += 1
        cur = (cur * (k - 1) + val_preds[best]) / k
    return {n: c / steps for n, c in counts.items() if c}


def paired_bootstrap(y, p_new, p_old, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        d.append(r2_score(y[i], p_new[i]) - r2_score(y[i], p_old[i]))
    d = np.array(d)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), float((d <= 0).mean())


def score(y, p):
    lo, hi = bootstrap_ci_r2(y, p)
    return {"R2": r2_score(y, p), "CI_low": lo, "CI_high": hi, "MAE": mean_absolute_error(y, p),
            "RMSE": float(np.sqrt(mean_squared_error(y, p)))}


if __name__ == "__main__":
    paper = pd.read_csv(REPO / "outputs" / "tables" / "final_table2_model_performance.csv")
    paper_pred = pd.read_csv(REPO / "outputs" / "tables" / "alternative_metrics_predictions.csv")
    rows, members = [], []
    for task, prefixes in POOLS.items():
        pool = load_pool(prefixes)
        if not pool:
            print("no candidates for", task)
            continue
        any_c = next(iter(pool.values()))
        y, va, te = any_c["y"], any_c["va"], any_c["te"]
        w = caruana({n: c["pred_train"][va] for n, c in pool.items()}, y[va])
        ens_a = sum(wt * pool[n]["pred_train"] for n, wt in w.items())
        ens_b = sum(wt * pool[n]["pred_trainval"] for n, wt in w.items())
        best_single = max(pool, key=lambda n: r2_score(y[va], pool[n]["pred_train"][va]))
        members += [{"task": task, "member": n, "weight": wt,
                     "val_R2": r2_score(y[va], pool[n]["pred_train"][va]),
                     "test_R2_train_only": r2_score(y[te], pool[n]["pred_train"][te]),
                     "test_R2_train_val": r2_score(y[te], pool[n]["pred_trainval"][te])} for n, wt in
                    sorted(w.items(), key=lambda kv: -kv[1])]
        ds = task.split(" ")[0]
        row = {"task": task, "n_candidates": len(pool), "ensemble_val_R2": r2_score(y[va], ens_a[va]),
               "best_single_by_val": best_single}
        for tag, p in [("train_only", ens_a), ("train_val", ens_b)]:
            row.update({f"{k}_{tag}": v for k, v in score(y[te], p[te]).items()})
        if "leak" not in task:
            pm = PAPER_BEST[ds]
            ref = paper[(paper.Dataset == ds) & (paper.Model == pm)].iloc[0]
            pp = paper_pred[(paper_pred.dataset == ds) & (paper_pred.model == pm.lower())]
            assert len(pp) == len(te) and np.allclose(pp.y_true.to_numpy(), y[te]), "test sets differ"
            row.update({"paper_best_model": pm, "paper_R2": float(ref["R²"]), "paper_MAE": float(ref["MAE"])})
            for tag, p in [("train_only", ens_a), ("train_val", ens_b)]:
                lo, hi, pv = paired_bootstrap(y[te], p[te], pp.y_pred.to_numpy())
                row.update({f"dR2_CI_low_{tag}": lo, f"dR2_CI_high_{tag}": hi, f"p_no_gain_{tag}": pv})
        rows.append(row)
        print(f"{task:36s} val={row['ensemble_val_R2']:.4f} test train-only={row['R2_train_only']:.4f} "
              f"train+val={row['R2_train_val']:.4f}  paper={row.get('paper_R2', float('nan')):.4f}", flush=True)
    pd.DataFrame(rows).to_csv(RESULTS / "improved_summary.csv", index=False)
    pd.DataFrame(members).to_csv(RESULTS / "improved_ensemble_members.csv", index=False)
