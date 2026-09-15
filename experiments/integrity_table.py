"""Leakage-aware comparison: paper numbers vs. leakage-free versions of the paper's models vs. our models."""
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

import finalize as FZ
from common import RESULTS, SEED, sequence_splits

summ = pd.read_csv(RESULTS / "improved_summary.csv").set_index("task")
retrain = pd.read_csv(RESULTS / "paper_deep_chronological_retrain.csv")
ckpt = pd.read_csv(RESULTS / "repro_deep_predictions.csv.gz")
tab_lf = pd.read_csv(RESULTS / "paper_tabular_leakfree.csv")
rows = []

# (1) datasets whose best paper model is the released LSTM checkpoint
for ds in ["processed_seq", "hydrographic", "biotoxin"]:
    _, y, (_, _, te) = sequence_splits(ds)
    temp, _ = train_test_split(np.arange(len(y)), test_size=0.15, random_state=SEED)
    unseen = ~np.isin(te, temp)
    pool = FZ.load_pool(FZ.POOLS[ds])
    c = next(iter(pool.values()))
    w = FZ.caruana({n: v["pred_train"][c["va"]] for n, v in pool.items()}, y[c["va"]])
    ours = sum(wt * pool[n]["pred_trainval"] for n, wt in w.items())[te]
    lstm = ckpt[(ckpt.dataset == ds) & (ckpt.model == "LSTM")].y_pred.to_numpy()
    r = retrain[(retrain.dataset == ds) & (retrain.model == "LSTM_chrono_retrain")].R2
    rows.append({"dataset": ds, "paper_reported_R2": r2_score(y[te], lstm),
                 "paper_LSTM_chrono_retrain_R2_mean": r.mean(), "paper_LSTM_chrono_retrain_R2_sd": r.std(),
                 "n_unseen_test_windows": int(unseen.sum()),
                 "paper_LSTM_R2_on_unseen_windows": r2_score(y[te][unseen], lstm[unseen]),
                 "ours_R2_all_test": r2_score(y[te], ours),
                 "ours_R2_on_same_unseen_windows": r2_score(y[te][unseen], ours[unseen])})

# (2) Chl-a tabular datasets whose rolling features contain the target
for ds, pm in [("rolling_mean", "XGB"), ("cleaned_data", "XGB")]:
    lf = tab_lf[tab_lf.dataset == ds].set_index("model").R2
    lstm_r = retrain[(retrain.dataset == ds) & (retrain.model == "LSTM_chrono_retrain")].R2.mean()
    best_name = lf.idxmax() if lf.max() >= lstm_r else "LSTM (chrono retrain)"
    rows.append({"dataset": ds, "paper_reported_R2": summ.loc[ds, "paper_R2"],
                 "paper_best_leakfree_model": best_name, "paper_best_leakfree_R2": max(lf.max(), lstm_r),
                 "ours_leakfree_R2_train_only": summ.loc[f"{ds} (leak-free features)", "R2_train_only"],
                 "ours_leakfree_R2_train_val": summ.loc[f"{ds} (leak-free features)", "R2_train_val"]})

out = pd.DataFrame(rows)
out.to_csv(RESULTS / "integrity_comparison.csv", index=False)
print(out.T.to_string())
