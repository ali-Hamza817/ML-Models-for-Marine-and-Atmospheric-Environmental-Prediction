"""Leakage audit of the released LSTM/Transformer checkpoints.

train_enhanced.py (the script that saved models/*.pth) splits sequences with two *shuffled*
train_test_split calls (seed 42), fits scaler_X on the random 70% train part and finally trains on
train+val (random 85%). The paper then scores the checkpoints on the chronological last 15%.
If that is how the checkpoints were made:
  (1) the pickled scaler_X statistics equal min/max of the random-70% subset, and
  (2) ~85% of the chronological test windows were in the checkpoint's training data.
We verify (1) exactly and report R² separately on seen vs unseen test windows.
"""
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

import reproduce_deep as R
from common import DEEP_DATASETS, REPO, RESULTS, SEED, sequence_splits

rows = []
for ds in DEEP_DATASETS:
    S, y, (_, _, te) = sequence_splits(ds)
    idx = np.arange(len(y))
    temp, rand_test = train_test_split(idx, test_size=0.15, random_state=SEED)  # 1 - 0.70 - 0.15
    rand_train, rand_val = train_test_split(temp, test_size=0.15 / 0.85, random_state=SEED)
    X32 = S.astype(np.float32)
    flat = X32[rand_train].reshape(-1, S.shape[-1])
    for name in ["lstm", "transformer"]:
        ck = torch.load(REPO / "models" / ds / f"{name}.pth", map_location="cpu")
        sx = ck["scaler_X"]
        scaler_match = bool(np.allclose(sx.data_min_, flat.min(0)) and np.allclose(sx.data_max_, flat.max(0)))
        seen = np.isin(te, temp)
        p = R.checkpoint_predict(ds, name, S[te])
        rows.append({
            "dataset": ds, "model": name.upper(), "scaler_X_matches_random70_train": scaler_match,
            "test_windows": len(te), "test_windows_seen_in_training": int(seen.sum()),
            "pct_seen": round(100 * seen.mean(), 1),
            "R2_all_test": r2_score(y[te], p),
            "R2_seen_subset": r2_score(y[te][seen], p[seen]),
            "R2_unseen_subset": r2_score(y[te][~seen], p[~seen]),
        })
        print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(RESULTS / "audit_checkpoint_leakage.csv", index=False)
