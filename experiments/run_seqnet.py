"""Neural sequence candidates -> results/preds/<ds>__<rep>__<kind>.npz (same format as improve_trees.py)."""
import json
import sys

import numpy as np
from sklearn.metrics import r2_score

import seqnet
from common import RESULTS, sequence_splits, tabular_splits

ds, rep, kind, dev = sys.argv[1:5]
S, y, split = sequence_splits(ds)
Z = tabular_splits(ds)[0].to_numpy(float)[30:30 + len(y)] if rep == "hybrid" else np.zeros((len(y), 0))
tr, va, te = split
pa, vr2 = seqnet.fit_predict(S, Z, y, split, kind=kind, seeds=5, refit=False, device=dev)
pb, _ = seqnet.fit_predict(S, Z, y, split, kind=kind, seeds=5, refit=True, device=dev)
tag = f"{ds}__{rep}__{kind}"
out = RESULTS / "preds"
out.mkdir(parents=True, exist_ok=True)
np.savez(out / f"{tag}.npz", y=y, tr=tr, va=va, te=te, pred_train=pa, pred_trainval=pb)
(out / f"{tag}.json").write_text(json.dumps({"seeds": 5, "val_r2": vr2}))
print(f"{tag:36s} val={r2_score(y[va], pa[va]):.4f} test(train-only)={r2_score(y[te], pa[te]):.4f} "
      f"test(train+val)={r2_score(y[te], pb[te]):.4f}", flush=True)
