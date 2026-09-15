import sys, numpy as np
from sklearn.metrics import r2_score
from common import sequence_splits, tabular_splits
import seqnet
ds, kind, hyb, dev = sys.argv[1], sys.argv[2], sys.argv[3] == "1", sys.argv[4]
S, y, split = sequence_splits(ds)
Z = tabular_splits(ds)[0].to_numpy(float)[30:30 + len(y)] if hyb else np.zeros((len(y), 0))
tr, va, te = split
for refit in (False, True):
    p, vr2 = seqnet.fit_predict(S, Z, y, split, kind=kind, seeds=3, refit=refit, device=dev)
    print(f"{ds:14s} {kind:4s} hybrid={hyb} refit={refit} val={vr2:.4f} test={r2_score(y[te], p[te]):.4f}", flush=True)
