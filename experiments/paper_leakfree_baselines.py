"""Paper's tabular models (exact legacy settings) on leak-free target-history features."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

import features as F
from common import RESULTS, metrics, tabular_splits
from reproduce_tabular import models

rows = []
for ds in ["rolling_mean", "cleaned_data"]:
    X, y, (tr, va, te), rule = tabular_splits(ds, F.leakfree_df(ds))
    sc = RobustScaler().fit(X.iloc[tr])
    for name, m in models().items():
        p = m.fit(sc.transform(X.iloc[tr]), y.iloc[tr]).predict(sc.transform(X.iloc[te]))
        rows.append({"dataset": ds, "model": name, "features": "leak-free", **metrics(y.iloc[te], p, y.iloc[tr])})
        print(ds, name, round(rows[-1]["R2"], 4), flush=True)
pd.DataFrame(rows).to_csv(RESULTS / "paper_tabular_leakfree.csv", index=False)
