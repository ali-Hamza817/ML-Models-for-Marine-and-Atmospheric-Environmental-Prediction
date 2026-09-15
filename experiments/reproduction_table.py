"""Paper Table 2 / S1 (R², 95% CI, MAE) vs. our reproduction, cell by cell."""
import pandas as pd

from common import REPO, RESULTS

paper = pd.read_csv(REPO / "outputs" / "tables" / "final_table2_model_performance.csv")
paper = paper.rename(columns={"Dataset": "dataset", "Model": "model", "R²": "paper_R2", "R² (95% CI)": "paper_CI", "MAE": "paper_MAE"})
ours = pd.concat([pd.read_csv(RESULTS / "repro_tabular_metrics_pinned.csv"), pd.read_csv(RESULTS / "repro_deep_metrics.csv")])
ours["repro_CI"] = ours.apply(lambda r: f"[{r.CI_low:.3f}, {r.CI_high:.3f}]", axis=1)
t = paper[["dataset", "model", "paper_R2", "paper_CI", "paper_MAE"]].merge(
    ours[["dataset", "model", "R2", "repro_CI", "MAE", "RMSE", "NRMSE"]].rename(columns={"R2": "repro_R2", "MAE": "repro_MAE"}),
    on=["dataset", "model"], how="left")
t["abs_diff_R2"] = (t.paper_R2 - t.repro_R2).abs()
t["match_4dp"] = t.abs_diff_R2 <= 5.1e-5  # 4-dp rounding tolerance (0.431850 -> 0.4318)
t.to_csv(RESULTS / "reproduction_vs_paper.csv", index=False)
print(t.to_string(index=False))
print(f"\n{t.match_4dp.sum()}/{len(t)} cells match within 4-dp rounding; max |ΔR²| = {t.abs_diff_R2.max():.2e}")
