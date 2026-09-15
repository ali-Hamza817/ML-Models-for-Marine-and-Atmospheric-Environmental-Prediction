from pathlib import Path
import pandas as pd, joblib, numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from src.utils_io import read_cfg

ROOT = Path(__file__).resolve().parents[1]

def metrics(y, yhat):
    return {
        "R2":  r2_score(y, yhat),
        "MAE": mean_absolute_error(y, yhat),
        "RMSE": mean_squared_error(y, yhat, squared=False)
    }

def eval_ds(ds_key, cfg):
    data_path = ROOT / "data_proc" / ds_key / "clean.csv"
    if not data_path.exists():
        print(f"跳过 {ds_key}: 数据文件不存在")
        return pd.DataFrame()

    df = pd.read_csv(data_path)
    tcol = cfg["datasets"][ds_key].get("target_col")
    if not tcol:
        print(f"跳过 {ds_key}: 无目标列")
        return pd.DataFrame()

    # 检查目标列是否存在，如果不存在则尝试查找相似的列
    if tcol not in df.columns:
        # 尝试查找包含关键词的列
        possible_cols = [col for col in df.columns if any(keyword in col.lower()
                        for keyword in ['chla', 'chlorophyll', 'target', 'value', 'abundance'])]
        if possible_cols:
            tcol = possible_cols[0]
            print(f"警告 {ds_key}: 使用 {tcol} 替代原目标列")
        else:
            print(f"跳过 {ds_key}: 目标列 {tcol} 不存在")
            return pd.DataFrame()

    X = df.drop(columns=[tcol]).select_dtypes(include=[np.number]).to_numpy()
    y = df[tcol].to_numpy()

    out = []
    model_dir = ROOT / "models" / ds_key
    if not model_dir.exists():
        print(f"跳过 {ds_key}: 模型目录不存在")
        return pd.DataFrame()

    for pkl in model_dir.glob("*.pkl"):
        try:
            mdl = joblib.load(pkl)
            yhat = mdl.predict(X)
            out.append({"dataset": ds_key, "model": pkl.stem, **metrics(y, yhat)})
        except Exception as e:
            print(f"评估失败 {ds_key}/{pkl.stem}: {e}")
            continue

    return pd.DataFrame(out)

if __name__ == "__main__":
    cfg = read_cfg()
    frames = [eval_ds(d, cfg) for d in cfg["datasets"]]
    perf   = pd.concat(frames, ignore_index=True)
    perf.to_csv(ROOT / "tables" / "perf_summary.csv", index=False)
    print("✓ evaluation done ⇒ tables/perf_summary.csv")
