import argparse, numpy as np, pandas as pd
from pathlib import Path
from sklearn.impute import KNNImputer
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import SelectKBest, f_regression
from tqdm import tqdm
from src.utils_io import read_cfg, save_df
import warnings
warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[1]

def load_table(meta):
    fp = ROOT / meta["file"]
    if meta["loader"] == "excel":
        df = pd.read_excel(fp, sheet_name=meta.get("sheet", 0))

        # 特殊处理phyto_wide数据集：跳过第一行（毒素类型标题）
        if meta.get("sheet") == "Phytoplankton":
            df = df.iloc[1:].reset_index(drop=True)  # 跳过第0行，从第1行开始

    else:
        df = pd.read_csv(fp, low_memory=False)
    for c in meta.get("drop_cols", []):
        if c in df.columns:
            df = df.drop(columns=c)
    return df

def parse_dates(df, meta):
    col = meta["date_col"]
    if "date_origin" in meta:
        df[col] = pd.to_datetime(df[col], unit="D",
                                 origin=meta["date_origin"], errors="coerce")
    else:
        df[col] = pd.to_datetime(df[col].astype(str),
                                 format=meta.get("date_format"), errors="coerce")
    return df

def handle_long_format(df, meta):
    """Data‑in‑Brief 转宽表: 每物种一列 abundance"""
    key_cols = [meta["date_col"], 'Taxa (standardized name)']
    val_col  = meta["target_col"]
    pivoted  = (df[key_cols + [val_col]]
                .pivot_table(index=meta["date_col"],
                             columns='Taxa (standardized name)',
                             values=val_col, aggfunc='mean')
                .reset_index())
    return pivoted

def impute(df, thresh):
    miss = df.isna().mean()
    keep = miss[miss < thresh].index
    df   = df[keep]
    num  = df.select_dtypes(include=[np.number]).columns
    if len(num) >= 2:
        df[num] = KNNImputer(n_neighbors=5).fit_transform(df[num])
    return df, miss

def quality_control(df, meta):
    """数据质量控制：异常值检测和清理"""
    if "quality_checks" not in meta:
        return df

    for check in meta["quality_checks"]:
        col = check["column"]
        if col in df.columns:
            min_val = check.get("min", -np.inf)
            max_val = check.get("max", np.inf)
            # 过滤异常值
            mask = (df[col] >= min_val) & (df[col] <= max_val)
            df = df[mask]
            print(f"  质控 {col}: 保留 {mask.sum()}/{len(mask)} 条记录")

    return df

def detect_outliers(df, method="iqr", threshold=3.0):
    """异常值检测"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outlier_mask = pd.Series(False, index=df.index)

    for col in numeric_cols:
        if method == "iqr":
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outliers = (df[col] < lower) | (df[col] > upper)
        elif method == "zscore":
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            outliers = z_scores > threshold
        elif method == "isolation_forest":
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            outliers = iso_forest.fit_predict(df[[col]].fillna(df[col].mean())) == -1
        else:
            outliers = pd.Series(False, index=df.index)

        outlier_mask |= outliers

    return ~outlier_mask  # 返回非异常值的mask

def add_rolling_features(df, targets, windows):
    """增强的滚动特征工程"""
    for target in targets:
        if target not in df.columns:
            continue
        for w in windows:
            # 基础统计特征
            df[f"{target}_mean_{w}"] = df[target].rolling(w).mean()
            df[f"{target}_std_{w}"] = df[target].rolling(w).std()
            df[f"{target}_min_{w}"] = df[target].rolling(w).min()
            df[f"{target}_max_{w}"] = df[target].rolling(w).max()
            df[f"{target}_median_{w}"] = df[target].rolling(w).median()

            # 趋势特征
            df[f"{target}_trend_{w}"] = df[target].rolling(w).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == w else np.nan
            )

            # 变化率特征
            df[f"{target}_pct_change_{w}"] = df[target].pct_change(w)

    return df

def make_sequences(df, date_col, target_col, window=30, horizon=1):
    """生成序列数据用于时间序列模型"""
    df_sorted = df.sort_values(date_col).reset_index(drop=True)

    sequences = []
    targets = []
    dates = []

    for i in range(window, len(df_sorted) - horizon + 1):
        # 特征序列
        seq_data = df_sorted.iloc[i-window:i].select_dtypes(include=[np.number]).values
        # 目标值
        target_val = df_sorted.iloc[i+horizon-1][target_col]
        # 日期
        target_date = df_sorted.iloc[i+horizon-1][date_col]

        sequences.append(seq_data)
        targets.append(target_val)
        dates.append(target_date)

    return np.array(sequences), np.array(targets), np.array(dates)

def feature_selection(df, target_col, max_features=50):
    """特征选择"""
    if target_col not in df.columns:
        return df

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    feature_cols = [col for col in numeric_cols if col != target_col]

    if len(feature_cols) <= max_features:
        return df

    # 处理无穷大值和NaN
    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # 替换无穷大值
    X = X.replace([np.inf, -np.inf], np.nan)
    y = y.replace([np.inf, -np.inf], np.nan)

    # 填充缺失值
    X = X.fillna(X.mean())
    y = y.fillna(y.mean())

    # 检查是否还有无效值
    if X.isnull().any().any() or y.isnull().any():
        print("  警告：特征选择跳过，存在无效值")
        return df

    try:
        selector = SelectKBest(score_func=f_regression, k=max_features)
        selector.fit(X, y)

        selected_features = [feature_cols[i] for i in selector.get_support(indices=True)]
        selected_features.append(target_col)  # 保留目标列

        # 保留日期列（如果存在）
        date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        for date_col in date_cols:
            if date_col not in selected_features:
                selected_features.insert(0, date_col)

        return df[selected_features]
    except Exception as e:
        print(f"  特征选择失败: {e}")
        return df

def add_roll(df, targets, windows):
    """保持向后兼容的滚动特征函数"""
    return add_rolling_features(df, targets, windows)

def process_one(name, meta, gcfg):
    out_dir = ROOT / "data_proc" / name
    out_dir.mkdir(parents=True, exist_ok=True)

    if meta.get("skip", False):
        return None

    print(f"处理数据集: {name}")

    # 1. 加载数据
    df = load_table(meta)
    original_rows = len(df)
    print(f"  原始数据: {original_rows} 行")

    # 2. 日期解析
    df = parse_dates(df, meta)

    # 3. 特殊处理：phyto_long -> 宽表
    if meta.get("long_format", False):
        df = handle_long_format(df, meta)

    # 4. 数据质量控制
    df = quality_control(df, meta)

    # 5. 排序和去重
    df = df.sort_values(meta["date_col"]).drop_duplicates()

    # 6. 异常值检测（如果启用）
    if gcfg.get("outlier_detection", False):
        outlier_mask = detect_outliers(df,
                                     method=gcfg.get("outlier_method", "iqr"),
                                     threshold=gcfg.get("outlier_threshold", 3.0))
        df = df[outlier_mask]
        print(f"  异常值过滤后: {len(df)} 行")

    # 7. 缺失值处理
    df, miss_ratio = impute(df, gcfg["impute_threshold"])
    print(f"  缺失值处理后: {len(df)} 行")

    # 8. 特征白名单过滤
    if "feature_whitelist" in meta and meta.get("target_col"):
        keep_cols = [meta["date_col"]] + meta["feature_whitelist"] + [meta["target_col"]]
        keep_cols = [col for col in keep_cols if col in df.columns]
        df = df[keep_cols]
        print(f"  特征白名单过滤后: {len(df.columns)} 列")

    # 9. 滚动特征工程（扩展到更多特征）
    if meta["freq"] in ("D", "W", "M"):
        target_cols = [meta["target_col"]] if meta.get("target_col") else []
        # 添加其他数值列的滚动特征
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        important_cols = [col for col in numeric_cols if any(keyword in col.lower()
                         for keyword in ['temp', 'sal', 'oxygen', 'nitrate', 'phosphate'])]
        target_cols.extend(important_cols[:3])  # 限制特征数量

        if target_cols:
            df = add_rolling_features(df, target_cols, gcfg["rolling_windows"])
            print(f"  滚动特征工程后: {len(df.columns)} 列")

    # 10. 特征选择（如果启用）
    if gcfg.get("feature_selection", False) and meta.get("target_col"):
        df = feature_selection(df, meta["target_col"], gcfg.get("max_features", 50))
        print(f"  特征选择后: {len(df.columns)} 列")

    # 11. 最终数据清理：处理无穷大值
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    # 11. 保存清理后的数据
    save_df(df, out_dir / "clean.csv")

    # 12. 生成序列数据（如果需要）
    if meta.get("target_col") and gcfg.get("sequence_window"):
        try:
            sequences, targets, dates = make_sequences(
                df, meta["date_col"], meta["target_col"],
                window=gcfg["sequence_window"],
                horizon=gcfg.get("sequence_horizon", 1)
            )
            np.savez(out_dir / "sequences.npz",
                    X=sequences, y=targets, dates=dates)
            print(f"  序列数据: {len(sequences)} 个序列")
        except Exception as e:
            print(f"  序列数据生成失败: {e}")

    # 安全获取日期范围
    try:
        date_min = df[meta["date_col"]].min()
        date_max = df[meta["date_col"]].max()
    except (KeyError, TypeError):
        date_min = "N/A"
        date_max = "N/A"

    return {
        "dataset": name,
        "original_rows": original_rows,
        "final_rows": len(df),
        "final_cols": len(df.columns),
        "date_min": date_min,
        "date_max": date_max,
        "data_reduction": f"{(1 - len(df)/original_rows)*100:.1f}%",
        **{f"{c}_miss": round(v, 3) for c, v in miss_ratio.items()}
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="all",
                        help="'all' or specific dataset key")
    args = parser.parse_args()

    cfg = read_cfg()
    targets = cfg["datasets"].keys() if args.dataset == "all" else [args.dataset]
    summary = []
    for d in tqdm(targets, desc="Pre‑processing"):
        s = process_one(d, cfg["datasets"][d], cfg["global"])
        if s: summary.append(s)

    if summary:
        save_df(pd.DataFrame(summary),
                ROOT / "data_proc" / "common_stats.csv")
        print("✓ preprocessing done ⇒ data_proc/")
