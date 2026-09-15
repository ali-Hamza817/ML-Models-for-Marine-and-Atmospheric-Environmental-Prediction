from pathlib import Path
import yaml, pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def read_cfg():
    with open(ROOT / "src" / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)

def save_df(df: pd.DataFrame, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
