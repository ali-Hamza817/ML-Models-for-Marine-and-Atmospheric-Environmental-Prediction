#!/usr/bin/env python3
"""
Run the PLOS ONE revision pipeline:
- permutation test (K=10000)
- hydrographic baseline diagnostic
- ERA5 downsampling ablation
- split/geography summaries
- regenerate tables and figures
"""

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def run(cmd):
    print(f"\n>> {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


def main():
    parser = argparse.ArgumentParser(description="PLOS revision pipeline")
    parser.add_argument("--fast", action="store_true", help="Use faster settings for sanity check")
    parser.add_argument("--permutations", type=int, default=10000, help="Permutation count (K)")
    parser.add_argument("--model", choices=["xgb", "ridge"], default="xgb", help="Sanity check model")
    args = parser.parse_args()

    python = sys.executable

    sanity_cmd = [
        python, str(REPO_ROOT / "code" / "scripts" / "complete_sanity_check.py"),
        "--n-permutations", str(args.permutations),
        "--model", args.model,
        "--output", str(REPO_ROOT / "outputs" / "tables" / "complete_sanity_check_results.csv"),
    ]
    if args.fast:
        sanity_cmd += ["--max-samples", "10000"]

    run(sanity_cmd)
    run([python, str(REPO_ROOT / "code" / "scripts" / "hydrographic_baseline_diagnostic.py")])
    run([
        python, str(REPO_ROOT / "code" / "scripts" / "downsample_era5_ablation.py"),
        "--sizes", "100000", "50000", "10000", "5000", "1000",
        "--model", "rf",
        "--repeats", "3",
    ])
    run([python, str(REPO_ROOT / "code" / "scripts" / "dataset_split_summary.py")])
    run([python, str(REPO_ROOT / "code" / "scripts" / "dataset_geography_summary.py")])
    run([python, str(REPO_ROOT / "code" / "scripts" / "cross_dataset_transfer_pilot.py")])
    run([python, str(REPO_ROOT / "code" / "scripts" / "generate_alternative_metrics_legacy.py")])
    run([python, str(REPO_ROOT / "code" / "scripts" / "generate_final_tables.py")])
    run([python, str(REPO_ROOT / "code" / "scripts" / "generate_figures.py")])
    run([python, str(REPO_ROOT / "code" / "scripts" / "export_plos_assets.py")])

    print("\n✅ PLOS revision pipeline completed.")


if __name__ == "__main__":
    main()
