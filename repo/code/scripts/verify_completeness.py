#!/usr/bin/env python3
"""
Marine ML Benchmark - Completeness Verification Script

This script verifies that all expected files are present and complete
in the Marine ML Benchmark reproducibility package.
"""

import os
import pandas as pd
import json
from pathlib import Path
import sys

def check_file_exists(filepath, description=""):
    """Check if a file exists and report status."""
    if os.path.exists(filepath):
        size = os.path.getsize(filepath)
        print(f"  ✅ {filepath} ({size:,} bytes) {description}")
        return True
    else:
        print(f"  ❌ {filepath} - MISSING {description}")
        return False

def verify_data_files():
    """Verify all data files are present."""
    print("📊 Verifying Data Files...")
    
    datasets = ['biotoxin', 'cast', 'cleaned_data', 'era5_daily', 'hydrographic', 
                'processed_seq', 'rolling_mean', 'phyto_long', 'phyto_wide']
    
    all_present = True
    
    for dataset in datasets:
        dataset_dir = f"data/processed/{dataset}"
        clean_file = f"{dataset_dir}/clean.csv"
        
        if check_file_exists(clean_file, f"- {dataset} processed data"):
            # Check if it's a valid CSV
            try:
                df = pd.read_csv(clean_file)
                print(f"    📈 {len(df)} samples, {len(df.columns)} columns")
            except Exception as e:
                print(f"    ⚠️  Error reading CSV: {e}")
                all_present = False
        else:
            all_present = False
        
        # Check for sequence files (deep learning datasets)
        seq_file = f"{dataset_dir}/sequences.npz"
        if os.path.exists(seq_file):
            check_file_exists(seq_file, "- sequence data for deep learning")
    
    # Check data documentation
    check_file_exists("data/README_DATA.md", "- data documentation")
    
    return all_present

def verify_model_files():
    """Verify all trained model files are present."""
    print("\n🤖 Verifying Model Files...")
    
    datasets = ['biotoxin', 'cast', 'cleaned_data', 'era5_daily', 'hydrographic', 
                'processed_seq', 'rolling_mean', 'phyto_long', 'phyto_wide']
    
    # Expected models per dataset
    traditional_models = ['rf', 'xgb', 'svr']
    deep_learning_models = ['lstm', 'transformer']
    
    # Datasets that support deep learning
    dl_datasets = ['biotoxin', 'cleaned_data', 'hydrographic', 'processed_seq', 'rolling_mean']
    
    all_present = True
    total_models = 0
    
    for dataset in datasets:
        dataset_dir = f"models/{dataset}"
        print(f"  📁 {dataset}:")
        
        # Check traditional ML models
        for model in traditional_models:
            model_file = f"{dataset_dir}/{model}.pkl"
            params_file = f"{dataset_dir}/{model}_params.json"
            
            if check_file_exists(model_file, f"- {model} model"):
                total_models += 1
            else:
                all_present = False
                
            if check_file_exists(params_file, f"- {model} parameters"):
                # Verify JSON is valid
                try:
                    with open(params_file, 'r') as f:
                        params = json.load(f)
                    print(f"      🔧 {len(params)} hyperparameters")
                except Exception as e:
                    print(f"      ⚠️  Error reading parameters: {e}")
            else:
                all_present = False
        
        # Check deep learning models (if applicable)
        if dataset in dl_datasets:
            for model in deep_learning_models:
                model_file = f"{dataset_dir}/{model}.pth"
                params_file = f"{dataset_dir}/{model}_params.json"
                
                if check_file_exists(model_file, f"- {model} model"):
                    total_models += 1
                else:
                    all_present = False
                    
                if check_file_exists(params_file, f"- {model} parameters"):
                    try:
                        with open(params_file, 'r') as f:
                            params = json.load(f)
                        print(f"      🔧 {len(params)} hyperparameters")
                    except Exception as e:
                        print(f"      ⚠️  Error reading parameters: {e}")
                else:
                    all_present = False
    
    print(f"\n  📊 Total models found: {total_models}")
    print(f"  📊 Expected models: 39 (7×5 + 2×3 = 35 + 4 excluded)")
    
    # Check model documentation
    check_file_exists("models/README_MODELS.md", "- model documentation")
    
    return all_present

def verify_results_files():
    """Verify all result files are present."""
    print("\n📈 Verifying Results Files...")
    
    # Main paper tables
    main_tables = [
        "final_table1_dataset_characteristics.csv",
        "final_table2_model_performance.csv", 
        "final_table3_best_performance.csv",
        "final_table4_validation_summary.csv"
    ]
    
    # Supplementary tables
    supp_tables = [
        "supplementary_table_s2_full_results.csv",
        "complete_sanity_check_results.csv",
        "small_sample_analysis.csv"
    ]
    
    # Paper figures
    figures = [
        "figure1_dataset_overview_final",
        "figure2_performance_heatmap_final",
        "figure3_performance_boxplots_final", 
        "figure4_model_robustness_final",
        "figure5_difficulty_vs_size_final",
        "figure6_feature_importance_final",
        "figure7_technical_roadmap_final"
    ]
    
    all_present = True
    
    print("  📊 Main Paper Tables:")
    for table in main_tables:
        table_path = f"outputs/tables/{table}"
        if check_file_exists(table_path):
            try:
                df = pd.read_csv(table_path)
                print(f"      📋 {len(df)} rows, {len(df.columns)} columns")
            except Exception as e:
                print(f"      ⚠️  Error reading table: {e}")
                all_present = False
        else:
            all_present = False
    
    print("  📊 Supplementary Tables:")
    for table in supp_tables:
        table_path = f"outputs/tables/{table}"
        if check_file_exists(table_path):
            try:
                df = pd.read_csv(table_path)
                print(f"      📋 {len(df)} rows, {len(df.columns)} columns")
            except Exception as e:
                print(f"      ⚠️  Error reading table: {e}")
                all_present = False
        else:
            all_present = False
    
    print("  📊 Paper Figures:")
    for figure in figures:
        png_path = f"outputs/figures/{figure}.png"
        pdf_path = f"outputs/figures/{figure}.pdf"
        
        png_ok = check_file_exists(png_path, "- PNG version")
        pdf_ok = check_file_exists(pdf_path, "- PDF version")
        
        if not (png_ok and pdf_ok):
            all_present = False
    
    # Check table index
    check_file_exists("outputs/tables/SUPPLEMENTARY_TABLES_INDEX.md", "- tables index")
    
    return all_present

def verify_code_files():
    """Verify all code files are present."""
    print("\n🔧 Verifying Code Files...")
    
    # Core source files
    src_files = [
        "__init__.py", "config.yaml", "preprocess.py", 
        "train_enhanced.py", "evaluate_enhanced.py", 
        "visualize.py", "utils_io.py"
    ]
    
    # Scripts
    script_files = [
        "run_full_pipeline.sh", "run_quick_test.sh", "generate_figures.py"
    ]
    
    all_present = True
    
    print("  📁 Core Source Files:")
    for file in src_files:
        file_path = f"code/src/{file}"
        if not check_file_exists(file_path):
            all_present = False
    
    print("  📁 Execution Scripts:")
    for file in script_files:
        file_path = f"code/scripts/{file}"
        if not check_file_exists(file_path):
            all_present = False
    
    # Check notebook
    check_file_exists("code/notebooks/demo_reproduction.ipynb", "- demo notebook")
    
    return all_present

def verify_documentation():
    """Verify all documentation files are present."""
    print("\n📚 Verifying Documentation...")
    
    doc_files = [
        ("README.md", "main documentation"),
        ("LICENSE", "license file"),
        ("CITATION.cff", "citation format"),
        ("CHANGELOG.md", "version history"),
        ("requirements.txt", "Python dependencies"),
        ("environment.yml", "Conda environment"),
        ("CONTENTS_MANIFEST.md", "contents manifest"),
        ("docs/METHODOLOGY.md", "detailed methodology"),
        ("docs/paper_figures_tables_detailed_explanation.md", "detailed analysis"),
        ("docs/SUPPLEMENTARY_TABLES_ANALYSIS.md", "supplementary analysis")
    ]
    
    all_present = True
    
    for file_path, description in doc_files:
        if not check_file_exists(file_path, f"- {description}"):
            all_present = False
    
    return all_present

def main():
    """Main verification function."""
    print("🔍 Marine ML Benchmark - Completeness Verification")
    print("=" * 60)
    
    # Change to repository root if needed
    if not os.path.exists("README.md"):
        print("❌ Please run this script from the repository root directory")
        sys.exit(1)
    
    # Run all verification checks
    checks = [
        ("Data Files", verify_data_files),
        ("Model Files", verify_model_files), 
        ("Results Files", verify_results_files),
        ("Code Files", verify_code_files),
        ("Documentation", verify_documentation)
    ]
    
    all_passed = True
    
    for check_name, check_func in checks:
        try:
            result = check_func()
            if not result:
                all_passed = False
        except Exception as e:
            print(f"❌ Error during {check_name} verification: {e}")
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 VERIFICATION PASSED - All files are present and complete!")
        print("✅ The Marine ML Benchmark package is ready for use.")
        print("\nNext steps:")
        print("  1. Run quick test: bash code/scripts/run_quick_test.sh")
        print("  2. Run full pipeline: bash code/scripts/run_full_pipeline.sh")
        print("  3. Explore notebooks: jupyter notebook code/notebooks/")
    else:
        print("❌ VERIFICATION FAILED - Some files are missing or incomplete.")
        print("Please check the error messages above and ensure all files are present.")
        sys.exit(1)

if __name__ == "__main__":
    main()
