#!/usr/bin/env python3
"""
Simple Test Runner for Marine ML Benchmark

This script runs basic tests without requiring pytest.
"""

import sys
import os
import traceback
import pandas as pd
import numpy as np
from pathlib import Path

# Add code directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def print_test_header(test_name):
    """Print test header."""
    print(f"\n{'='*50}")
    print(f"🧪 Testing: {test_name}")
    print(f"{'='*50}")

def print_test_result(test_name, passed, error=None):
    """Print test result."""
    if passed:
        print(f"✅ {test_name}: PASSED")
    else:
        print(f"❌ {test_name}: FAILED")
        if error:
            print(f"   Error: {error}")

def test_basic_imports():
    """Test basic package imports."""
    print_test_header("Basic Imports")
    
    tests = [
        ("pandas", lambda: __import__('pandas')),
        ("numpy", lambda: __import__('numpy')),
        ("sklearn", lambda: __import__('sklearn')),
        ("matplotlib", lambda: __import__('matplotlib')),
        ("seaborn", lambda: __import__('seaborn')),
    ]
    
    passed_count = 0
    for name, test_func in tests:
        try:
            test_func()
            print_test_result(f"Import {name}", True)
            passed_count += 1
        except Exception as e:
            print_test_result(f"Import {name}", False, str(e))
    
    return passed_count, len(tests)

def test_data_loading():
    """Test data loading functionality."""
    print_test_header("Data Loading")
    
    # Check if data files exist
    data_dir = Path("data/processed")
    if not data_dir.exists():
        print_test_result("Data directory exists", False, "data/processed not found")
        return 0, 1
    
    datasets = ['biotoxin', 'cast', 'cleaned_data', 'era5_daily', 'hydrographic']
    passed_count = 0
    
    for dataset in datasets:
        try:
            data_file = data_dir / dataset / "clean.csv"
            if data_file.exists():
                df = pd.read_csv(data_file)
                if len(df) > 0 and len(df.columns) > 0:
                    print_test_result(f"Load {dataset}", True)
                    passed_count += 1
                else:
                    print_test_result(f"Load {dataset}", False, "Empty dataframe")
            else:
                print_test_result(f"Load {dataset}", False, "File not found")
        except Exception as e:
            print_test_result(f"Load {dataset}", False, str(e))
    
    return passed_count, len(datasets)

def test_model_functionality():
    """Test basic model functionality."""
    print_test_header("Model Functionality")
    
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import r2_score, mean_absolute_error
        
        # Generate synthetic data
        np.random.seed(42)
        X = np.random.randn(1000, 5)
        y = X[:, 0] + X[:, 1] + 0.1 * np.random.randn(1000)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        if r2 > 0.5:  # Should achieve reasonable performance on synthetic data
            print_test_result("Random Forest Training", True)
            print(f"   R² = {r2:.4f}, MAE = {mae:.4f}")
            return 1, 1
        else:
            print_test_result("Random Forest Training", False, f"Poor performance: R² = {r2:.4f}")
            return 0, 1
            
    except Exception as e:
        print_test_result("Random Forest Training", False, str(e))
        return 0, 1

def test_results_files():
    """Test if results files exist and are readable."""
    print_test_header("Results Files")
    
    files_to_check = [
        ("outputs/tables/final_table1_dataset_characteristics.csv", "Table 1"),
        ("outputs/tables/final_table2_model_performance.csv", "Table 2"),
        ("outputs/figures/figure1_dataset_overview_final.png", "Figure 1"),
        ("outputs/figures/figure2_performance_heatmap_final.png", "Figure 2"),
    ]
    
    passed_count = 0
    for file_path, description in files_to_check:
        try:
            if Path(file_path).exists():
                if file_path.endswith('.csv'):
                    df = pd.read_csv(file_path)
                    if len(df) > 0:
                        print_test_result(f"{description} exists and readable", True)
                        passed_count += 1
                    else:
                        print_test_result(f"{description} exists and readable", False, "Empty file")
                else:
                    # For image files, just check existence
                    print_test_result(f"{description} exists", True)
                    passed_count += 1
            else:
                print_test_result(f"{description} exists", False, "File not found")
        except Exception as e:
            print_test_result(f"{description} readable", False, str(e))
    
    return passed_count, len(files_to_check)

def test_configuration():
    """Test configuration loading."""
    print_test_header("Configuration")
    
    try:
        import yaml
        config_file = Path("configs/config.yaml")
        
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            if isinstance(config, dict) and len(config) > 0:
                print_test_result("Configuration loading", True)
                print(f"   Found {len(config)} configuration sections")
                return 1, 1
            else:
                print_test_result("Configuration loading", False, "Empty or invalid config")
                return 0, 1
        else:
            print_test_result("Configuration loading", False, "Config file not found")
            return 0, 1
            
    except Exception as e:
        print_test_result("Configuration loading", False, str(e))
        return 0, 1

def main():
    """Run all tests."""
    print("🚀 Marine ML Benchmark - Simple Test Suite")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not Path("README.md").exists():
        print("❌ Please run this script from the repository root directory")
        sys.exit(1)
    
    # Run all tests
    test_functions = [
        test_basic_imports,
        test_data_loading,
        test_model_functionality,
        test_results_files,
        test_configuration
    ]
    
    total_passed = 0
    total_tests = 0
    
    for test_func in test_functions:
        try:
            passed, count = test_func()
            total_passed += passed
            total_tests += count
        except Exception as e:
            print(f"❌ Test function {test_func.__name__} failed: {e}")
            traceback.print_exc()
            total_tests += 1
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"🎯 Test Summary")
    print(f"{'='*60}")
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {total_passed}")
    print(f"Failed: {total_tests - total_passed}")
    print(f"Success Rate: {total_passed/total_tests*100:.1f}%")
    
    if total_passed == total_tests:
        print("\n✅ All tests passed! The package is working correctly.")
        return 0
    elif total_passed >= total_tests * 0.8:
        print("\n⚠️  Most tests passed. Some optional features may be missing.")
        return 0
    else:
        print("\n❌ Many tests failed. Please check the installation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
