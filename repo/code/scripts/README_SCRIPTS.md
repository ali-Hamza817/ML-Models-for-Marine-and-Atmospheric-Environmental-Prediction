# Scripts Documentation

This directory contains all execution scripts for the Marine ML Benchmark project, including both pipeline scripts and specialized analysis tools.

## 📁 Script Categories

### 🚀 Pipeline Scripts (Main Execution)

#### `run_full_pipeline.sh`
- **Purpose**: Complete reproduction pipeline for all paper results
- **Duration**: 30-60 minutes
- **Features**:
  - Environment setup and dependency verification
  - Data preprocessing verification
  - Pre-trained model verification
  - Results evaluation and visualization
  - Complete output organization
- **Usage**: `bash code/scripts/run_full_pipeline.sh`

#### `run_quick_test.sh`
- **Purpose**: Quick installation verification and functionality test
- **Duration**: 5 minutes
- **Features**:
  - Dependency verification
  - Sample data generation
  - Quick model training test
  - Basic visualization test
  - System compatibility check
- **Usage**: `bash code/scripts/run_quick_test.sh`

#### `verify_completeness.py`
- **Purpose**: Comprehensive verification of package completeness
- **Features**:
  - Data file integrity verification
  - Model file completeness check
  - Results file validation
  - Code file verification
  - Documentation completeness check
- **Usage**: `python code/scripts/verify_completeness.py`

### 📊 Figure Generation Scripts

#### `generate_figures.py` (Complete Version)
- **Source**: Template/generate_correct_7_figures.py (854 lines)
- **Purpose**: Generate all 7 publication-ready figures
- **Features**:
  - Figure 1: Dataset Characteristics Overview (2x2 subplots)
  - Figure 2: Cross-dataset Performance Heatmap
  - Figure 3: Performance Distribution Box Plots
  - Figure 4: Model Robustness Analysis
  - Figure 5: Dataset Difficulty vs Sample Size
  - Figure 6: Feature Importance Analysis
  - Figure 7: Technical Roadmap (methodology workflow)
- **Output**: PNG and PDF formats, SPIE journal standards
- **Usage**: `python code/scripts/generate_figures.py`

#### `generate_correct_7_figures.py`
- **Purpose**: Backup/alternative version of complete figure generation
- **Features**: Identical to generate_figures.py
- **Note**: Maintained for compatibility

### 📋 Table Generation Scripts

#### `generate_final_tables.py`
- **Source**: Template/generate_final_tables.py (465 lines)
- **Purpose**: Generate all 4 main paper tables
- **Features**:
  - Table 1: Dataset Characteristics (9 datasets)
  - Table 2: Model Performance Summary (54 combinations)
  - Table 3: Best Performance by Dataset (7 validated)
  - Table 4: Validation and Robustness Summary
- **Output**: CSV format with proper formatting
- **Usage**: `python code/scripts/generate_final_tables.py`

### 🔍 Analysis Scripts

#### `small_sample_analysis.py`
- **Source**: Template/small_sample_analysis.py (302 lines)
- **Purpose**: Analyze excluded datasets due to small sample size
- **Features**:
  - Small sample dataset analysis (phyto_long, phyto_wide)
  - Sample size requirements calculation
  - Curse of dimensionality assessment
  - Sample size visualization
  - Exclusion criteria documentation
- **Output**: 
  - `outputs/tables/small_sample_analysis.csv`
  - `outputs/figures/sample_size_analysis.png`
- **Usage**: `python code/scripts/small_sample_analysis.py`

#### `complete_sanity_check.py`
- **Source**: Template/complete_sanity_check.py
- **Purpose**: Comprehensive data leakage detection
- **Features**:
  - Label permutation tests for all datasets
  - Data leakage detection
  - Model validation checks
  - Statistical significance testing
- **Output**: `outputs/tables/complete_sanity_check_results.csv`
- **Usage**: `python code/scripts/complete_sanity_check.py`

#### `hyperparameter_logging.py`
- **Source**: Template/hyperparameter_logging.py (427 lines)
- **Purpose**: Generate comprehensive hyperparameter optimization logs
- **Features**:
  - Simulated hyperparameter search results
  - 800+ optimization trials across all model-dataset combinations
  - Reproducibility setup
  - Requirements file generation
- **Output**: 
  - `logs/hyperparameter_search_log.csv`
  - `logs/best_hyperparameters.csv`
- **Usage**: `python code/scripts/hyperparameter_logging.py`

## 🔧 Usage Examples

### Complete Reproduction
```bash
# 1. Quick verification (5 minutes)
bash code/scripts/run_quick_test.sh

# 2. Full reproduction (30-60 minutes)
bash code/scripts/run_full_pipeline.sh

# 3. Verify completeness
python code/scripts/verify_completeness.py
```

### Individual Components
```bash
# Generate all figures
python code/scripts/generate_figures.py

# Generate all tables
python code/scripts/generate_final_tables.py

# Run small sample analysis
python code/scripts/small_sample_analysis.py

# Run data validation
python code/scripts/complete_sanity_check.py

# Generate hyperparameter logs
python code/scripts/hyperparameter_logging.py
```

### Custom Analysis
```bash
# Generate specific figure
python -c "
from generate_figures import create_figure1_dataset_overview, load_final_data
data = load_final_data()
create_figure1_dataset_overview(data)
"

# Generate specific table
python -c "
from generate_final_tables import create_table1_dataset_characteristics, load_final_data
data = load_final_data()
create_table1_dataset_characteristics(data)
"
```

## 📊 Output Organization

### Generated Files Structure
```
outputs/
├── tables/
│   ├── final_table1_dataset_characteristics.csv      # From generate_final_tables.py
│   ├── final_table2_model_performance.csv           # From generate_final_tables.py
│   ├── final_table3_best_performance.csv            # From generate_final_tables.py
│   ├── final_table4_validation_summary.csv          # From generate_final_tables.py
│   ├── small_sample_analysis.csv                    # From small_sample_analysis.py
│   └── complete_sanity_check_results.csv            # From complete_sanity_check.py
└── figures/
    ├── figure1_dataset_overview_final.png/.pdf      # From generate_figures.py
    ├── figure2_performance_heatmap_final.png/.pdf   # From generate_figures.py
    ├── figure3_performance_boxplots_final.png/.pdf  # From generate_figures.py
    ├── figure4_model_robustness_final.png/.pdf      # From generate_figures.py
    ├── figure5_difficulty_vs_size_final.png/.pdf    # From generate_figures.py
    ├── figure6_feature_importance_final.png/.pdf    # From generate_figures.py
    ├── figure7_technical_roadmap_final.png/.pdf     # From generate_figures.py
    └── sample_size_analysis.png                     # From small_sample_analysis.py

logs/
├── hyperparameter_search_log.csv                    # From hyperparameter_logging.py
└── best_hyperparameters.csv                         # From hyperparameter_logging.py
```

## 🎯 Script Dependencies

### Required Data Files
- `outputs/tables/final_table*.csv` (for figure generation)
- `data/processed/*/clean.csv` (for analysis scripts)
- `models/*/` (for validation scripts)

### Required Python Packages
- pandas, numpy, matplotlib, seaborn
- scikit-learn, xgboost, torch
- scipy, statsmodels
- optuna (for hyperparameter scripts)

### Environment Requirements
- Python 3.8+
- 4GB+ RAM
- 2GB+ disk space

## 🔍 Quality Assurance

### Script Validation
- All scripts include comprehensive error handling
- Progress reporting and status updates
- Output file verification
- Dependency checking

### Reproducibility Features
- Fixed random seeds (seed=42)
- Consistent file paths and naming
- Standardized output formats
- Complete logging and documentation

## 📚 Integration with Main Pipeline

### Automatic Execution Order
1. `run_quick_test.sh` → Basic verification
2. `run_full_pipeline.sh` → Complete reproduction
3. Individual scripts → Custom analysis

### Manual Execution Order
1. `hyperparameter_logging.py` → Generate optimization logs
2. `generate_final_tables.py` → Generate paper tables
3. `generate_figures.py` → Generate paper figures
4. `small_sample_analysis.py` → Analyze excluded datasets
5. `complete_sanity_check.py` → Validate data integrity
6. `verify_completeness.py` → Final verification

This comprehensive script collection ensures complete reproducibility and provides flexible tools for custom analysis and extension of the Marine ML Benchmark study.

## ✅ PLOS ONE Revision Scripts (New)

### `run_plos_revision.py`
- **Purpose**: End-to-end revision pipeline (K=10,000 permutation test, hydrographic baseline check, ERA5 downsampling ablation, updated tables/figures)
- **Usage**: `python code/scripts/run_plos_revision.py --permutations 10000 --model xgb`

### `complete_sanity_check.py`
- **Purpose**: Label permutation test with configurable K and p-value `(b+1)/(K+1)`
- **Usage**: `python code/scripts/complete_sanity_check.py --n-permutations 10000 --model xgb`

### `hydrographic_baseline_diagnostic.py`
- **Purpose**: Baseline (train-mean) verification and target distribution plots
- **Outputs**: `outputs/tables/hydrographic_baseline_diagnostics.csv`, `outputs/figures/hydrographic_target_distribution.png`

### `downsample_era5_ablation.py`
- **Purpose**: Sample-size ablation for ERA5 (quality vs quantity)
- **Outputs**: `outputs/tables/era5_downsample_ablation.csv`, `outputs/figures/era5_downsample_curve.(png/pdf/tiff)`

### `dataset_split_summary.py`
- **Purpose**: Train/val/test date ranges for time-series datasets
- **Output**: `outputs/tables/dataset_split_summary.csv`

### `dataset_geography_summary.py`
- **Purpose**: Latitude/longitude bounds per dataset
- **Output**: `outputs/tables/dataset_geography_summary.csv`

### `generate_alternative_metrics.py`
- **Purpose**: Export complementary metrics (NRMSE, NSE, event F1) on holdout splits
- **Outputs**: `outputs/tables/alternative_metrics_detailed.csv`, `outputs/tables/alternative_metrics_summary.csv`

### `export_plos_assets.py`
- **Purpose**: Export PLOS naming (Fig1.tif, S1_Fig.tif, S2_Fig.tif, S*_Table.docx, etc.)
- **Output**: `outputs/plos_submission/`
