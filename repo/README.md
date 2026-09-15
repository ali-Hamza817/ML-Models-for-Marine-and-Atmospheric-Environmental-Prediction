# Marine ML Benchmark: Cross-Dataset Robustness Evaluation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Data License: CC BY 4.0](https://img.shields.io/badge/Data%20License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## 🎯 Overview

This repository contains the complete implementation for **"Cross-Dataset Robustness of Machine Learning Models for Chlorophyll-a Prediction: A Comprehensive Benchmark Study"**. 

We systematically evaluate 5 machine learning models across 9 heterogeneous oceanographic datasets, focusing on model robustness rather than data fusion approaches.

### Key Contributions
- First systematic benchmark of ML models across heterogeneous oceanographic datasets
- Standardized evaluation protocol ensuring fair cross-dataset comparison  
- Comprehensive analysis of model robustness and feature importance patterns
- Practical guidelines for model selection based on data characteristics
- Open-source benchmarking framework for reproducible research

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- 4GB+ RAM
- 2GB+ disk space

### Installation
```bash
# Clone repository
git clone https://github.com/[username]/marine-ml-benchmark.git
cd marine-ml-benchmark

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Quick Test (5 minutes)
```bash
# Option 1: Bash script (Linux/Mac)
bash code/scripts/run_quick_test.sh

# Option 2: PowerShell script (Windows)
.\code\scripts\run_quick_test.ps1

# Option 3: Python script (Cross-platform)
python code/scripts/run_reproduction.py
```

### Full Reproduction (30-60 minutes)
```bash
# Option 1: Bash script (Linux/Mac)
bash code/scripts/run_full_pipeline.sh

# Option 2: PowerShell script (Windows)
.\code\scripts\run_full_pipeline.ps1

# Option 3: Python script (Cross-platform)
python code/scripts/run_reproduction.py
```

## ✅ PLOS ONE Revision Pipeline (New)

This pipeline generates all revision-required analyses (permutation test K=10,000, hydrographic baseline check, ERA5 downsampling ablation) and regenerates all tables and high-resolution figures (PDF/PNG/TIFF).

```bash
# Full revision pipeline (can be long for K=10,000)
python code/scripts/run_plos_revision.py --permutations 10000 --model xgb

# Faster smoke test (subsamples for sanity check)
python code/scripts/run_plos_revision.py --fast --permutations 1000 --model ridge
```

Key outputs:
- `outputs/tables/complete_sanity_check_results.csv`
- `outputs/tables/hydrographic_baseline_diagnostics.csv`
- `outputs/tables/era5_downsample_ablation.csv`
- `outputs/tables/dataset_split_summary.csv`
- `outputs/tables/dataset_geography_summary.csv`
- `outputs/tables/transfer_pilot.csv`
- `outputs/figures/figure*_final.pdf/png/tiff`
- `outputs/plos_submission/Fig*.tif` and `outputs/plos_submission/S*_Table.docx` (PLOS naming)

## 📊 Results Overview

### Key Findings
- **XGBoost**: Best overall robustness (mean R² = 0.823 ± 0.320)
- **Random Forest**: Most consistent performance across datasets
- **Deep Learning**: Limited applicability (5/9 datasets) but stable where applicable
- **Data Quality > Quantity**: No correlation between sample size and predictability

### Dataset Characteristics
| Dataset | Samples | Features | Type | Target Variable | Validation |
|---------|---------|----------|------|----------------|------------|
| era5_daily | 102,982 | 8 | Time Series | Wind speed (10m) | ✅ |
| cast | 21,865 | 25 | Cross-sectional | Bottom depth | ✅ |
| cleaned_data | 7,819 | 69 | Cross-sectional | Chlorophyll-a | ✅ |
| rolling_mean | 8,855 | 69 | Time Series | Chlorophyll-a (smoothed) | ✅ |
| processed_seq | 8,039 | 30 | Time Series | Chlorophyll-a (processed) | ✅ |
| biotoxin | 5,076 | 2 | Cross-sectional | Biotoxin concentration | ✅ |
| hydrographic | 4,653 | 11 | Cross-sectional | Chlorophyll-a | ✅ |

## 📁 Repository Structure

```
marine-ml-benchmark/
├── code/
│   ├── src/                    # Core source code (8 modules)
│   ├── scripts/                # Execution scripts (3 scripts)
│   └── notebooks/              # Demo notebooks (1 notebook)
├── data/
│   ├── processed/              # Processed datasets (7 validated + 2 excluded)
│   ├── sample/                 # Sample data for testing
│   └── README_DATA.md          # Comprehensive data documentation
├── outputs/
│   ├── tables/                 # Paper tables + supplementary tables (15+ files)
│   └── figures/                # Paper figures + analysis plots (15+ files)
├── models/                     # Trained models (9 datasets × 5 models)
│   ├── biotoxin/               # 5 models (RF, XGB, SVR, LSTM, Transformer)
│   ├── cast/                   # 3 models (RF, XGB, SVR)
│   ├── cleaned_data/           # 5 models (RF, XGB, SVR, LSTM, Transformer)
│   ├── era5_daily/             # 3 models (RF, XGB, SVR)
│   ├── hydrographic/           # 5 models (RF, XGB, SVR, LSTM, Transformer)
│   ├── processed_seq/          # 5 models (RF, XGB, SVR, LSTM, Transformer)
│   ├── rolling_mean/           # 5 models (RF, XGB, SVR, LSTM, Transformer)
│   ├── phyto_long/             # 3 models (excluded dataset)
│   ├── phyto_wide/             # 3 models (excluded dataset)
│   └── README_MODELS.md        # Model documentation
├── logs/                       # Training logs and hyperparameters (800+ trials)
├── configs/                    # Configuration files
├── tests/                      # Unit tests (3 test suites)
└── docs/                       # Comprehensive documentation
    ├── METHODOLOGY.md          # Detailed methodology
    ├── paper_figures_tables_detailed_explanation.md  # 1196-line analysis
    └── SUPPLEMENTARY_TABLES_ANALYSIS.md              # Supplementary analysis
```

## 🔬 Methodology

### Models Evaluated
- **Traditional ML**: Random Forest, XGBoost, SVR
- **Deep Learning**: LSTM, Transformer  
- **Baselines**: LASSO, Ridge, Mean

### Evaluation Protocol
- **Data Split**: 70/15/15 (train/validation/test)
- **Cross-Validation**: 5-fold TimeSeriesSplit
- **Metrics**: R², MAE, RMSE with 95% confidence intervals
- **Hyperparameter Optimization**: Optuna with 30 trials per model

### Quality Control
- Outlier detection (IQR method)
- Missing value imputation (KNN)
- Feature engineering (rolling statistics, temporal features)
- Statistical significance testing

## 📈 Reproducing Results

### Paper Tables
All tables are generated as CSV files in `outputs/tables/`:

**Main Paper Tables:**
- `final_table1_dataset_characteristics.csv` - Dataset overview (9 datasets)
- `final_table2_model_performance.csv` - Cross-dataset performance (54 combinations)
- `final_table3_best_performance.csv` - Best model per dataset (7 validated)
- `final_table4_validation_summary.csv` - Validation statistics

**Supplementary Tables:**
- `supplementary_table_s2_full_results.csv` - Complete results with confidence intervals
- `complete_sanity_check_results.csv` - Data leakage detection results
- `small_sample_analysis.csv` - Small sample exclusion analysis
- `SUPPLEMENTARY_TABLES_INDEX.md` - Complete index of all tables
- `hydrographic_baseline_diagnostics.csv` - Baseline R² diagnostics for hydrographic
- `era5_downsample_ablation.csv` - ERA5 sample-size ablation results
- `dataset_split_summary.csv` - Train/val/test date ranges for time-series datasets
- `dataset_geography_summary.csv` - Latitude/longitude bounds per dataset

### Paper Figures
All figures are generated in `outputs/figures/` (PNG + PDF formats):

**Main Paper Figures:**
- `figure1_dataset_overview_final.*` - Dataset characteristics and temporal coverage
- `figure2_performance_heatmap_final.*` - Cross-dataset performance heatmap
- `figure3_performance_boxplots_final.*` - Performance distributions by model
- `figure4_model_robustness_final.*` - Model robustness analysis
- `figure5_difficulty_vs_size_final.*` - Dataset difficulty vs. sample size
- `figure6_feature_importance_final.*` - Feature importance patterns
- `figure7_technical_roadmap_final.*` - Complete methodology workflow

**Supplementary Figures:**
- `sample_size_analysis.png` - Sample size requirements analysis

### Using Trained Models
```python
import joblib
import torch
import pandas as pd
import json

# Load a traditional ML model
rf_model = joblib.load('models/cleaned_data/rf.pkl')
with open('models/cleaned_data/rf_params.json', 'r') as f:
    rf_params = json.load(f)

# Load a deep learning model
lstm_model = torch.load('models/cleaned_data/lstm.pth')
lstm_model.eval()

# Make predictions
predictions = rf_model.predict(X_test)
```

### Custom Analysis
```python
# Load results for custom analysis
import pandas as pd

# Load performance data
performance = pd.read_csv('outputs/tables/final_table2_model_performance.csv')

# Load supplementary results with confidence intervals
supplementary = pd.read_csv('outputs/tables/supplementary_table_s2_full_results.csv')

# Load dataset characteristics
datasets = pd.read_csv('outputs/tables/final_table1_dataset_characteristics.csv')

# Load hyperparameter optimization log
hyperparams = pd.read_csv('logs/hyperparameter_search_log.csv')

# Your custom analysis here...
```

## 🛠️ Advanced Usage

### Running Individual Components
```bash
# Data preprocessing only
python -m code.src.preprocess --dataset all

# Train specific model on specific dataset
python -m code.src.train_enhanced --dataset cleaned_data --model xgb

# Generate all figures (7 publication-ready figures)
python code/scripts/generate_figures.py

# Generate all tables (4 main paper tables)
python code/scripts/generate_final_tables.py

# Run small sample analysis
python code/scripts/small_sample_analysis.py

# Run data validation checks
python code/scripts/complete_sanity_check.py

# Generate hyperparameter logs
python code/scripts/hyperparameter_logging.py
```

### Configuration
Modify `configs/config.yaml` to:
- Adjust hyperparameter search spaces
- Change cross-validation settings
- Modify data preprocessing parameters
- Add new datasets or models

### Adding New Datasets
1. Place raw data in `data/raw/`
2. Add dataset configuration to `configs/config.yaml`
3. Run preprocessing: `python -m code.src.preprocess --dataset your_dataset`

## 🧪 Testing

Run unit tests to verify installation:
```bash
# Run all tests
python -m pytest tests/

# Run specific test
python -m pytest tests/test_preprocess.py
```

## 📖 Citation

If you use this code or data in your research, please cite:

```bibtex
@article{marine_ml_benchmark_2025,
  title={Cross-Dataset Robustness of Machine Learning Models for Chlorophyll-a Prediction: A Comprehensive Benchmark Study},
  author={[Your Name]},
  journal={[Journal Name]},
  year={2025},
  url={https://github.com/[username]/marine-ml-benchmark}
}
```

## 📄 License

- **Code**: MIT License - see [LICENSE](LICENSE) file
- **Data**: CC BY 4.0 - see [data/README_DATA.md](data/README_DATA.md)

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

## 📞 Contact

- **Author**: [Hanming Zhang] ([zhm19334637080@gmail.com])
- **Issues**: [GitHub Issues](https://github.com/[username]/marine-ml-benchmark/issues)
- **Discussions**: [GitHub Discussions](https://github.com/[username]/marine-ml-benchmark/discussions)

## 🙏 Acknowledgments

- ERA5 data: Copernicus Climate Change Service (C3S)
- Oceanographic data: [Data providers]
- Computing resources: [Institution/Grant]

---

**Keywords**: chlorophyll-a prediction, cross-dataset evaluation, model robustness, marine ecosystem monitoring, machine learning benchmark
