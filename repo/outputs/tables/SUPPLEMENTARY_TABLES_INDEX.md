# Supplementary Tables Index

This document provides an index and description of all supplementary tables included in the Marine ML Benchmark study.

## Main Paper Tables

### Table 1: Dataset Characteristics
- **File**: `final_table1_dataset_characteristics.csv`
- **Description**: Comprehensive overview of all 9 datasets including sample sizes, features, temporal coverage, and validation status
- **Rows**: 9 datasets
- **Key Columns**: Dataset, Samples, Features, Type, Target Variable, Time Range, Validation Status

### Table 2: Model Performance Summary
- **File**: `final_table2_model_performance.csv`
- **Description**: Cross-dataset performance results for all model-dataset combinations
- **Rows**: 54 model-dataset combinations
- **Key Columns**: Dataset, Model, R², R² (95% CI), p-value, MAE, Type

### Table 3: Best Performance per Dataset
- **File**: `final_table3_best_performance.csv`
- **Description**: Best performing model for each validated dataset
- **Rows**: 7 validated datasets
- **Key Columns**: Dataset, Best Model, R², MAE, Model Type

### Table 4: Validation Summary Statistics
- **File**: `final_table4_validation_summary.csv`
- **Description**: Statistical summary of model performance across datasets
- **Rows**: Summary statistics by model type
- **Key Columns**: Model, Mean R², Std R², Success Rate, Applicability

## Supplementary Tables

### Supplementary Table S1: Complete Performance Results with Confidence Intervals
- **File**: `supplementary_table_s2_full_results.csv`
- **Description**: Detailed performance results for all 54 model-dataset combinations with bootstrap confidence intervals
- **Rows**: 54 combinations
- **Key Columns**: Dataset, Model, Type, R², R² (±95% CI), p-value, MAE, Formatted R²
- **Statistical Details**: 
  - Bootstrap confidence intervals (1000 iterations)
  - Statistical significance testing (p < 0.05)
  - Complete performance metrics for all models

### Supplementary Table S2: Hyperparameter Optimization Log
- **File**: `hyperparameter_search_log.csv` (in logs/ directory)
- **Description**: Complete log of 800+ hyperparameter optimization trials
- **Rows**: 801 trials (including header)
- **Key Columns**: trial_id, dataset, model, params, param_hash, cv_score_mean, cv_score_std, training_time_seconds, timestamp, random_seed, cv_folds, status
- **Optimization Details**:
  - Optuna TPE (Tree-structured Parzen Estimator) optimization
  - 30 trials per model-dataset combination
  - 5-fold cross-validation for each trial
  - Complete parameter space exploration

### Supplementary Table S3: Data Leakage Detection Results
- **File**: `complete_sanity_check_results.csv`
- **Description**: Label permutation test results to detect data leakage
- **Rows**: 9 datasets (all datasets tested)
- **Key Columns**: dataset, original_r2, permuted_r2, pass_sanity_check, n_features, n_samples
- **Validation Details**:
  - Label permutation test for each dataset
  - Comparison of original vs. permuted performance
  - Pass/fail criteria for data leakage detection
  - Feature and sample size information

### Supplementary Table S4: Small Sample Analysis
- **File**: `small_sample_analysis.csv`
- **Description**: Analysis of excluded datasets due to insufficient sample size
- **Rows**: 2 excluded datasets (phyto_long, phyto_wide)
- **Key Columns**: Dataset, Samples, Variables, Sample/Feature Ratio, Benchmark requirement, Meets benchmark requirement, Curse of Dimensionality Risk, Exclusion Reason, Data Quality, Recommendation
- **Analysis Details**:
  - Benchmark sample threshold (N ≥ 500)
  - Sample/feature ratio rule for multivariate cross-sectional data
  - Data quality evaluation
  - Recommendations for future use

## Enhanced and Alternative Table Versions

### Enhanced Table Versions
- **File**: `improved_table1_dataset_characteristics.csv`
- **Description**: Enhanced version of Table 1 with additional metadata
- **File**: `improved_table2_with_delta.csv`
- **Description**: Table 2 with performance differences between models
- **File**: `improved_table3_best_performance.csv`
- **Description**: Enhanced best performance table with additional statistics
- **File**: `improved_table4_validation_summary.csv`
- **Description**: Expanded validation summary with robustness metrics

### Paper-Ready Formatted Versions
- **File**: `final_table2_enhanced_with_ci.csv`
- **Description**: Table 2 formatted for publication with confidence intervals
- **File**: `final_table2_for_paper.csv`
- **Description**: Compact version of Table 2 optimized for paper layout

### Summary and Meta-Analysis
- **File**: `final_tables_summary.md`
- **Description**: Comprehensive summary of all tables with key findings and interpretations

## Usage Guidelines

### Loading Tables in Python
```python
import pandas as pd

# Load main performance results
performance = pd.read_csv('outputs/tables/final_table2_model_performance.csv')

# Load supplementary results with confidence intervals
supplementary = pd.read_csv('outputs/tables/supplementary_table_s2_full_results.csv')

# Load hyperparameter optimization log
hyperparams = pd.read_csv('logs/hyperparameter_search_log.csv')

# Load data leakage detection results
sanity_check = pd.read_csv('outputs/tables/complete_sanity_check_results.csv')
```

### Key Findings Summary

1. **Model Robustness**: XGBoost demonstrates best overall robustness (mean R² = 0.823 ± 0.320)
2. **Deep Learning Limitations**: LSTM and Transformer only applicable to 5/9 datasets
3. **Data Quality Impact**: No correlation between sample size and predictability
4. **Statistical Validation**: All results include bootstrap confidence intervals and significance testing
5. **Reproducibility**: Complete hyperparameter logs ensure full reproducibility

## File Integrity

All supplementary tables have been validated for:
- Data completeness and consistency
- Statistical accuracy of confidence intervals
- Proper formatting for analysis and publication
- Cross-reference consistency between related tables

## Citation

When using these supplementary tables, please cite the main paper and reference the specific supplementary table used in your analysis.

---

*This index was generated as part of the Marine ML Benchmark reproducibility package. For questions about specific tables, refer to the main documentation or the detailed methodology.*
