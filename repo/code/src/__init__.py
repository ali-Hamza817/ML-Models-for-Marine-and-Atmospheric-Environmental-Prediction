"""
Marine ML Benchmark: Cross-Dataset Robustness Evaluation

This package provides tools for evaluating machine learning model robustness
across heterogeneous oceanographic datasets for chlorophyll-a prediction.

Modules:
    preprocess: Data preprocessing and feature engineering
    train_enhanced: Model training with hyperparameter optimization
    evaluate_enhanced: Model evaluation with statistical analysis
    visualize: Results visualization and plotting
    utils_io: Input/output utilities and configuration handling
"""

__version__ = "1.0.0"
__author__ = "Marine ML Benchmark Contributors"
__email__ = "[your.email@institution.edu]"

# Minimal public API (avoid importing missing symbols)
from .utils_io import read_cfg

__all__ = ["read_cfg"]
