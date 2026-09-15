# Changelog

All notable changes to the Marine ML Benchmark project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-01-15

### Added
- Initial release of Marine ML Benchmark
- Complete implementation of cross-dataset robustness evaluation framework
- Support for 5 machine learning models (RF, XGBoost, SVR, LSTM, Transformer)
- Standardized preprocessing pipeline for 7 validated oceanographic datasets
- Comprehensive evaluation protocol with statistical analysis
- Bootstrap confidence intervals and significance testing
- SHAP-based feature importance analysis
- Publication-ready figure generation (7 figures)
- Complete documentation and methodology description
- Unit tests for all major components
- Reproducible execution scripts (full pipeline and quick test)

### Datasets
- **era5_daily**: ERA5 meteorological reanalysis data (102,982 samples)
- **cast**: CTD oceanographic measurements (21,865 samples)
- **cleaned_data**: Multi-parameter oceanographic data (7,819 samples)
- **rolling_mean**: 7-day smoothed chlorophyll-a data (8,855 samples)
- **processed_seq**: Time series processed data (8,039 samples)
- **biotoxin**: Harmful algal bloom toxin data (5,076 samples)
- **hydrographic**: Hydrographic survey data (4,653 samples)

### Models
- **Traditional ML**: Random Forest, XGBoost, Support Vector Regression
- **Deep Learning**: LSTM, Transformer (for time series data)
- **Baselines**: LASSO, Ridge, Mean predictor

### Features
- **Data Processing**: Quality control, outlier detection, missing value imputation
- **Feature Engineering**: Temporal features, rolling statistics, sequence generation
- **Hyperparameter Optimization**: Optuna-based optimization with 30 trials per model
- **Cross-validation**: TimeSeriesSplit for temporal data, StratifiedKFold for cross-sectional
- **Statistical Analysis**: Bootstrap confidence intervals, paired t-tests, effect sizes
- **Visualization**: 7 publication-ready figures with SPIE journal formatting
- **Reproducibility**: Fixed random seeds, comprehensive logging, unit tests

### Documentation
- **README.md**: Complete project overview and usage instructions
- **METHODOLOGY.md**: Detailed methodology documentation
- **data/README_DATA.md**: Comprehensive dataset documentation
- **LICENSE**: MIT license for code, CC BY 4.0 for data
- **CITATION.cff**: Standardized citation format

### Results
- **Key Finding**: XGBoost demonstrates best overall robustness (mean R² = 0.823 ± 0.320)
- **Model Ranking**: XGBoost > Random Forest > SVR > LSTM > Transformer
- **Dataset Difficulty**: 6 easy (R² > 0.9), 2 medium (0.6-0.9), 1 hard (R² < 0.3)
- **Deep Learning Limitation**: Only applicable to 5/9 datasets due to data structure requirements
- **Data Quality Impact**: No correlation between sample size and predictability

### Technical Specifications
- **Python**: 3.8+ compatibility
- **Dependencies**: scikit-learn, XGBoost, PyTorch, Optuna, matplotlib, seaborn
- **Performance**: Optimized for standard CPU hardware
- **Memory**: Requires 4GB+ RAM for full pipeline
- **Storage**: ~2GB for complete dataset and results

### Quality Assurance
- **Code Coverage**: >90% test coverage for core functions
- **Validation**: All 7 datasets pass quality validation criteria
- **Reproducibility**: All results reproducible with provided scripts and seeds
- **Statistical Rigor**: Proper confidence intervals and significance testing
- **Documentation**: Comprehensive documentation for all components

## [Unreleased]

### Planned Features
- Multi-region validation (beyond Bohai Sea)
- Transfer learning evaluation
- Ensemble methods implementation
- Real-time prediction capabilities
- Interactive web dashboard
- Additional deep learning architectures
- Uncertainty quantification methods

### Potential Improvements
- GPU acceleration for deep learning models
- Distributed computing support for large datasets
- Advanced feature selection methods
- Automated hyperparameter tuning
- Model interpretability enhancements
- Performance optimization

## Development Notes

### Version Numbering
- **Major version** (X.0.0): Breaking changes, major new features
- **Minor version** (0.X.0): New features, backward compatible
- **Patch version** (0.0.X): Bug fixes, minor improvements

### Release Process
1. Update version numbers in relevant files
2. Update CHANGELOG.md with new features and changes
3. Run full test suite and validation
4. Generate updated documentation
5. Create release tag and GitHub release
6. Update citation information

### Contributing Guidelines
- All changes must include appropriate tests
- Documentation must be updated for new features
- Code must pass all existing tests
- Follow existing code style and conventions
- Include performance benchmarks for significant changes

### Known Issues
- Deep learning models require significant computational resources
- Some datasets have temporal gaps that may affect time series models
- Bootstrap confidence intervals can be computationally intensive
- Large datasets may require substantial memory for processing

### Future Compatibility
- Python 3.8+ support will be maintained
- Backward compatibility for data formats
- API stability for core functions
- Migration guides for breaking changes

---

For detailed information about any release, please refer to the corresponding GitHub release notes and documentation.
