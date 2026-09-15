
# Supplementary Material: Small Sample Size Datasets

## S1. Exclusion Criteria for Small Sample Datasets

Two datasets were excluded from the main model comparison analysis due to insufficient sample sizes:

### S1.1 phyto_long Dataset
- **Sample size**: N = 82
- **Variables**: 1 (phytoplankton abundance)
- **Exclusion reason**: Below benchmark inclusion threshold
- **Benchmark requirement**: N ≥ 500
- **Data quality**: High-quality measurements but insufficient quantity for robust model evaluation

### S1.2 phyto_wide Dataset  
- **Sample size**: N = 440
- **Variables**: 46 (species-level phytoplankton data)
- **Exclusion reason**: Below benchmark inclusion threshold and insufficient sample-to-feature ratio
- **Sample-to-feature ratio**: 9.6:1 (benchmark minimum: 10:1 for multivariate cross-sectional data)
- **Benchmark requirement**: N ≥ 500 and sample/feature ratio ≥ 10
- **Data quality**: Comprehensive species-level data but dimensionality challenges

## S2. Sample Size Guidelines Applied

Following the benchmark inclusion rules used in the revised manuscript:

1. **Benchmark sample threshold**: Minimum N ≥ 500 for benchmark inclusion
2. **Multivariate cross-sectional rule**: Require sample/feature ratio ≥ 10
3. **Chronological evaluation**: Sufficient held-out test size must remain after splitting
4. **Excluded datasets**: Can still support descriptive statistics and exploratory summaries

## S3. Descriptive Statistics for Excluded Datasets

### phyto_long (N=82):
- Mean phytoplankton abundance: [value]
- Standard deviation: [value]  
- Range: [min] - [max]
- Data collection period: [period]

### phyto_wide (N=440):
- Number of species detected: 46
- Mean species richness per sample: [value]
- Dominant species: [list top 5]
- Spatial coverage: [description]

## S4. Implications for Marine Ecosystem Modeling

The exclusion of small sample datasets highlights important considerations:

1. **Data collection priorities**: Emphasis on sustained, long-term monitoring
2. **Species-level modeling**: Requires substantial sample sizes for reliable predictions
3. **Community-level approaches**: May be more feasible with limited samples
4. **Temporal vs. spatial trade-offs**: Balance between temporal resolution and sample size

## S5. Recommendations for Future Studies

1. **Minimum sample sizes**: N ≥ 500 for benchmark inclusion
2. **Data aggregation strategies**: Consider temporal or spatial pooling
3. **Dimensionality reduction**: Apply PCA or feature selection for high-dimensional data
4. **Ensemble approaches**: Combine multiple small datasets when appropriate

---

*Note: Complete raw data and metadata for excluded datasets are available in the supplementary data repository.*
