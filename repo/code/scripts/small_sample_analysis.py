#!/usr/bin/env python3
"""
Small Sample Size Analysis and Documentation
极小样本集分析和文档说明
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_TABLES = REPO_ROOT / "outputs" / "tables"
OUT_FIGS = REPO_ROOT / "outputs" / "figures"
OUT_REVISION_FIGS = REPO_ROOT / "outputs" / "revision_figures"

def analyze_small_samples():
    """分析极小样本数据集"""
    print("📊 Small Sample Size Analysis")
    print("=" * 50)
    
    # 小样本数据集信息
    small_datasets = {
        'phyto_long': {
            'samples': 82,
            'variables': 1,
            'type': 'Cross-sectional',
            'target': 'phytoplankton_abundance',
            'reason_excluded': 'Below benchmark inclusion threshold (N < 500)',
            'benchmark_requirement': 'N ≥ 500',
            'data_quality': 'High quality but insufficient quantity'
        },
        'phyto_wide': {
            'samples': 440,
            'variables': 46,
            'type': 'Cross-sectional', 
            'target': 'phytoplankton_abundance',
            'reason_excluded': 'Below benchmark inclusion threshold and insufficient sample/feature ratio',
            'benchmark_requirement': 'N ≥ 500 and sample/feature ratio ≥ 10',
            'data_quality': 'High dimensional species data'
        }
    }
    
    # 创建分析报告
    analysis_results = []
    
    for dataset, info in small_datasets.items():
        # 计算样本-特征比
        sample_feature_ratio = info['samples'] / info['variables']
        
        # 判断是否满足benchmark纳入要求
        if dataset == 'phyto_long':
            meets_benchmark = info['samples'] >= 500
        else:
            meets_benchmark = (info['samples'] >= 500) and (sample_feature_ratio >= 10)
        
        # 维度诅咒风险
        curse_risk = "HIGH" if info['variables'] > info['samples'] else "LOW"
        
        analysis_results.append({
            'Dataset': dataset,
            'Samples': info['samples'],
            'Variables': info['variables'],
            'Sample/Feature Ratio': f"{sample_feature_ratio:.2f}",
            'Benchmark requirement': info['benchmark_requirement'],
            'Meets benchmark requirement': meets_benchmark,
            'Curse of Dimensionality Risk': curse_risk,
            'Exclusion Reason': info['reason_excluded'],
            'Data Quality': info['data_quality'],
            'Recommendation': 'Descriptive statistics only' if not meets_benchmark else 'Proceed with caution'
        })
        
        print(f"\n📋 {dataset.upper()}")
        print(f"   Samples: {info['samples']}")
        print(f"   Variables: {info['variables']}")
        print(f"   Sample/Feature ratio: {sample_feature_ratio:.2f}")
        print(f"   Benchmark requirement: {info['benchmark_requirement']}")
        print(f"   Status: {'✅ ELIGIBLE' if meets_benchmark else '❌ EXCLUDED'}")
        print(f"   Curse risk: {curse_risk}")
        print(f"   Reason: {info['reason_excluded']}")
    
    # 保存分析结果
    analysis_df = pd.DataFrame(analysis_results)
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    analysis_df.to_csv(OUT_TABLES / "small_sample_analysis.csv", index=False)
    
    return analysis_df

def create_sample_size_guidelines():
    """创建样本量指导原则"""
    print(f"\n📏 Sample Size Guidelines")
    print("=" * 50)
    
    guidelines = {
        'Machine Learning Task': [
            'Simple Linear Regression',
            'Multiple Linear Regression', 
            'Random Forest',
            'Support Vector Regression',
            'Neural Networks (Simple)',
            'Deep Learning (LSTM)',
            'Cross-validation (5-fold)',
            'Bootstrap Confidence Intervals'
        ],
        'Minimum Sample Size': [
            30,
            '10 × n_features',
            '5 × n_features (min 100)',
            '10 × n_features (min 200)',
            '100 × n_features',
            '1000+ (time series)',
            '50 (10 per fold)',
            '30 (for stable estimates)'
        ],
        'Recommended Size': [
            100,
            '20 × n_features',
            '10 × n_features (min 500)',
            '20 × n_features (min 1000)',
            '500 × n_features',
            '5000+ (time series)',
            '500 (100 per fold)',
            '100 (for robust estimates)'
        ],
        'Our Datasets Status': [
            'Most datasets adequate',
            'Adequate for low-dim datasets',
            'Adequate for most datasets',
            'Adequate for most datasets',
            'Only large datasets adequate',
            'Only era5_daily adequate',
            'Most datasets adequate',
            'Most datasets adequate'
        ]
    }
    
    guidelines_df = pd.DataFrame(guidelines)
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    guidelines_df.to_csv(OUT_TABLES / "sample_size_guidelines.csv", index=False)
    
    print(guidelines_df.to_string(index=False))
    
    return guidelines_df

def generate_supplementary_text():
    """生成补充材料文本"""
    
    supplement_text = """
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
"""
    
    # 保存补充材料文本
    with open(REPO_ROOT / "supplementary_small_samples.md", 'w', encoding='utf-8') as f:
        f.write(supplement_text)
    
    print(f"✅ Supplementary material text generated")
    print(f"   File: supplementary_small_samples.md")
    
    return supplement_text

def create_sample_size_visualization():
    """创建样本量可视化"""
    print(f"\n📊 Creating Sample Size Visualization")
    
    # 所有数据集的样本量
    datasets = {
        'era5_daily': 102982,
        'cast': 21865, 
        'rolling_mean': 8855,
        'processed_seq': 8039,
        'cleaned_data': 7819,
        'biotoxin': 5076,
        'hydrographic': 4653,
        'phyto_wide': 440,
        'phyto_long': 82
    }
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 子图1：所有数据集（对数尺度）
    names = list(datasets.keys())
    sizes = list(datasets.values())
    colors = ['green' if s >= 500 else 'red' for s in sizes]
    
    bars1 = ax1.bar(names, sizes, color=colors, alpha=0.7)
    ax1.set_yscale('log')
    ax1.set_ylabel('Sample Size (log scale)')
    ax1.set_title('Dataset Sample Sizes')
    ax1.tick_params(axis='x', rotation=45)
    
    # 添加阈值线
    ax1.axhline(
        y=500,
        color='green',
        linestyle='--',
        alpha=0.8,
        label='Benchmark-eligible by sample size (N ≥ 500)'
    )
    ax1.axhline(
        y=499,
        color='red',
        linestyle=':',
        alpha=0.8,
        label='Below benchmark sample threshold (N < 500)'
    )
    ax1.legend()
    
    # 子图2：样本量分布
    size_categories = ['<100', '100-500', '500-1000', '1000-5000', '5000+']
    counts = [
        sum(1 for s in sizes if s < 100),
        sum(1 for s in sizes if 100 <= s < 500),
        sum(1 for s in sizes if 500 <= s < 1000),
        sum(1 for s in sizes if 1000 <= s < 5000),
        sum(1 for s in sizes if s >= 5000)
    ]
    
    colors2 = ['red', 'orange', 'yellow', 'lightgreen', 'green']
    ax2.pie(counts, labels=size_categories, colors=colors2, autopct='%1.0f%%')
    ax2.set_title('Sample Size Distribution')
    
    plt.tight_layout()
    OUT_FIGS.mkdir(parents=True, exist_ok=True)
    OUT_REVISION_FIGS.mkdir(parents=True, exist_ok=True)
    try:
        plt.savefig(OUT_FIGS / "sample_size_analysis.png", dpi=300, bbox_inches='tight')
    except PermissionError:
        print("⚠️ Could not write to outputs/figures; saved revision copy instead")
    plt.savefig(OUT_REVISION_FIGS / "sample_size_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Sample size visualization saved: figures/sample_size_analysis.png")

def main():
    """主函数"""
    print("📊 Small Sample Size Analysis and Documentation")
    print("=" * 60)
    
    # 分析小样本数据集
    analysis_df = analyze_small_samples()
    
    # 创建样本量指导原则
    guidelines_df = create_sample_size_guidelines()
    
    # 生成补充材料文本
    supplement_text = generate_supplementary_text()
    
    # 创建可视化
    create_sample_size_visualization()
    
    print(f"\n🎯 SMALL SAMPLE ANALYSIS COMPLETE")
    print("=" * 60)
    print("✅ Small sample datasets analyzed")
    print("✅ Exclusion criteria documented") 
    print("✅ Sample size guidelines established")
    print("✅ Supplementary material generated")
    print("✅ Visualization created")
    
    print(f"\n📁 Generated Files:")
    print(f"   - tables/small_sample_analysis.csv")
    print(f"   - tables/sample_size_guidelines.csv")
    print(f"   - supplementary_small_samples.md")
    print(f"   - figures/sample_size_analysis.png")

if __name__ == "__main__":
    main()
