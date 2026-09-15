
# 压缩版图表生成报告

## 生成的文件

### 主要图表 (3图+1表)

#### Figure 1: Cross-dataset Overview
- **文件**: Fig1_overview.pdf/png
- **内容**: (a) R² heatmap, (b) ΔR² vs baseline, (c) model win-rate, (d) normalized MAE
- **题注**: Figure 1. Cross-dataset overview: (a) R² heatmap across models and datasets; (b) improvement over the baseline (ΔR²) with 95% CIs; (c) model win-rate across datasets; (d) normalized MAE. Full numeric results are available in Table S1 at DOI: 10.5281/zenodo.16832373.

#### Figure 2: Robustness and Generalization
- **文件**: Fig2_robustness.pdf/png
- **内容**: (a) learning curves, (b) robustness to noise/missingness, (c) temporal holdout, (d) stability across random seeds/splits
- **题注**: Figure 2. Robustness and generalization: (a) learning curves; (b) robustness to noise/missingness; (c) temporal holdout; (d) stability across random seeds/splits. Extended analyses are in Figs. S2–S4 (DOI: 10.5281/zenodo.16832373).

#### Figure 3: Model Interpretability
- **文件**: Fig3_interpretability.pdf/png
- **内容**: (a) global feature importance, (b) partial dependence for the top driver
- **题注**: Figure 3. Model interpretability on a representative dataset: (a) global feature importance; (b) partial dependence for the top driver. More examples are in Fig. S3 (DOI: 10.5281/zenodo.16832373).

#### Table 1: Main Results Summary
- **文件**: Table1_main_results.csv
- **内容**: Dataset | Type | #Samples | Best Model | R² | MAE | ΔR² vs Baseline | Difficulty
- **题注**: Table 1. Cross-dataset summary of main results. For each dataset we report the best-performing model and metrics (R², MAE, improvement over the baseline), together with the permutation-test p-value and difficulty rank. Complete per-model results appear in Table S1 (DOI: 10.5281/zenodo.16832373).

## 技术规格

- **字体**: Times New Roman, 9pt基础
- **分辨率**: 300 DPI
- **格式**: PDF + PNG双格式
- **配色**: 色盲友好，统一模型颜色
- **尺寸**: 适配SPIE期刊标准

## 数据来源

所有图表基于项目中的真实数据：
- outputs/tables/supplementary_table_s2_full_results.csv (54条记录)
- outputs/tables/final_table1_dataset_characteristics.csv (9个数据集)
- 7个验证通过的数据集，2个排除的小样本数据集

## 正文替换建议

### 图表引用更新
- "Fig.2–5 展示整体表现与稳健性" → "Fig.1–2 展示整体表现与稳健性；可解释性见Fig.3与Fig. S3。"
- "见Table 3难度排名、Table 4验证汇总" → "见Table 1主结果；全量与排名见Table S1（DOI）。"

### 文字替换
- 数据概览："7/9数据集经QA保留，规模4.6k–103k，涵盖时序与截面任务（Fig. S1）。"
- 质量>数量："样本量与可预测性相关性弱于质量（Fig. S2）。"
- 流程："我们采用统一管线：QA→时序切分→训练/评估→显著性→稳健性（细节见DOI）。"

## 补充材料移动

原有图表移至补充材料：
- Fig. S1: 原Fig.1 数据集概览
- Fig. S2: 原Fig.5 难度vs样本量
- Fig. S3: 原Fig.6 完整特征重要性
- Fig. S4: 原Fig.7 技术路线图
- Table S1: 原Table 2 完整性能结果
- Table S2: 原Table 1 数据集目录

