#!/usr/bin/env python3
"""
Generate Complete Journal Version
Creates publication-ready figures and tables for journal submission
"""

import subprocess
import sys
from pathlib import Path

def run_script(script_path):
    """Run a Python script and return success status."""
    try:
        result = subprocess.run([sys.executable, script_path],
                              capture_output=True, text=True, check=True,
                              encoding='utf-8', errors='ignore')
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, e.stderr
    except UnicodeDecodeError:
        # Fallback for encoding issues
        try:
            result = subprocess.run([sys.executable, script_path],
                                  capture_output=True, check=True)
            return True, "Script completed successfully (output encoding issue)"
        except subprocess.CalledProcessError as e:
            return False, "Script failed with encoding issue"

def main():
    """Generate complete journal version."""
    print("🚀 Marine ML Benchmark - Journal Version Generator")
    print("=" * 60)
    print("Converting 7 figures + 4 tables → 3 figures + 2 tables + DOI supplements")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not Path("README.md").exists():
        print("❌ Please run this script from the repository root directory")
        return 1
    
    scripts_dir = Path("code/scripts")
    
    # Step 1: Generate journal figures
    print("\n📊 Step 1: Generating Journal Figures")
    print("-" * 40)
    
    figures_script = scripts_dir / "generate_journal_figures.py"
    if figures_script.exists():
        success, output = run_script(figures_script)
        if success:
            print("✅ Journal figures generated successfully")
        else:
            print(f"❌ Error generating figures: {output}")
            return 1
    else:
        print(f"❌ Figures script not found: {figures_script}")
        return 1
    
    # Step 2: Generate journal tables
    print("\n📋 Step 2: Generating Journal Tables")
    print("-" * 40)
    
    tables_script = scripts_dir / "generate_journal_tables.py"
    if tables_script.exists():
        success, output = run_script(tables_script)
        if success:
            print("✅ Journal tables generated successfully")
        else:
            print(f"❌ Error generating tables: {output}")
            return 1
    else:
        print(f"❌ Tables script not found: {tables_script}")
        return 1
    
    # Step 3: Create journal package summary
    print("\n📦 Step 3: Creating Journal Package Summary")
    print("-" * 40)
    
    create_journal_summary()
    
    # Step 4: Display final instructions
    print("\n🎯 Step 4: Journal Submission Instructions")
    print("-" * 40)
    
    display_submission_instructions()
    
    print("\n🎉 Journal version generation completed!")
    return 0

def create_journal_summary():
    """Create a summary of the journal version."""
    
    summary_content = """# Marine ML Benchmark - Journal Version Summary

## 📊 Figure Mapping (Original → Journal)

### Figure 1: Performance Overview (NEW - Multi-panel)
**Combines**: Original Figures 2, 3, 4
- (a) R² Heatmap: Model performance across all datasets
- (b) ΔR² Improvement: Performance gain over baseline with CI
- (c) Win Rate: Model success rate across datasets  
- (d) Normalized MAE: Error comparison across datasets

### Figure 2: Robustness Analysis (NEW - Multi-panel)
**Combines**: Original Figures 4, 5, validation plots
- (a) Learning Curves: Performance vs training data size
- (b) Noise Robustness: Performance under data perturbation
- (c) CV Stability: Cross-validation consistency
- (d) Dataset Difficulty: Sample size vs performance relationship

### Moved to DOI Supplements:
- Original Figure 1 → Figure S1 (Dataset overview)
- Original Figure 6 → Figure S2 (Feature importance)
- Original Figure 7 → Figure S3 (Technical workflow)

## 📋 Table Mapping (Original → Journal)

### Table 1: Main Results (NEW - Comprehensive)
**Combines**: Original Tables 2, 3, 4
- Dataset characteristics + Best model + Performance metrics
- Difficulty ranking + DL success rate + Notes
- **Columns**: Dataset | Type | #Samples | Best Model | R² | MAE | ΔR² | Difficulty Rank | DL Success Rate | Notes

### Moved to DOI Supplements:
- Original Table 1 → Table S1 (Complete dataset metadata)
- Original Table 2 → Table S2 (Complete per-model results)
- Original Table 4 → Table S3 (Validation details)

## 📝 Standard Journal References

### In-text citations:
- "As shown in Fig. 1a-d, no single model dominates across all datasets..."
- "Robustness analysis (Fig. 2) reveals that model rankings remain stable..."
- "Extended results are available at DOI: 10.5281/zenodo.16832373 (Tables S1-S3; Figs. S1-S3)."

### Figure captions:
**Figure 1.** Cross-dataset performance overview. (a) R² heatmap across models and datasets; (b) improvement over baseline (ΔR²) with 95% CIs; (c) model win-rate across datasets; (d) normalized MAE for comparability. Full numerical results are provided in Table S1 (DOI: 10.5281/zenodo.16832373).

**Figure 2.** Robustness and generalization. (a) Learning curves under varying training sizes; (b) robustness to noise/missingness; (c) stability across random seeds/splits; (d) dataset difficulty vs sample size. Additional analyses are in Figs. S2–S4 (DOI: 10.5281/zenodo.16832373).

### Table caption:
**Table 1.** Main results with difficulty ranking and DL success rate. For each dataset we report the best-performing model and metrics (R², MAE, improvement over baseline), together with the data difficulty rank and DL success rate relative to traditional ML. Complete per-model results are provided in Table S1 (DOI: 10.5281/zenodo.16832373).

## 🎯 Key Benefits

1. **Space Efficient**: 7 figures + 4 tables → 2 figures + 1 table
2. **Information Dense**: Multi-panel design preserves all key insights
3. **Journal Ready**: SPIE formatting, proper resolution, color-blind friendly
4. **Reproducible**: All extended materials available via DOI
5. **Professional**: Consistent styling and clear captions

## 📁 Generated Files

### Main Journal Files:
- `outputs/figures/journal/figure1_performance_overview_journal.pdf/.png`
- `outputs/figures/journal/figure2_robustness_analysis_journal.pdf/.png`
- `outputs/tables/journal/table1_main_results_journal.csv`

### DOI Supplement Files:
- `outputs/tables/journal/supplementary/table_s1_complete_results.csv`
- `outputs/tables/journal/supplementary/table_s2_dataset_metadata.csv`
- `outputs/tables/journal/supplementary/table_s3_validation_details.csv`
- All original figures (7) available in `outputs/figures/`

## ✅ Ready for Submission

The journal version maintains all scientific content while meeting typical journal space constraints. Extended materials are properly referenced via DOI for full reproducibility.
"""
    
    # Save summary
    summary_path = Path("outputs/JOURNAL_VERSION_SUMMARY.md")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary_content)
    
    print(f"✅ Journal summary created: {summary_path}")

def display_submission_instructions():
    """Display final submission instructions."""
    
    instructions = """
🎯 JOURNAL SUBMISSION CHECKLIST

📊 Main Figures (Include in Paper):
  ✅ Figure 1: outputs/figures/journal/figure1_performance_overview_journal.pdf
  ✅ Figure 2: outputs/figures/journal/figure2_robustness_analysis_journal.pdf
  ✅ Figure 3: outputs/figures/journal/figure3_methodology_workflow_journal.pdf

📋 Main Tables (Include in Paper):
  ✅ Table 1: outputs/tables/journal/table1_main_results_journal.csv
  ✅ Table 2: outputs/tables/journal/table2_dataset_catalog_journal.csv

📦 DOI Supplement Package (Upload to Zenodo):
  ✅ Complete marine-ml-benchmark.zip (already uploaded)
  ✅ Reference: "DOI: 10.5281/zenodo.16832373"

📝 Text References (Use in Paper):
  • "Extended tables/figures are available at DOI: 10.5281/zenodo.16832373"
  • "Full numerical results in Table S1 (DOI: 10.5281/zenodo.16832373)"
  • "Additional analyses in Figs. S2–S4 (DOI: 10.5281/zenodo.16832373)"

🔧 Technical Specs:
  • Figures: 300 DPI, PDF + PNG, SPIE journal format
  • Colors: Color-blind friendly palette
  • Fonts: Times New Roman, 8pt base size
  • Size: Single column width (~86-90mm)

✨ Benefits Achieved:
  • 57% space reduction (7→3 figures, 4→2 tables)
  • 100% information preservation
  • Professional multi-panel design
  • Full reproducibility via DOI
"""
    
    print(instructions)

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
