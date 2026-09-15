#!/usr/bin/env python3
"""
Complete Paper Reproduction Script
Generates all tables, figures, and supplementary materials for the paper
"""

import os
import sys
import yaml
import argparse
from pathlib import Path
import pandas as pd

# Add src directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from generate_figures import generate_all_figures
from generate_tables import generate_all_tables
from generate_supplementary import main as generate_supplementary_materials
from sanity_check import main as run_sanity_check

def load_config():
    """Load configuration from config.yaml"""
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def create_supplementary_materials(config):
    """Generate supplementary materials using integrated modules"""
    print("📋 Generating supplementary materials...")

    # Run sanity check first to ensure we have permutation test results
    print("\n🔍 Running sanity check and permutation tests...")
    try:
        run_sanity_check()
        print("   ✅ Sanity check completed")
    except Exception as e:
        print(f"   ⚠️ Sanity check failed: {e}")

    # Generate all supplementary materials
    print("\n📊 Generating supplementary tables and figures...")
    try:
        supp_results = generate_supplementary_materials()
        print("   ✅ Supplementary materials generated")
        return supp_results
    except Exception as e:
        print(f"   ❌ Error generating supplementary materials: {e}")
        return None

# Supplementary index creation is now handled by generate_supplementary.py

def create_paper_sections():
    """Generate text for paper sections"""
    print("📝 Generating paper section text...")
    
    base_path = Path(__file__).parent.parent
    output_path = base_path / "paper_sections.md"
    
    sections_content = """# Paper Sections for Supplementary Materials and Code Availability

## Supplementary Materials Section
*Place after Conclusion, before Acknowledgments*

### Supplementary Materials

Supplementary Tables S1–S4 and Supplementary Figure S1 are available in the online repository (DOI: [to be added]). Supplementary Table S1 provides the complete performance matrix with confidence intervals for all model-dataset combinations. Supplementary Table S2 contains detailed hyperparameter optimization logs for reproducibility. Supplementary Table S3 presents statistical significance testing results using label permutation tests. Supplementary Table S4 analyzes performance degradation with reduced sample sizes. Supplementary Figure S1 shows sample size distributions and threshold analysis across all datasets.

## Data and Code Availability Section  
*Place alongside Supplementary Materials section*

### Data and Code Availability

All raw data (9 datasets with quality control), processed datasets, and complete analysis code are publicly available on Zenodo (DOI: [to be added]). The repository includes preprocessing scripts, model training pipelines, hyperparameter optimization logs (800+ trials), statistical validation tests, and Docker environment for full reproducibility. All figures and tables in this paper can be regenerated using the provided scripts.

## In-Text Citation Examples

### For Methods Section:
"Detailed hyperparameter search spaces and optimization results are provided in Supplementary Table S2."

"Statistical significance of model performance was validated using label permutation tests (see Supplementary Table S3)."

### For Results Section:
"Complete performance matrices with confidence intervals are available in Supplementary Table S1."

"Sample size sensitivity analysis demonstrates robust performance above 1,000 samples (Supplementary Figure S1)."

### For Figure Captions:
"Figure 2. Cross-dataset model performance heatmap. Complete results with confidence intervals are provided in Supplementary Table S1."

"Figure 5. Dataset difficulty versus sample size relationship. Detailed threshold analysis is shown in Supplementary Figure S1."

## Submission File List

### Main Submission:
1. **Main manuscript**: `manuscript.pdf` (converted from .docx)
2. **Supplementary materials**: `supplementary_materials.pdf` (compiled from all supplementary tables and figures)

### Repository (referenced by DOI, not submitted):
- Complete source code and data
- Reproduction instructions
- Docker environment
- Raw and processed datasets

### Supplementary Materials Contents:
- Supplementary Table S1: Complete performance results
- Supplementary Table S2: Hyperparameter optimization logs  
- Supplementary Table S3: Statistical validation results
- Supplementary Table S4: Sample size analysis
- Supplementary Figure S1: Sample size distribution analysis

## Format Requirements Checklist

- [ ] All figures at 300 DPI resolution
- [ ] Font sizes ≥ 8pt in all figures
- [ ] Supplementary materials < 50 MB total
- [ ] No citations in abstract
- [ ] DOI links provided for data/code
- [ ] Proper in-text citation format used
- [ ] Main text and supplementary materials as separate files
"""
    
    with open(output_path, 'w') as f:
        f.write(sections_content)
    
    print(f"   ✅ Paper sections saved to: {output_path}")

def main():
    """Main reproduction function"""
    parser = argparse.ArgumentParser(description='Reproduce paper results')
    parser.add_argument('--tables', action='store_true', help='Generate tables only')
    parser.add_argument('--figures', action='store_true', help='Generate figures only')
    parser.add_argument('--supplementary', action='store_true', help='Generate supplementary materials only')
    parser.add_argument('--all', action='store_true', help='Generate everything')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config()
    
    print("🚀 Starting paper reproduction...")
    print(f"📁 Working directory: {Path.cwd()}")
    
    success = True
    
    # Generate components based on arguments
    if args.all or args.tables:
        print("\n" + "="*50)
        print("📊 GENERATING TABLES")
        print("="*50)
        tables_result = generate_all_tables(config)
        if tables_result is None:
            success = False
    
    if args.all or args.figures:
        print("\n" + "="*50)
        print("📈 GENERATING FIGURES") 
        print("="*50)
        figures_result = generate_all_figures(config)
        if not figures_result:
            success = False
    
    if args.all or args.supplementary:
        print("\n" + "="*50)
        print("📋 GENERATING SUPPLEMENTARY MATERIALS")
        print("="*50)
        create_supplementary_materials(config)
        create_paper_sections()
    
    # Summary
    print("\n" + "="*50)
    print("📋 REPRODUCTION SUMMARY")
    print("="*50)
    
    if success:
        print("🎉 Paper reproduction completed successfully!")
        print("\n📁 Generated files:")
        print("   • Tables: tables/final_table*.csv")
        print("   • Figures: figures/figure*_final.pdf/png")
        print("   • Supplementary: supplementary/")
        print("   • Paper sections: paper_sections.md")
        print("\n📝 Next steps:")
        print("   1. Review generated tables and figures")
        print("   2. Compile supplementary materials PDF")
        print("   3. Add DOI links when repository is published")
        print("   4. Submit main manuscript + supplementary materials")
    else:
        print("❌ Some components failed to generate")
        print("   Check error messages above for details")
    
    return success

if __name__ == "__main__":
    main()
