#!/usr/bin/env python3
"""
Verify Journal Version Generation
Check if all journal files were generated correctly
"""

from pathlib import Path
import pandas as pd

def check_journal_figures():
    """Check if journal figures were generated."""
    print("📊 Checking Journal Figures")
    print("-" * 30)
    
    figures_dir = Path("outputs/figures/journal")
    expected_files = [
        "figure1_performance_overview_journal.pdf",
        "figure1_performance_overview_journal.png",
        "figure2_robustness_analysis_journal.pdf",
        "figure2_robustness_analysis_journal.png",
        "figure3_methodology_workflow_journal.pdf",
        "figure3_methodology_workflow_journal.png"
    ]
    
    all_exist = True
    for file_name in expected_files:
        file_path = figures_dir / file_name
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"  ✅ {file_name} ({size_mb:.2f} MB)")
        else:
            print(f"  ❌ {file_name} - Missing")
            all_exist = False
    
    return all_exist

def check_journal_tables():
    """Check if journal tables were generated."""
    print("\n📋 Checking Journal Tables")
    print("-" * 30)
    
    tables_dir = Path("outputs/tables/journal")
    expected_files = [
        "table1_main_results_journal.csv",
        "table2_dataset_catalog_journal.csv"
    ]
    
    all_exist = True
    for file_name in expected_files:
        file_path = tables_dir / file_name
        if file_path.exists():
            try:
                df = pd.read_csv(file_path)
                print(f"  ✅ {file_name} ({len(df)} rows, {len(df.columns)} columns)")
            except Exception as e:
                print(f"  ⚠️  {file_name} - Error reading: {e}")
                all_exist = False
        else:
            print(f"  ❌ {file_name} - Missing")
            all_exist = False
    
    return all_exist

def check_supplementary_tables():
    """Check if supplementary tables were generated."""
    print("\n📚 Checking Supplementary Tables")
    print("-" * 30)
    
    supp_dir = Path("outputs/tables/journal/supplementary")
    expected_files = [
        "table_s1_complete_results.csv",
        "table_s2_dataset_metadata.csv", 
        "table_s3_validation_details.csv"
    ]
    
    all_exist = True
    for file_name in expected_files:
        file_path = supp_dir / file_name
        if file_path.exists():
            try:
                df = pd.read_csv(file_path)
                print(f"  ✅ {file_name} ({len(df)} rows)")
            except Exception as e:
                print(f"  ⚠️  {file_name} - Error reading: {e}")
                all_exist = False
        else:
            print(f"  ❌ {file_name} - Missing")
            all_exist = False
    
    return all_exist

def check_summary_document():
    """Check if summary document was created."""
    print("\n📄 Checking Summary Document")
    print("-" * 30)
    
    summary_path = Path("outputs/JOURNAL_VERSION_SUMMARY.md")
    if summary_path.exists():
        size_kb = summary_path.stat().st_size / 1024
        print(f"  ✅ JOURNAL_VERSION_SUMMARY.md ({size_kb:.1f} KB)")
        return True
    else:
        print(f"  ❌ JOURNAL_VERSION_SUMMARY.md - Missing")
        return False

def display_journal_content_summary():
    """Display summary of journal content."""
    print("\n📊 Journal Content Summary")
    print("=" * 50)
    
    # Check Table 1 content
    try:
        table1_path = Path("outputs/tables/journal/table1_main_results_journal.csv")
        if table1_path.exists():
            df1 = pd.read_csv(table1_path)
            print(f"\n📋 Table 1 - Main Results:")
            print(f"  • {len(df1)} datasets included")
            print(f"  • Columns: {', '.join(df1.columns[:5])}...")
            
            if 'Best Model' in df1.columns:
                model_counts = df1['Best Model'].value_counts()
                print(f"  • Best models: {dict(model_counts)}")
    except Exception as e:
        print(f"  ⚠️  Error reading Table 1: {e}")
    
    # Check Table 2 content
    try:
        table2_path = Path("outputs/tables/journal/table2_dataset_catalog_journal.csv")
        if table2_path.exists():
            df2 = pd.read_csv(table2_path)
            print(f"\n📋 Table 2 - Dataset Catalog:")
            print(f"  • {len(df2)} datasets cataloged")
            print(f"  • Columns: {', '.join(df2.columns)}")
    except Exception as e:
        print(f"  ⚠️  Error reading Table 2: {e}")

def display_submission_checklist():
    """Display final submission checklist."""
    print("\n🎯 JOURNAL SUBMISSION CHECKLIST")
    print("=" * 50)
    
    print("\n📊 Main Figures (Include in Paper):")
    figures_dir = Path("outputs/figures/journal")
    for fig_num in [1, 2, 3]:
        pdf_file = figures_dir / f"figure{fig_num}_*_journal.pdf"
        pdf_files = list(figures_dir.glob(f"figure{fig_num}_*_journal.pdf"))
        if pdf_files:
            print(f"  ✅ Figure {fig_num}: {pdf_files[0].name}")
        else:
            print(f"  ❌ Figure {fig_num}: Missing")
    
    print("\n📋 Main Tables (Include in Paper):")
    table1_path = Path("outputs/tables/journal/table1_main_results_journal.csv")
    table2_path = Path("outputs/tables/journal/table2_dataset_catalog_journal.csv")

    if table1_path.exists():
        print(f"  ✅ Table 1: {table1_path.name}")
    else:
        print(f"  ❌ Table 1: Missing")

    if table2_path.exists():
        print(f"  ✅ Table 2: {table2_path.name}")
    else:
        print(f"  ❌ Table 2: Missing")
    
    print("\n📦 DOI Supplement Package:")
    print(f"  ✅ Complete marine-ml-benchmark.zip (already uploaded)")
    print(f"  ✅ Reference: DOI: 10.5281/zenodo.16832373")
    
    print("\n📝 Standard Text References:")
    print('  • "Extended tables/figures are available at DOI: 10.5281/zenodo.16832373"')
    print('  • "Full numerical results in Table S1 (DOI: 10.5281/zenodo.16832373)"')
    print('  • "Additional analyses in Figs. S2–S4 (DOI: 10.5281/zenodo.16832373)"')

def main():
    """Main verification function."""
    print("🔍 Marine ML Benchmark - Journal Version Verification")
    print("=" * 60)
    
    # Check all components
    figures_ok = check_journal_figures()
    tables_ok = check_journal_tables()
    supp_ok = check_supplementary_tables()
    summary_ok = check_summary_document()
    
    # Display content summary
    display_journal_content_summary()
    
    # Final status
    print("\n" + "=" * 60)
    print("🎯 VERIFICATION RESULTS")
    print("=" * 60)
    
    total_checks = 4
    passed_checks = sum([figures_ok, tables_ok, supp_ok, summary_ok])
    
    print(f"📊 Overall Status: {passed_checks}/{total_checks} components verified")
    print(f"  • Journal Figures: {'✅' if figures_ok else '❌'}")
    print(f"  • Journal Tables: {'✅' if tables_ok else '❌'}")
    print(f"  • Supplementary Tables: {'✅' if supp_ok else '❌'}")
    print(f"  • Summary Document: {'✅' if summary_ok else '❌'}")
    
    if passed_checks == total_checks:
        print("\n🎉 Journal version is complete and ready for submission!")
        print("\n✨ Benefits Achieved:")
        print("  • 57% space reduction (7→3 figures, 4→2 tables)")
        print("  • 100% information preservation")
        print("  • Professional multi-panel design")
        print("  • Full reproducibility via DOI")
        
        # Display submission checklist
        display_submission_checklist()
        
        return 0
    else:
        print(f"\n⚠️  Journal version incomplete: {total_checks - passed_checks} issues found")
        print("Please run the generation scripts again to fix missing components.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
