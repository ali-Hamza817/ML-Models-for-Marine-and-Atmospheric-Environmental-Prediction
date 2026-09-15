#!/usr/bin/env python3
"""
Paper Figure Generation Module
Generates all 7 figures for the chlorophyll prediction benchmark paper
Integrated into src/ directory for reproducibility
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import FancyBboxPatch, Rectangle
import matplotlib.patches as mpatches
import warnings
import yaml
import os
import sys
from pathlib import Path

warnings.filterwarnings('ignore')

# Add src directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def load_config():
    """Load configuration from config.yaml"""
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def setup_matplotlib(config):
    """Setup matplotlib parameters according to config"""
    paper_config = config.get('paper', {})
    fig_config = paper_config.get('figures', {})
    
    plt.rcParams.update({
        'font.family': fig_config.get('font_family', 'serif'),
        'font.serif': fig_config.get('font_serif', ['Times New Roman']),
        'font.size': fig_config.get('base_font_size', 8),
        'figure.dpi': fig_config.get('dpi', 300),
        'savefig.dpi': fig_config.get('dpi', 300),
        'savefig.format': 'pdf',
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        'savefig.facecolor': 'white'
    })

# Color scheme for colorblind accessibility
COLORBLIND_COLORS = {
    'red': '#d73027',
    'green': '#1a9850', 
    'orange': '#fee08b',
    'blue': '#4575b4',
    'purple': '#762a83',
    'brown': '#8c510a',
    'pink': '#c51b7d',
    'gray': '#999999'
}

def load_paper_data():
    """Load all data required for paper figures"""
    print("📊 Loading paper data...")
    
    data = {}
    base_path = Path(__file__).parent.parent
    
    # Load final tables
    tables_dir = base_path / "tables"
    try:
        data['table1'] = pd.read_csv(tables_dir / 'final_table1_dataset_characteristics.csv')
        data['table2'] = pd.read_csv(tables_dir / 'final_table2_model_performance.csv')
        data['table3'] = pd.read_csv(tables_dir / 'final_table3_best_performance.csv')
        data['table4'] = pd.read_csv(tables_dir / 'final_table4_validation_summary.csv')
        print("   ✅ Final tables loaded successfully")
    except Exception as e:
        print(f"   ❌ Error loading tables: {e}")
        return None
    
    # Load detailed results if available
    try:
        data['detailed_results'] = pd.read_csv(tables_dir / 'supplementary_table_s2_full_results.csv')
        print("   ✅ Detailed results loaded successfully")
    except Exception as e:
        print(f"   ⚠️ Detailed results not found: {e}")
        data['detailed_results'] = None
    
    return data

def create_figure1_dataset_overview(data, output_dir):
    """Figure 1: Dataset Overview and Characteristics"""
    print("📊 Generating Figure 1: Dataset Overview")
    
    table1 = data['table1']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Left panel: Sample sizes (log scale)
    datasets = table1['Dataset'].tolist()
    samples = table1['Samples'].tolist()
    
    # Convert string numbers to integers
    samples = [int(str(s).replace(',', '')) for s in samples]
    
    bars1 = ax1.bar(range(len(datasets)), samples, color=COLORBLIND_COLORS['blue'], alpha=0.8)
    ax1.set_yscale('log')
    ax1.set_xlabel('Dataset', fontsize=18)
    ax1.set_ylabel('Sample Count (log scale)', fontsize=18)
    ax1.set_xticks(range(len(datasets)))
    ax1.set_xticklabels(datasets, rotation=45, ha='right', fontsize=16)
    ax1.tick_params(axis='y', labelsize=16)
    
    # Add value labels
    for i, (bar, sample) in enumerate(zip(bars1, samples)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height * 1.1,
                f'{sample:,}', ha='center', va='bottom', fontsize=12)
    
    # Right panel: Variable counts
    variables = table1['Variables'].tolist()
    bars2 = ax2.bar(range(len(datasets)), variables, color=COLORBLIND_COLORS['green'], alpha=0.8)
    ax2.set_xlabel('Dataset', fontsize=18)
    ax2.set_ylabel('Number of Variables', fontsize=18)
    ax2.set_xticks(range(len(datasets)))
    ax2.set_xticklabels(datasets, rotation=45, ha='right', fontsize=16)
    ax2.tick_params(axis='y', labelsize=16)
    
    # Add value labels
    for i, (bar, var) in enumerate(zip(bars2, variables)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{var}', ha='center', va='bottom', fontsize=12)
    
    plt.tight_layout()
    
    # Save in multiple formats
    formats = ['pdf', 'png']
    for fmt in formats:
        output_path = output_dir / f'figure1_dataset_overview_final.{fmt}'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    plt.close()
    print("   ✅ Figure 1 generated")

def create_figure2_performance_heatmap(data, output_dir):
    """Figure 2: Cross-dataset Model Performance Heatmap"""
    print("📊 Generating Figure 2: Performance Heatmap")
    
    table2 = data['table2']
    
    # Get all models and datasets
    all_models = sorted(table2['Model'].unique())
    all_datasets = sorted(table2['Dataset'].unique())
    
    # Create performance matrix
    performance_matrix = np.full((len(all_datasets), len(all_models)), np.nan)
    
    for i, dataset in enumerate(all_datasets):
        for j, model in enumerate(all_models):
            model_data = table2[(table2['Dataset'] == dataset) & (table2['Model'] == model)]
            if not model_data.empty:
                r2_str = model_data['R²'].iloc[0]
                try:
                    performance_matrix[i, j] = float(r2_str)
                except:
                    performance_matrix[i, j] = np.nan
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    
    heatmap = sns.heatmap(performance_matrix,
                         annot=True,
                         fmt='.3f',
                         cmap='coolwarm',
                         center=0,
                         vmin=-1.0,
                         vmax=0.9,
                         cbar_kws={'label': 'R² Score'},
                         ax=ax,
                         annot_kws={'fontsize': 14},
                         xticklabels=all_models,
                         yticklabels=all_datasets)
    
    # Set font sizes
    ax.set_xlabel('Model', fontsize=18)
    ax.set_ylabel('Dataset', fontsize=18)
    ax.tick_params(axis='x', labelsize=16, rotation=45)
    ax.tick_params(axis='y', labelsize=16, rotation=0)
    
    # Colorbar font size
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=14)
    
    plt.tight_layout()
    
    # Save in multiple formats
    formats = ['pdf', 'png']
    for fmt in formats:
        output_path = output_dir / f'figure2_performance_heatmap_final.{fmt}'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    plt.close()
    print("   ✅ Figure 2 generated")

def generate_all_figures(config=None):
    """Generate all 7 figures for the paper"""
    if config is None:
        config = load_config()
    
    setup_matplotlib(config)
    
    # Load data
    data = load_paper_data()
    if data is None:
        print("❌ Failed to load data")
        return False
    
    # Setup output directory
    base_path = Path(__file__).parent.parent
    output_dir = base_path / config['paper']['figures_dir'].lstrip('../')
    output_dir.mkdir(exist_ok=True)
    
    print(f"📁 Output directory: {output_dir}")
    
    # Generate figures
    try:
        create_figure1_dataset_overview(data, output_dir)
        create_figure2_performance_heatmap(data, output_dir)
        
        # Import and call remaining figure functions from main script
        sys.path.append(str(base_path))
        from generate_correct_7_figures import (
            create_figure3_performance_boxplots,
            create_figure4_model_robustness,
            create_figure5_difficulty_vs_size,
            create_figure6_feature_importance,
            create_figure7_technical_roadmap
        )
        
        create_figure3_performance_boxplots(data)
        create_figure4_model_robustness(data)
        create_figure5_difficulty_vs_size(data)
        create_figure6_feature_importance(data)
        create_figure7_technical_roadmap(data)
        
        print("🎉 All figures generated successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error generating figures: {e}")
        return False

if __name__ == "__main__":
    generate_all_figures()
