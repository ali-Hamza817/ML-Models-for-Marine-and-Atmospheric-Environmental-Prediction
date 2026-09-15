import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import joblib
import shap
import warnings
warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = ROOT / "figures"
TABLES_DIR = ROOT / "tables"
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data_proc"

# 确保输出目录存在
FIGURES_DIR.mkdir(exist_ok=True)

def set_style():
    """设置统一的可视化风格"""
    sns.set_theme(style="whitegrid")
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.titlesize'] = 16

def plot_boxplot(save_path=None):
    """绘制模型性能箱线图"""
    print("绘制性能箱线图...")
    
    # 加载性能数据
    perf_path = TABLES_DIR / "perf_summary.csv"
    if not perf_path.exists():
        print("  性能数据不存在，跳过")
        return
    
    df = pd.read_csv(perf_path)
    
    # 创建图形
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    
    # 按数据集绘制R2箱线图
    sns.boxplot(x="dataset", y="R2", data=df, ax=axes[0])
    axes[0].set_title("R² by Dataset")
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha='right')
    
    # 按数据集绘制RMSE箱线图
    sns.boxplot(x="dataset", y="RMSE", data=df, ax=axes[1])
    axes[1].set_title("RMSE by Dataset")
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right')
    
    # 按模型绘制R2箱线图
    sns.boxplot(x="model", y="R2", data=df, ax=axes[2])
    axes[2].set_title("R² by Model")
    
    plt.tight_layout()
    
    # 保存图形
    if save_path is None:
        save_path = FIGURES_DIR / "fig_performance_boxplot.svg"
    plt.savefig(save_path)
    plt.savefig(save_path.with_suffix('.png'), dpi=300)
    print(f"  已保存至 {save_path}")
    
    return fig

def plot_radar(dataset="cleaned_data", model="xgb", top_n=10, save_path=None):
    """绘制SHAP重要性雷达图"""
    print(f"绘制SHAP雷达图 ({dataset}, {model})...")
    
    # 检查模型文件
    model_path = MODELS_DIR / dataset / f"{model}.pkl"
    if not model_path.exists():
        print(f"  模型文件不存在: {model_path}")
        return
    
    # 检查数据文件
    data_path = DATA_DIR / dataset / "clean.csv"
    if not data_path.exists():
        print(f"  数据文件不存在: {data_path}")
        return
    
    try:
        # 加载模型和数据
        mdl = joblib.load(model_path)
        df = pd.read_csv(data_path)
        
        # 确定目标列
        target_cols = [col for col in df.columns if any(kw in col.lower() 
                      for kw in ['g2chla', 'target', 'chlorophyll'])]
        if not target_cols:
            print("  无法确定目标列")
            return
        
        tcol = target_cols[0]
        X = df.drop(columns=[tcol]).select_dtypes(include=[np.number])
        
        # 计算SHAP值
        if hasattr(mdl, 'named_steps') and model in ['rf', 'xgb']:
            explainer = shap.TreeExplainer(mdl.named_steps[model])
            shap_values = explainer.shap_values(X)
            
            if isinstance(shap_values, list):
                shap_values = shap_values[0]  # 对于多输出模型
                
            mean_abs = np.abs(shap_values).mean(0)
        else:
            print("  不支持的模型类型，跳过")
            return
        
        # 选择Top-N特征
        idx = np.argsort(mean_abs)[-top_n:]
        labels = X.columns[idx]
        values = mean_abs[idx]
        
        # 绘制雷达图
        angles = np.linspace(0, 2*np.pi, top_n, endpoint=False)
        values = np.concatenate((values, [values[0]]))
        angles = np.concatenate((angles, [angles[0]]))
        
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, polar=True)
        ax.plot(angles, values, linewidth=2)
        ax.fill(angles, values, alpha=0.3)
        ax.set_thetagrids(angles[:-1] * 180/np.pi, labels, fontsize=10)
        ax.set_title(f"Top-{top_n} SHAP Importance ({dataset}, {model})")
        plt.tight_layout()
        
        # 保存图形
        if save_path is None:
            save_path = FIGURES_DIR / f"fig_radar_{dataset}_{model}.svg"
        plt.savefig(save_path)
        plt.savefig(save_path.with_suffix('.png'), dpi=300)
        print(f"  已保存至 {save_path}")
        
        return fig
    
    except Exception as e:
        print(f"  绘制雷达图失败: {e}")
        return None

def plot_shap_summary(dataset="cleaned_data", model="xgb", save_path=None):
    """绘制SHAP摘要图"""
    print(f"绘制SHAP摘要图 ({dataset}, {model})...")
    
    # 检查模型文件
    model_path = MODELS_DIR / dataset / f"{model}.pkl"
    if not model_path.exists():
        print(f"  模型文件不存在: {model_path}")
        return
    
    # 检查数据文件
    data_path = DATA_DIR / dataset / "clean.csv"
    if not data_path.exists():
        print(f"  数据文件不存在: {data_path}")
        return
    
    try:
        # 加载模型和数据
        mdl = joblib.load(model_path)
        df = pd.read_csv(data_path)
        
        # 确定目标列
        target_cols = [col for col in df.columns if any(kw in col.lower() 
                      for kw in ['g2chla', 'target', 'chlorophyll'])]
        if not target_cols:
            print("  无法确定目标列")
            return
        
        tcol = target_cols[0]
        X = df.drop(columns=[tcol]).select_dtypes(include=[np.number])
        
        # 计算SHAP值
        if hasattr(mdl, 'named_steps') and model in ['rf', 'xgb']:
            explainer = shap.TreeExplainer(mdl.named_steps[model])
            shap_values = explainer.shap_values(X)
            
            if isinstance(shap_values, list):
                shap_values = shap_values[0]  # 对于多输出模型
            
            # 绘制SHAP摘要图
            plt.figure(figsize=(10, 12))
            shap.summary_plot(shap_values, X, show=False)
            
            # 保存图形
            if save_path is None:
                save_path = FIGURES_DIR / f"fig_shap_summary_{dataset}_{model}.svg"
            plt.tight_layout()
            plt.savefig(save_path)
            plt.savefig(save_path.with_suffix('.png'), dpi=300)
            print(f"  已保存至 {save_path}")
            
            return plt.gcf()
        else:
            print("  不支持的模型类型，跳过")
            return None
    
    except Exception as e:
        print(f"  绘制SHAP摘要图失败: {e}")
        return None

def plot_model_comparison(save_path=None):
    """绘制模型比较图"""
    print("绘制模型比较图...")
    
    # 加载性能数据
    perf_path = TABLES_DIR / "perf_summary.csv"
    if not perf_path.exists():
        print("  性能数据不存在，跳过")
        return
    
    df = pd.read_csv(perf_path)
    
    # 计算每个模型在各数据集上的平均性能
    model_perf = df.pivot_table(
        index='model', 
        columns='dataset', 
        values='R2',
        aggfunc='mean'
    ).round(3)
    
    # 绘制热力图
    plt.figure(figsize=(12, 8))
    sns.heatmap(model_perf, annot=True, cmap='YlGnBu', linewidths=0.5)
    plt.title('Model Performance (R²) Across Datasets')
    plt.tight_layout()
    
    # 保存图形
    if save_path is None:
        save_path = FIGURES_DIR / "fig_model_comparison.svg"
    plt.savefig(save_path)
    plt.savefig(save_path.with_suffix('.png'), dpi=300)
    print(f"  已保存至 {save_path}")
    
    return plt.gcf()

def plot_interactive_comparison():
    """创建交互式模型比较图"""
    print("创建交互式模型比较图...")
    
    # 加载性能数据
    perf_path = TABLES_DIR / "perf_summary.csv"
    if not perf_path.exists():
        print("  性能数据不存在，跳过")
        return
    
    df = pd.read_csv(perf_path)
    
    # 创建交互式散点图
    fig = px.scatter(
        df, x="MAE", y="R2", color="model", 
        size="RMSE", hover_name="dataset",
        title="Model Performance Comparison",
        labels={"R2": "R² Score", "MAE": "Mean Absolute Error"},
        size_max=20
    )
    
    # 添加趋势线
    fig.update_layout(
        xaxis_title="MAE (lower is better)",
        yaxis_title="R² (higher is better)",
        legend_title="Model",
        height=600,
        width=900
    )
    
    # 保存为HTML
    save_path = FIGURES_DIR / "interactive_model_comparison.html"
    fig.write_html(save_path)
    print(f"  已保存至 {save_path}")
    
    return fig

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="生成可视化图表")
    parser.add_argument("--all", action="store_true", help="生成所有图表")
    parser.add_argument("--boxplot", action="store_true", help="生成性能箱线图")
    parser.add_argument("--radar", action="store_true", help="生成SHAP雷达图")
    parser.add_argument("--shap", action="store_true", help="生成SHAP摘要图")
    parser.add_argument("--compare", action="store_true", help="生成模型比较图")
    parser.add_argument("--interactive", action="store_true", help="生成交互式比较图")
    parser.add_argument("--dataset", default="cleaned_data", help="用于SHAP分析的数据集")
    parser.add_argument("--model", default="xgb", help="用于SHAP分析的模型")
    
    args = parser.parse_args()
    
    # 设置统一风格
    set_style()
    
    # 如果没有指定任何参数，默认生成所有图表
    if not any([args.all, args.boxplot, args.radar, args.shap, args.compare, args.interactive]):
        args.all = True
    
    if args.all or args.boxplot:
        plot_boxplot()
    
    if args.all or args.radar:
        plot_radar(dataset=args.dataset, model=args.model)
    
    if args.all or args.shap:
        plot_shap_summary(dataset=args.dataset, model=args.model)
    
    if args.all or args.compare:
        plot_model_comparison()
    
    if args.all or args.interactive:
        plot_interactive_comparison()
    
    print("可视化完成!")
