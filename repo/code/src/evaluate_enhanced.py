import numpy as np, pandas as pd, joblib, torch
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, precision_recall_fscore_support
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from src.utils_io import read_cfg
import warnings
warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[1]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 重新定义深度学习模型类（用于加载）
import torch.nn as nn

class DeepModelWrapper:
    """深度学习模型包装器，用于评估"""
    def __init__(self, model, scaler_X, scaler_y, device=DEVICE):
        self.model = model.to(device)
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y
        self.device = device

    def predict(self, X):
        """预测方法，兼容sklearn接口"""
        self.model.eval()
        X_scaled = self.scaler_X.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        with torch.no_grad():
            outputs = self.model(X_tensor).squeeze()
            predictions = self.scaler_y.inverse_transform(outputs.detach().cpu().numpy().reshape(-1, 1)).flatten()
        return predictions

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

class TransformerModel(nn.Module):
    def __init__(self, input_size, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super(TransformerModel, self).__init__()

        # 降维处理：先将高维特征降到合理范围
        self.feature_reduction = nn.Linear(input_size, min(32, input_size//2))
        self.input_projection = nn.Linear(min(32, input_size//2), d_model)

        # 简化位置编码
        self.pos_encoding = nn.Parameter(torch.randn(100, d_model) * 0.1)

        # 使用更小的Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model*2,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 输出层
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model//2, 1)
        )

        # 权重初始化
        self._init_weights()

    def _init_weights(self):
        """改进的权重初始化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        # 特征降维
        x = self.feature_reduction(x)
        x = torch.relu(x)

        # 输入投影
        x = self.input_projection(x)

        # 位置编码 - 确保在正确设备上
        pos_enc = self.pos_encoding[:seq_len, :].unsqueeze(0).expand(batch_size, -1, -1)
        x = x + pos_enc

        # Transformer编码
        x = self.transformer(x)

        # 全局平均池化 + 层归一化
        x = self.layer_norm(x.mean(dim=1))
        x = self.dropout(x)

        # 输出
        x = self.fc(x)
        return x

class TransformerModelOld(nn.Module):
    """旧版本的Transformer模型（兼容性）"""
    def __init__(self, input_size, d_model=128, nhead=8, num_layers=3, dropout=0.1):
        super(TransformerModelOld, self).__init__()
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1000, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        seq_len = x.size(1)
        x = self.input_projection(x)
        x = x + self.pos_encoding[:seq_len, :].unsqueeze(0)
        x = self.transformer(x)
        x = self.dropout(x.mean(dim=1))
        x = self.fc(x)
        return x

class DeepModelWrapper:
    def __init__(self, model, scaler_X, scaler_y, device=DEVICE):
        self.model = model.to(device)
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y
        self.device = device
        
    def predict(self, X):
        self.model.eval()
        X_scaled = self.scaler_X.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        with torch.no_grad():
            outputs = self.model(X_tensor).squeeze()
            predictions = self.scaler_y.inverse_transform(outputs.detach().cpu().numpy().reshape(-1, 1)).flatten()
        return predictions

def bootstrap_metrics(y_true, y_pred, n_bootstrap=1000, confidence_level=0.95):
    """计算Bootstrap置信区间"""
    n_samples = len(y_true)
    bootstrap_scores = {'R2': [], 'MAE': [], 'RMSE': []}
    
    np.random.seed(42)
    for _ in range(n_bootstrap):
        # 重采样
        indices = np.random.choice(n_samples, n_samples, replace=True)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]
        
        # 计算指标
        bootstrap_scores['R2'].append(r2_score(y_true_boot, y_pred_boot))
        bootstrap_scores['MAE'].append(mean_absolute_error(y_true_boot, y_pred_boot))
        bootstrap_scores['RMSE'].append(mean_squared_error(y_true_boot, y_pred_boot, squared=False))
    
    # 计算置信区间
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    results = {}
    for metric, scores in bootstrap_scores.items():
        scores = np.array(scores)
        results[f'{metric}_mean'] = np.mean(scores)
        results[f'{metric}_std'] = np.std(scores)
        results[f'{metric}_ci_lower'] = np.percentile(scores, lower_percentile)
        results[f'{metric}_ci_upper'] = np.percentile(scores, upper_percentile)
    
    return results


def nrmse(y_true, y_pred):
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    denom = np.max(y_true) - np.min(y_true)
    if denom == 0:
        return np.nan
    return rmse / denom


def nse(y_true, y_pred):
    denom = np.sum((y_true - np.mean(y_true)) ** 2)
    if denom == 0:
        return np.nan
    return 1 - (np.sum((y_true - y_pred) ** 2) / denom)


def event_metrics(y_true, y_pred, threshold=None):
    if threshold is None:
        threshold = np.percentile(y_true, 90)
    y_true_bin = (y_true >= threshold).astype(int)
    y_pred_bin = (y_pred >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_bin, y_pred_bin, average="binary", zero_division=0
    )
    return {
        "event_threshold": float(threshold),
        "event_precision": float(precision),
        "event_recall": float(recall),
        "event_f1": float(f1),
    }

def load_deep_model(model_path, model_type, input_size=None):
    """加载深度学习模型"""
    checkpoint = torch.load(model_path, map_location='cpu')

    # 过滤掉不需要的参数（如learning_rate）
    model_params = checkpoint['model_params'].copy()
    model_params.pop('learning_rate', None)  # 移除learning_rate参数

    # 从checkpoint中获取正确的input_size
    if 'input_size' in checkpoint:
        actual_input_size = checkpoint['input_size']
    else:
        # 从模型权重中推断input_size
        state_dict = checkpoint['model_state_dict']
        if model_type == "lstm":
            # LSTM的第一层权重形状是 [hidden_size*4, input_size]
            first_weight = state_dict['lstm.weight_ih_l0']
            actual_input_size = first_weight.shape[1]
        elif model_type == "transformer":
            # Transformer有两种版本：新版本有feature_reduction，旧版本只有input_projection
            if 'feature_reduction.weight' in state_dict:
                # 新版本：从feature_reduction层获取input_size
                first_weight = state_dict['feature_reduction.weight']
                actual_input_size = first_weight.shape[1]
            elif 'input_projection.weight' in state_dict:
                # 旧版本：从input_projection层获取input_size
                first_weight = state_dict['input_projection.weight']
                actual_input_size = first_weight.shape[1]
            else:
                raise ValueError("无法从Transformer模型中确定input_size")
        else:
            actual_input_size = input_size or 70  # 默认值

    if model_type == "lstm":
        model = LSTMModel(input_size=actual_input_size, **model_params)
    elif model_type == "transformer":
        # 根据模型结构选择正确的Transformer类
        state_dict = checkpoint['model_state_dict']
        if 'feature_reduction.weight' in state_dict:
            # 新版本Transformer
            model = TransformerModel(input_size=actual_input_size, **model_params)
        else:
            # 旧版本Transformer
            model = TransformerModelOld(input_size=actual_input_size, **model_params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.load_state_dict(checkpoint['model_state_dict'])

    return DeepModelWrapper(model, checkpoint['scaler_X'], checkpoint['scaler_y'])

def evaluate_dataset(ds_key, cfg):
    """评估单个数据集的所有模型"""
    print(f"评估数据集: {ds_key}")
    
    # 检查目标列
    target_col = cfg["datasets"][ds_key].get("target_col")
    if not target_col:
        print(f"  跳过 {ds_key}: 无目标列")
        return pd.DataFrame()
    
    # 加载数据
    data_path = ROOT / "data_proc" / ds_key / "clean.csv"
    if not data_path.exists():
        print(f"  跳过 {ds_key}: 数据文件不存在")
        return pd.DataFrame()
    
    df = pd.read_csv(data_path)
    if target_col not in df.columns:
        print(f"  跳过 {ds_key}: 目标列不存在")
        return pd.DataFrame()
    
    # 准备传统ML数据
    X_tabular = df.drop(columns=[target_col]).select_dtypes(include=[np.number]).values
    y_true = df[target_col].values
    
    # 准备序列数据（如果存在）
    seq_path = ROOT / "data_proc" / ds_key / "sequences.npz"
    X_sequence, y_sequence = None, None
    if seq_path.exists():
        try:
            data = np.load(seq_path)
            X_sequence, y_sequence = data['X'], data['y']
        except:
            pass
    
    results = []
    model_dir = ROOT / "models" / ds_key
    
    if not model_dir.exists():
        print(f"  跳过 {ds_key}: 模型目录不存在")
        return pd.DataFrame()
    
    # 评估所有模型
    for model_file in model_dir.glob("*"):
        if model_file.suffix == ".pkl":
            # 传统机器学习模型
            model_name = model_file.stem
            try:
                model = joblib.load(model_file)
                y_pred = model.predict(X_tabular)
                
                # 基础指标
                base_metrics = {
                    "dataset": ds_key,
                    "model": model_name,
                    "R2": r2_score(y_true, y_pred),
                    "MAE": mean_absolute_error(y_true, y_pred),
                    "RMSE": mean_squared_error(y_true, y_pred, squared=False),
                    "NRMSE": nrmse(y_true, y_pred),
                    "NSE": nse(y_true, y_pred),
                    "n_samples": len(y_true)
                }

                if ds_key == "biotoxin":
                    base_metrics.update(event_metrics(y_true, y_pred))
                
                # Bootstrap置信区间
                bootstrap_results = bootstrap_metrics(y_true, y_pred, 
                                                    n_bootstrap=cfg["global"].get("bootstrap_samples", 1000),
                                                    confidence_level=cfg["global"].get("confidence_level", 0.95))
                
                # 合并结果
                result = {**base_metrics, **bootstrap_results}
                results.append(result)
                print(f"  {model_name}: R² = {result['R2']:.4f} ± {result['R2_std']:.4f}")
                
            except Exception as e:
                print(f"  评估失败 {model_name}: {e}")
        
        elif model_file.suffix == ".pth":
            # 深度学习模型
            model_name = model_file.stem
            if X_sequence is None or y_sequence is None:
                print(f"  跳过 {model_name}: 无序列数据")
                continue
            
            try:
                model = load_deep_model(model_file, model_name)
                y_pred = model.predict(X_sequence)
                
                # 基础指标
                base_metrics = {
                    "dataset": ds_key,
                    "model": model_name,
                    "R2": r2_score(y_sequence, y_pred),
                    "MAE": mean_absolute_error(y_sequence, y_pred),
                    "RMSE": mean_squared_error(y_sequence, y_pred, squared=False),
                    "NRMSE": nrmse(y_sequence, y_pred),
                    "NSE": nse(y_sequence, y_pred),
                    "n_samples": len(y_sequence)
                }

                if ds_key == "biotoxin":
                    base_metrics.update(event_metrics(y_sequence, y_pred))
                
                # Bootstrap置信区间
                bootstrap_results = bootstrap_metrics(y_sequence, y_pred,
                                                    n_bootstrap=cfg["global"].get("bootstrap_samples", 1000),
                                                    confidence_level=cfg["global"].get("confidence_level", 0.95))
                
                # 合并结果
                result = {**base_metrics, **bootstrap_results}
                results.append(result)
                print(f"  {model_name}: R² = {result['R2']:.4f} ± {result['R2_std']:.4f}")
                
            except Exception as e:
                print(f"  评估失败 {model_name}: {e}")
    
    return pd.DataFrame(results)

def generate_summary_stats(results_df):
    """生成汇总统计"""
    if results_df.empty:
        return pd.DataFrame()
    
    # 按模型汇总
    model_summary = results_df.groupby('model').agg({
        'R2': ['mean', 'std', 'min', 'max'],
        'MAE': ['mean', 'std', 'min', 'max'],
        'RMSE': ['mean', 'std', 'min', 'max'],
        'n_samples': 'sum'
    }).round(4)
    
    # 按数据集汇总
    dataset_summary = results_df.groupby('dataset').agg({
        'R2': ['mean', 'std', 'min', 'max'],
        'MAE': ['mean', 'std', 'min', 'max'],
        'RMSE': ['mean', 'std', 'min', 'max'],
        'n_samples': 'first'
    }).round(4)
    
    return model_summary, dataset_summary

if __name__ == "__main__":
    cfg = read_cfg()
    
    print("开始模型评估...")
    all_results = []
    
    # 评估所有数据集
    for dataset in cfg["datasets"].keys():
        try:
            results = evaluate_dataset(dataset, cfg)
            if not results.empty:
                all_results.append(results)
        except Exception as e:
            print(f"评估数据集 {dataset} 失败: {e}")
    
    if all_results:
        # 合并所有结果
        final_results = pd.concat(all_results, ignore_index=True)
        
        # 保存详细结果
        final_results.to_csv(ROOT / "tables" / "evaluation_detailed.csv", index=False)
        
        # 生成汇总统计
        model_summary, dataset_summary = generate_summary_stats(final_results)
        
        # 保存汇总结果
        model_summary.to_csv(ROOT / "tables" / "model_summary.csv")
        dataset_summary.to_csv(ROOT / "tables" / "dataset_summary.csv")
        
        # 保存简化版本（向后兼容）
        simple_results = final_results[['dataset', 'model', 'R2', 'MAE', 'RMSE']].copy()
        simple_results.to_csv(ROOT / "tables" / "perf_summary.csv", index=False)
        
        print(f"\n✓ 评估完成")
        print(f"详细结果: tables/evaluation_detailed.csv")
        print(f"模型汇总: tables/model_summary.csv")
        print(f"数据集汇总: tables/dataset_summary.csv")
        print(f"简化结果: tables/perf_summary.csv")
        
        # 显示最佳模型
        best_models = final_results.loc[final_results.groupby('dataset')['R2'].idxmax()]
        print("\n最佳模型 (按数据集):")
        for _, row in best_models.iterrows():
            print(f"  {row['dataset']}: {row['model']} (R² = {row['R2']:.4f})")
    
    else:
        print("没有找到评估结果")
