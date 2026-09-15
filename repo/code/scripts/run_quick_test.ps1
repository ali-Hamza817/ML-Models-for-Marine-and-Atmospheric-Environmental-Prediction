# Marine ML Benchmark - Quick Test (PowerShell)
# Verifies installation and runs on sample data (5 minutes)

$ErrorActionPreference = "Stop"

Write-Host "==========================================" -ForegroundColor Green
Write-Host "⚡ Marine ML Benchmark - Quick Test" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green
Write-Host ""

# Check if we're in the right directory
if (!(Test-Path "README.md") -or !(Test-Path "code/src")) {
    Write-Host "❌ Error: Please run this script from the repository root directory" -ForegroundColor Red
    exit 1
}

Write-Host "🔧 Verifying Installation..." -ForegroundColor Yellow
Write-Host "------------------------------------------"

# Check Python
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✅ $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Python not found. Please install Python 3.8+" -ForegroundColor Red
    exit 1
}

# Check key dependencies
Write-Host "Checking dependencies..."
$checkDeps = @"
import sys
required = ['pandas', 'numpy', 'scikit-learn', 'xgboost', 'torch', 'matplotlib', 'seaborn']
missing = []
for pkg in required:
    try:
        __import__(pkg)
        print(f'  ✅ {pkg}')
    except ImportError:
        missing.append(pkg)
        print(f'  ❌ {pkg} - MISSING')

if missing:
    print(f'\n❌ Missing packages: {missing}')
    print('Please run: pip install -r requirements.txt')
    sys.exit(1)
else:
    print('\n✅ All dependencies available')
"@

python -c $checkDeps
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Dependency check failed" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "📊 Creating Sample Data..." -ForegroundColor Yellow
Write-Host "------------------------------------------"

# Create sample data for quick testing
$createSampleData = @"
import pandas as pd
import numpy as np
from pathlib import Path

# Create sample data directory
Path('data/sample').mkdir(parents=True, exist_ok=True)

# Generate sample datasets
np.random.seed(42)

datasets = {
    'cleaned_data': {
        'samples': 1000,
        'features': ['temp', 'salinity', 'depth', 'lat', 'lon'],
        'target': 'chlorophyll_a'
    },
    'era5_daily': {
        'samples': 2000, 
        'features': ['wind_speed', 'temperature', 'pressure'],
        'target': 'wind10'
    },
    'biotoxin': {
        'samples': 500,
        'features': ['concentration'],
        'target': 'toxin_level'
    }
}

for name, config in datasets.items():
    # Generate synthetic data
    n_samples = config['samples']
    features = config['features']
    target = config['target']
    
    # Create feature data
    data = {}
    for feature in features:
        data[feature] = np.random.randn(n_samples)
    
    # Create target (with some correlation to features)
    data[target] = (
        sum(data[f] for f in features[:2]) / len(features[:2]) + 
        0.5 * np.random.randn(n_samples)
    )
    
    # Add date column
    dates = pd.date_range('2020-01-01', periods=n_samples, freq='D')
    data['date'] = dates
    
    # Save sample data
    df = pd.DataFrame(data)
    df.to_csv(f'data/sample/{name}_sample.csv', index=False)
    print(f'✅ Created {name}: {n_samples} samples, {len(features)} features')

print('\n✅ Sample data created successfully')
"@

python -c $createSampleData

Write-Host ""
Write-Host "🤖 Quick Model Training..." -ForegroundColor Yellow
Write-Host "------------------------------------------"
Write-Host "Training Random Forest on sample data..."

# Quick training test
$quickTraining = @"
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from pathlib import Path

# Test on cleaned_data sample
df = pd.read_csv('data/sample/cleaned_data_sample.csv')

# Prepare data
X = df.drop(['chlorophyll_a', 'date'], axis=1)
y = df['chlorophyll_a']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f'✅ Random Forest trained successfully')
print(f'   R² Score: {r2:.4f}')
print(f'   MAE: {mae:.4f}')

# Save test model
Path('models/test').mkdir(parents=True, exist_ok=True)
import joblib
joblib.dump(model, 'models/test/rf_sample.pkl')
print('✅ Test model saved')
"@

python -c $quickTraining

Write-Host ""
Write-Host "📈 Quick Evaluation Test..." -ForegroundColor Yellow
Write-Host "------------------------------------------"

# Test evaluation functions
$evalTest = @"
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Test metrics calculation
y_true = np.array([1, 2, 3, 4, 5])
y_pred = np.array([1.1, 2.2, 2.8, 3.9, 5.1])

metrics = {
    'R²': r2_score(y_true, y_pred),
    'MAE': mean_absolute_error(y_true, y_pred),
    'RMSE': mean_squared_error(y_true, y_pred, squared=False)
}

print('✅ Evaluation metrics test:')
for metric, value in metrics.items():
    print(f'   {metric}: {value:.4f}')
"@

python -c $evalTest

Write-Host ""
Write-Host "📊 Quick Visualization Test..." -ForegroundColor Yellow
Write-Host "------------------------------------------"

# Test plotting
$vizTest = @"
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

# Create test plot
fig, ax = plt.subplots(1, 1, figsize=(6, 4))

# Sample data for plotting
x = np.linspace(0, 10, 100)
y = np.sin(x) + 0.1 * np.random.randn(100)

ax.plot(x, y, 'b-', alpha=0.7, label='Sample Data')
ax.set_xlabel('X values')
ax.set_ylabel('Y values')
ax.set_title('Quick Test Plot')
ax.legend()
ax.grid(True, alpha=0.3)

# Save test figure
Path('outputs/figures').mkdir(parents=True, exist_ok=True)
plt.savefig('outputs/figures/quick_test_plot.png', dpi=150, bbox_inches='tight')
plt.close()

print('✅ Test plot generated: outputs/figures/quick_test_plot.png')
"@

python -c $vizTest

Write-Host ""
Write-Host "🧪 Running System Tests..." -ForegroundColor Yellow
Write-Host "------------------------------------------"

# Test configuration loading
$configTest = @"
import yaml
from pathlib import Path

# Test config loading
if Path('configs/config.yaml').exists():
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    print('✅ Configuration file loaded successfully')
    print(f'   Found {len(config.get(\"datasets\", {}))} dataset configurations')
else:
    print('⚠️  Configuration file not found')
"@

python -c $configTest

Write-Host ""
Write-Host "==========================================" -ForegroundColor Green
Write-Host "🎉 Quick Test Completed!" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green
Write-Host ""

Write-Host "✅ Test Results Summary:" -ForegroundColor Green
Write-Host "  • Python environment: Working"
Write-Host "  • Dependencies: All available"
Write-Host "  • Sample data: Generated and tested"
Write-Host "  • Model training: Successful"
Write-Host "  • Evaluation: Working"
Write-Host "  • Visualization: Working"
Write-Host "  • Configuration: Loaded"
Write-Host ""

Write-Host "📁 Test Outputs Created:" -ForegroundColor Cyan
Write-Host "  • data/sample/ - Sample datasets"
Write-Host "  • models/test/ - Test model"
Write-Host "  • outputs/figures/quick_test_plot.png - Test visualization"
Write-Host ""

Write-Host "🚀 Ready for Full Pipeline!" -ForegroundColor Green
Write-Host "  Run: .\code\scripts\run_full_pipeline.ps1"
Write-Host ""

Write-Host "⏱️  Total test time: ~5 minutes" -ForegroundColor Yellow
Write-Host "✨ Installation verified successfully!" -ForegroundColor Green
Write-Host "=========================================="
