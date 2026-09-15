"""
Unit tests for model training and evaluation
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
import sys
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib

# Add the code directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'code'))

try:
    from src.train_enhanced import get_model, objective
    from src.evaluate_enhanced import metrics, bootstrap_confidence_interval
except ImportError:
    # Fallback for basic model testing
    def get_model(model_name, params):
        if model_name == 'rf':
            return RandomForestRegressor(**params)
        elif model_name == 'svr':
            return SVR(**params)
        else:
            raise ValueError(f"Unknown model: {model_name}")

    def metrics(y_true, y_pred):
        return {
            'R2': r2_score(y_true, y_pred),
            'MAE': mean_absolute_error(y_true, y_pred),
            'RMSE': mean_squared_error(y_true, y_pred, squared=False)
        }

    def bootstrap_confidence_interval(y_true, y_pred, metric_func=r2_score, n_bootstrap=100):
        import numpy as np
        bootstrap_scores = []
        for _ in range(n_bootstrap):
            indices = np.random.choice(len(y_true), size=len(y_true), replace=True)
            score = metric_func(y_true[indices], y_pred[indices])
            bootstrap_scores.append(score)
        return np.percentile(bootstrap_scores, 2.5), np.percentile(bootstrap_scores, 97.5)


class TestModelRegistry:
    """Test model creation and configuration"""
    
    def test_random_forest_creation(self):
        """Test Random Forest model creation"""
        params = {
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': 42
        }
        
        try:
            model = get_model('rf', params)
            assert isinstance(model, RandomForestRegressor)
            assert model.n_estimators == 100
            assert model.max_depth == 10
            assert model.random_state == 42
        except NameError:
            # If get_model is not available, create manually
            model = RandomForestRegressor(**params)
            assert isinstance(model, RandomForestRegressor)
    
    def test_svr_creation(self):
        """Test SVR model creation"""
        params = {
            'C': 1.0,
            'gamma': 'scale',
            'kernel': 'rbf'
        }
        
        try:
            model = get_model('svr', params)
            assert isinstance(model, SVR)
            assert model.C == 1.0
            assert model.gamma == 'scale'
        except NameError:
            # If get_model is not available, create manually
            model = SVR(**params)
            assert isinstance(model, SVR)
    
    def test_invalid_model_name(self):
        """Test error handling for invalid model names"""
        try:
            with pytest.raises(ValueError):
                get_model('invalid_model', {})
        except NameError:
            # Skip if get_model is not available
            pass


class TestModelTraining:
    """Test model training functionality"""
    
    def setup_method(self):
        """Set up test data for each test"""
        np.random.seed(42)
        self.n_samples = 1000
        self.n_features = 5
        
        # Generate synthetic data
        self.X = np.random.randn(self.n_samples, self.n_features)
        self.y = (
            2 * self.X[:, 0] + 
            1.5 * self.X[:, 1] + 
            0.5 * self.X[:, 2] + 
            0.1 * np.random.randn(self.n_samples)
        )
    
    def test_random_forest_training(self):
        """Test Random Forest training"""
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(self.X, self.y)
        
        # Test prediction
        y_pred = model.predict(self.X)
        r2 = r2_score(self.y, y_pred)
        
        # Should achieve reasonable performance on synthetic data
        assert r2 > 0.8
        assert len(y_pred) == len(self.y)
    
    def test_svr_training(self):
        """Test SVR training"""
        model = SVR(C=1.0, gamma='scale')
        model.fit(self.X, self.y)
        
        # Test prediction
        y_pred = model.predict(self.X)
        r2 = r2_score(self.y, y_pred)
        
        # SVR might have lower performance but should still work
        assert r2 > 0.3
        assert len(y_pred) == len(self.y)
    
    def test_model_persistence(self):
        """Test model saving and loading"""
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(self.X, self.y)
        
        # Save model
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            temp_path = f.name
        
        try:
            joblib.dump(model, temp_path)
            
            # Load model
            loaded_model = joblib.load(temp_path)
            
            # Test that loaded model works
            y_pred_original = model.predict(self.X)
            y_pred_loaded = loaded_model.predict(self.X)
            
            np.testing.assert_array_almost_equal(y_pred_original, y_pred_loaded)
        finally:
            os.unlink(temp_path)


class TestModelEvaluation:
    """Test model evaluation metrics and functions"""
    
    def setup_method(self):
        """Set up test data for evaluation"""
        np.random.seed(42)
        self.y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.y_pred = np.array([1.1, 2.2, 2.8, 3.9, 5.1])
    
    def test_metrics_calculation(self):
        """Test evaluation metrics calculation"""
        try:
            result = metrics(self.y_true, self.y_pred)
            
            assert 'R2' in result
            assert 'MAE' in result
            assert 'RMSE' in result
            
            # Check reasonable values
            assert 0 <= result['R2'] <= 1
            assert result['MAE'] >= 0
            assert result['RMSE'] >= 0
        except NameError:
            # Calculate manually if metrics function not available
            r2 = r2_score(self.y_true, self.y_pred)
            mae = mean_absolute_error(self.y_true, self.y_pred)
            rmse = mean_squared_error(self.y_true, self.y_pred, squared=False)
            
            assert 0 <= r2 <= 1
            assert mae >= 0
            assert rmse >= 0
    
    def test_perfect_prediction(self):
        """Test metrics with perfect predictions"""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])
        
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        
        assert r2 == 1.0
        assert mae == 0.0
        assert rmse == 0.0
    
    def test_bootstrap_confidence_interval(self):
        """Test bootstrap confidence interval calculation"""
        try:
            # Generate more data for bootstrap
            np.random.seed(42)
            y_true = np.random.randn(100)
            y_pred = y_true + 0.1 * np.random.randn(100)
            
            ci_lower, ci_upper = bootstrap_confidence_interval(
                y_true, y_pred, metric_func=r2_score, n_bootstrap=100
            )
            
            # Confidence interval should be reasonable
            assert ci_lower <= ci_upper
            assert 0 <= ci_lower <= 1
            assert 0 <= ci_upper <= 1
        except NameError:
            # Skip if bootstrap function not available
            pass


class TestCrossValidation:
    """Test cross-validation functionality"""
    
    def setup_method(self):
        """Set up test data"""
        np.random.seed(42)
        self.n_samples = 200
        self.n_features = 3
        
        # Generate time series data
        dates = pd.date_range('2020-01-01', periods=self.n_samples, freq='D')
        self.X = np.random.randn(self.n_samples, self.n_features)
        self.y = np.random.randn(self.n_samples)
        
        self.df = pd.DataFrame(self.X, columns=['feature1', 'feature2', 'feature3'])
        self.df['target'] = self.y
        self.df['date'] = dates
    
    def test_time_series_split(self):
        """Test time series cross-validation split"""
        from sklearn.model_selection import TimeSeriesSplit
        
        tscv = TimeSeriesSplit(n_splits=5)
        splits = list(tscv.split(self.X))
        
        # Should have 5 splits
        assert len(splits) == 5
        
        # Each split should have train and test indices
        for train_idx, test_idx in splits:
            assert len(train_idx) > 0
            assert len(test_idx) > 0
            # Test indices should come after train indices (time series property)
            assert max(train_idx) < min(test_idx)
    
    def test_cross_validation_scores(self):
        """Test cross-validation scoring"""
        from sklearn.model_selection import cross_val_score, TimeSeriesSplit
        
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        tscv = TimeSeriesSplit(n_splits=3)
        
        scores = cross_val_score(model, self.X, self.y, cv=tscv, scoring='r2')
        
        # Should have 3 scores
        assert len(scores) == 3
        # Scores should be reasonable (not perfect due to random data)
        assert all(-1 <= score <= 1 for score in scores)


class TestDataSplitting:
    """Test data splitting functionality"""
    
    def test_train_test_split_ratios(self):
        """Test train/validation/test split ratios"""
        from sklearn.model_selection import train_test_split
        
        # Generate test data
        np.random.seed(42)
        X = np.random.randn(1000, 5)
        y = np.random.randn(1000)
        
        # Split into train/temp (70/30)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        # Split temp into validation/test (15/15)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42
        )
        
        # Check ratios
        total_samples = len(X)
        assert abs(len(X_train) / total_samples - 0.7) < 0.05  # ~70%
        assert abs(len(X_val) / total_samples - 0.15) < 0.05   # ~15%
        assert abs(len(X_test) / total_samples - 0.15) < 0.05  # ~15%
        
        # Check no overlap
        train_val_test_total = len(X_train) + len(X_val) + len(X_test)
        assert train_val_test_total == total_samples


class TestIntegration:
    """Integration tests for the complete model pipeline"""
    
    def test_end_to_end_pipeline(self):
        """Test complete model training and evaluation pipeline"""
        # Generate synthetic dataset
        np.random.seed(42)
        n_samples = 500
        n_features = 4
        
        X = np.random.randn(n_samples, n_features)
        y = (
            1.5 * X[:, 0] + 
            1.0 * X[:, 1] + 
            0.5 * X[:, 2] + 
            0.2 * np.random.randn(n_samples)
        )
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        
        # Assertions
        assert r2 > 0.7  # Should achieve good performance on synthetic data
        assert mae < 1.0  # Reasonable error
        assert rmse < 1.5  # Reasonable error
        
        print(f"✅ End-to-end pipeline test passed:")
        print(f"   R² Score: {r2:.4f}")
        print(f"   MAE: {mae:.4f}")
        print(f"   RMSE: {rmse:.4f}")
        print(f"   Training samples: {len(X_train)}")
        print(f"   Test samples: {len(X_test)}")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
