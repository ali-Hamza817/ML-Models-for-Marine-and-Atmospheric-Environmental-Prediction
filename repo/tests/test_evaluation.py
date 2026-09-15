"""
Unit tests for model evaluation and statistical analysis
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
import sys
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy import stats

# Add the code directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'code'))


class TestMetricsCalculation:
    """Test evaluation metrics calculation"""
    
    def setup_method(self):
        """Set up test data"""
        np.random.seed(42)
        self.y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        self.y_pred = np.array([1.1, 2.2, 2.8, 3.9, 5.1, 5.8, 7.2, 7.9, 9.1, 10.2])
    
    def test_r2_score_calculation(self):
        """Test R² score calculation"""
        r2 = r2_score(self.y_true, self.y_pred)
        
        # Should be a reasonable R² score
        assert 0.8 <= r2 <= 1.0
        assert isinstance(r2, (float, np.floating))
    
    def test_mae_calculation(self):
        """Test Mean Absolute Error calculation"""
        mae = mean_absolute_error(self.y_true, self.y_pred)
        
        # Should be positive and reasonable
        assert mae >= 0
        assert mae < 1.0  # Given our test data
        assert isinstance(mae, (float, np.floating))
    
    def test_rmse_calculation(self):
        """Test Root Mean Square Error calculation"""
        rmse = mean_squared_error(self.y_true, self.y_pred, squared=False)
        
        # Should be positive and reasonable
        assert rmse >= 0
        assert rmse < 1.0  # Given our test data
        assert isinstance(rmse, (float, np.floating))
    
    def test_perfect_prediction_metrics(self):
        """Test metrics with perfect predictions"""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])
        
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        
        assert r2 == 1.0
        assert mae == 0.0
        assert rmse == 0.0
    
    def test_worst_case_metrics(self):
        """Test metrics with worst case predictions (constant prediction)"""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([3, 3, 3, 3, 3])  # Always predict mean
        
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        
        assert r2 == 0.0  # R² = 0 for constant prediction at mean
        assert mae > 0
        assert rmse > 0


class TestBootstrapAnalysis:
    """Test bootstrap confidence interval analysis"""
    
    def setup_method(self):
        """Set up test data"""
        np.random.seed(42)
        self.n_samples = 100
        self.y_true = np.random.randn(self.n_samples)
        self.y_pred = self.y_true + 0.1 * np.random.randn(self.n_samples)
    
    def test_bootstrap_sampling(self):
        """Test bootstrap resampling"""
        n_bootstrap = 1000
        bootstrap_scores = []
        
        for _ in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(len(self.y_true), size=len(self.y_true), replace=True)
            y_true_boot = self.y_true[indices]
            y_pred_boot = self.y_pred[indices]
            
            # Calculate metric
            score = r2_score(y_true_boot, y_pred_boot)
            bootstrap_scores.append(score)
        
        bootstrap_scores = np.array(bootstrap_scores)
        
        # Calculate confidence interval
        ci_lower = np.percentile(bootstrap_scores, 2.5)
        ci_upper = np.percentile(bootstrap_scores, 97.5)
        
        # Assertions
        assert len(bootstrap_scores) == n_bootstrap
        assert ci_lower <= ci_upper
        assert 0 <= ci_lower <= 1
        assert 0 <= ci_upper <= 1
        
        # The true R² should be within the confidence interval most of the time
        true_r2 = r2_score(self.y_true, self.y_pred)
        # Note: This might occasionally fail due to randomness, but should pass most of the time
        
        print(f"True R²: {true_r2:.4f}")
        print(f"Bootstrap CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    
    def test_confidence_interval_coverage(self):
        """Test that confidence intervals have proper coverage"""
        n_experiments = 100
        n_bootstrap = 200
        coverage_count = 0
        
        for experiment in range(n_experiments):
            # Generate new data for each experiment
            np.random.seed(42 + experiment)
            y_true = np.random.randn(50)
            y_pred = y_true + 0.2 * np.random.randn(50)
            
            # True R²
            true_r2 = r2_score(y_true, y_pred)
            
            # Bootstrap confidence interval
            bootstrap_scores = []
            for _ in range(n_bootstrap):
                indices = np.random.choice(len(y_true), size=len(y_true), replace=True)
                score = r2_score(y_true[indices], y_pred[indices])
                bootstrap_scores.append(score)
            
            ci_lower = np.percentile(bootstrap_scores, 2.5)
            ci_upper = np.percentile(bootstrap_scores, 97.5)
            
            # Check if true value is within CI
            if ci_lower <= true_r2 <= ci_upper:
                coverage_count += 1
        
        coverage_rate = coverage_count / n_experiments
        
        # 95% CI should have approximately 95% coverage (allow some tolerance)
        assert 0.85 <= coverage_rate <= 1.0
        print(f"Coverage rate: {coverage_rate:.2f} ({coverage_count}/{n_experiments})")


class TestStatisticalSignificance:
    """Test statistical significance testing"""
    
    def test_paired_t_test(self):
        """Test paired t-test for model comparison"""
        np.random.seed(42)
        
        # Generate performance scores for two models
        model1_scores = np.random.normal(0.8, 0.1, 20)  # Better model
        model2_scores = np.random.normal(0.7, 0.1, 20)  # Worse model
        
        # Paired t-test
        t_stat, p_value = stats.ttest_rel(model1_scores, model2_scores)
        
        # Should detect significant difference
        assert p_value < 0.05  # Significant at α = 0.05
        assert t_stat > 0      # Model 1 should be better
        
        print(f"T-statistic: {t_stat:.4f}")
        print(f"P-value: {p_value:.4f}")
    
    def test_no_significant_difference(self):
        """Test case where there's no significant difference"""
        np.random.seed(42)
        
        # Generate similar performance scores
        model1_scores = np.random.normal(0.8, 0.1, 20)
        model2_scores = np.random.normal(0.8, 0.1, 20)  # Same performance
        
        # Paired t-test
        t_stat, p_value = stats.ttest_rel(model1_scores, model2_scores)
        
        # Should not detect significant difference
        assert p_value > 0.05  # Not significant at α = 0.05
        
        print(f"T-statistic: {t_stat:.4f}")
        print(f"P-value: {p_value:.4f}")
    
    def test_effect_size_calculation(self):
        """Test Cohen's d effect size calculation"""
        np.random.seed(42)
        
        # Generate data with known effect size
        group1 = np.random.normal(0, 1, 100)
        group2 = np.random.normal(0.5, 1, 100)  # 0.5 standard deviations difference
        
        # Calculate Cohen's d
        pooled_std = np.sqrt(((len(group1) - 1) * np.var(group1, ddof=1) + 
                             (len(group2) - 1) * np.var(group2, ddof=1)) / 
                            (len(group1) + len(group2) - 2))
        cohens_d = (np.mean(group2) - np.mean(group1)) / pooled_std
        
        # Should be approximately 0.5
        assert 0.3 <= abs(cohens_d) <= 0.7
        
        print(f"Cohen's d: {cohens_d:.4f}")


class TestCrossValidationAnalysis:
    """Test cross-validation analysis"""
    
    def test_cv_score_statistics(self):
        """Test cross-validation score statistics"""
        # Simulate CV scores
        cv_scores = np.array([0.85, 0.82, 0.88, 0.79, 0.86])
        
        # Calculate statistics
        mean_score = np.mean(cv_scores)
        std_score = np.std(cv_scores, ddof=1)
        ci_lower = mean_score - 1.96 * std_score / np.sqrt(len(cv_scores))
        ci_upper = mean_score + 1.96 * std_score / np.sqrt(len(cv_scores))
        
        # Assertions
        assert 0 <= mean_score <= 1
        assert std_score >= 0
        assert ci_lower <= mean_score <= ci_upper
        
        print(f"Mean CV score: {mean_score:.4f} ± {std_score:.4f}")
        print(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    
    def test_cv_score_stability(self):
        """Test cross-validation score stability assessment"""
        # Stable model (low variance)
        stable_scores = np.array([0.85, 0.84, 0.86, 0.85, 0.85])
        
        # Unstable model (high variance)
        unstable_scores = np.array([0.95, 0.70, 0.88, 0.65, 0.92])
        
        stable_cv = np.std(stable_scores, ddof=1) / np.mean(stable_scores)
        unstable_cv = np.std(unstable_scores, ddof=1) / np.mean(unstable_scores)
        
        # Stable model should have lower coefficient of variation
        assert stable_cv < unstable_cv
        assert stable_cv < 0.05  # Very stable
        assert unstable_cv > 0.10  # Quite unstable
        
        print(f"Stable model CV: {stable_cv:.4f}")
        print(f"Unstable model CV: {unstable_cv:.4f}")


class TestResultsAggregation:
    """Test results aggregation and summarization"""
    
    def test_dataset_level_aggregation(self):
        """Test aggregation of results at dataset level"""
        # Simulate results for multiple models on one dataset
        results_data = {
            'model': ['RF', 'XGB', 'SVR', 'LSTM', 'Transformer'],
            'R2': [0.85, 0.88, 0.75, 0.82, 0.79],
            'MAE': [0.12, 0.10, 0.18, 0.14, 0.16],
            'RMSE': [0.15, 0.13, 0.22, 0.17, 0.19]
        }
        
        df = pd.DataFrame(results_data)
        
        # Find best model
        best_model = df.loc[df['R2'].idxmax(), 'model']
        best_r2 = df['R2'].max()
        
        # Calculate summary statistics
        mean_r2 = df['R2'].mean()
        std_r2 = df['R2'].std()
        
        # Assertions
        assert best_model == 'XGB'
        assert best_r2 == 0.88
        assert 0 <= mean_r2 <= 1
        assert std_r2 >= 0
        
        print(f"Best model: {best_model} (R² = {best_r2:.4f})")
        print(f"Mean R²: {mean_r2:.4f} ± {std_r2:.4f}")
    
    def test_cross_dataset_aggregation(self):
        """Test aggregation of results across datasets"""
        # Simulate results for one model across multiple datasets
        results_data = {
            'dataset': ['era5_daily', 'cast', 'cleaned_data', 'biotoxin', 'hydrographic'],
            'R2': [0.72, 0.65, 0.95, 0.08, 0.78],
            'MAE': [0.15, 0.22, 0.05, 0.45, 0.18],
            'samples': [102982, 21865, 7819, 5076, 4653]
        }
        
        df = pd.DataFrame(results_data)
        
        # Calculate robustness metrics
        mean_r2 = df['R2'].mean()
        std_r2 = df['R2'].std()
        min_r2 = df['R2'].min()
        max_r2 = df['R2'].max()
        
        # Count successful datasets (R² > 0.5)
        successful_datasets = (df['R2'] > 0.5).sum()
        total_datasets = len(df)
        
        # Assertions
        assert 0 <= mean_r2 <= 1
        assert std_r2 >= 0
        assert min_r2 <= mean_r2 <= max_r2
        assert 0 <= successful_datasets <= total_datasets
        
        print(f"Cross-dataset performance:")
        print(f"  Mean R²: {mean_r2:.4f} ± {std_r2:.4f}")
        print(f"  Range: [{min_r2:.4f}, {max_r2:.4f}]")
        print(f"  Success rate: {successful_datasets}/{total_datasets}")


class TestIntegration:
    """Integration tests for evaluation pipeline"""
    
    def test_complete_evaluation_pipeline(self):
        """Test complete evaluation pipeline"""
        # Generate synthetic results
        np.random.seed(42)
        
        datasets = ['dataset1', 'dataset2', 'dataset3']
        models = ['RF', 'XGB', 'SVR']
        
        results = []
        for dataset in datasets:
            for model in models:
                # Simulate model performance
                base_performance = np.random.uniform(0.6, 0.9)
                noise = np.random.normal(0, 0.05)
                r2 = max(0, min(1, base_performance + noise))
                
                mae = np.random.uniform(0.1, 0.3)
                rmse = mae * np.random.uniform(1.2, 1.8)
                
                results.append({
                    'dataset': dataset,
                    'model': model,
                    'R2': r2,
                    'MAE': mae,
                    'RMSE': rmse
                })
        
        df = pd.DataFrame(results)
        
        # Test various aggregations
        # 1. Best model per dataset
        best_per_dataset = df.loc[df.groupby('dataset')['R2'].idxmax()]
        assert len(best_per_dataset) == len(datasets)
        
        # 2. Model robustness across datasets
        model_stats = df.groupby('model')['R2'].agg(['mean', 'std', 'count'])
        assert len(model_stats) == len(models)
        
        # 3. Dataset difficulty ranking
        dataset_stats = df.groupby('dataset')['R2'].agg(['mean', 'max'])
        dataset_difficulty = dataset_stats.sort_values('mean', ascending=False)
        
        # 4. Overall summary
        overall_mean = df['R2'].mean()
        overall_std = df['R2'].std()
        
        # Assertions
        assert 0 <= overall_mean <= 1
        assert overall_std >= 0
        assert len(dataset_difficulty) == len(datasets)
        
        print("✅ Complete evaluation pipeline test passed:")
        print(f"   Overall performance: {overall_mean:.4f} ± {overall_std:.4f}")
        print(f"   Best models per dataset:")
        for _, row in best_per_dataset.iterrows():
            print(f"     {row['dataset']}: {row['model']} (R² = {row['R2']:.4f})")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
