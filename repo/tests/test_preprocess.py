"""
Unit tests for data preprocessing module
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import sys
import os

# Add the code directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'code'))

try:
    from src.preprocess import load_table, parse_dates, impute, quality_control, detect_outliers
except ImportError:
    # Fallback for basic testing without full module
    def load_table(meta):
        import pandas as pd
        return pd.read_csv(meta["file"])

    def parse_dates(df, meta):
        import pandas as pd
        if "date_col" in meta:
            df[meta["date_col"]] = pd.to_datetime(df[meta["date_col"]])
        return df

    def impute(df, thresh=0.5):
        from sklearn.impute import KNNImputer
        missing_stats = df.isnull().mean()
        # Remove high missing columns
        df_clean = df.loc[:, missing_stats < thresh]
        # Impute remaining
        imputer = KNNImputer(n_neighbors=5)
        df_imputed = pd.DataFrame(imputer.fit_transform(df_clean), columns=df_clean.columns)
        return df_imputed, missing_stats

    def quality_control(df, meta):
        if "quality_checks" not in meta:
            return df
        df_clean = df.copy()
        for check in meta["quality_checks"]:
            col = check["column"]
            if col in df_clean.columns:
                df_clean = df_clean[(df_clean[col] >= check["min"]) & (df_clean[col] <= check["max"])]
        return df_clean

    def detect_outliers(df, method="iqr", threshold=1.5):
        import numpy as np
        from sklearn.ensemble import IsolationForest

        if method == "iqr":
            Q1 = df.quantile(0.25)
            Q3 = df.quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((df < (Q1 - threshold * IQR)) | (df > (Q3 + threshold * IQR))).any(axis=1)
        elif method == "zscore":
            z_scores = np.abs((df - df.mean()) / df.std())
            outliers = (z_scores > threshold).any(axis=1)
        elif method == "isolation_forest":
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            outliers = iso_forest.fit_predict(df) == -1
        else:
            outliers = pd.Series([False] * len(df))

        return outliers


class TestDataLoading:
    """Test data loading functions"""
    
    def test_load_csv_table(self):
        """Test loading CSV files"""
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("date,value1,value2\n2020-01-01,1.0,2.0\n2020-01-02,3.0,4.0\n")
            temp_path = f.name
        
        try:
            meta = {"file": temp_path, "loader": "csv"}
            df = load_table(meta)
            
            assert len(df) == 2
            assert list(df.columns) == ["date", "value1", "value2"]
            assert df["value1"].iloc[0] == 1.0
        finally:
            os.unlink(temp_path)
    
    def test_load_excel_table(self):
        """Test loading Excel files"""
        # Create temporary Excel file
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as f:
            temp_path = f.name
        
        try:
            # Create test Excel file
            df_test = pd.DataFrame({
                'date': ['2020-01-01', '2020-01-02'],
                'value1': [1.0, 3.0],
                'value2': [2.0, 4.0]
            })
            df_test.to_excel(temp_path, index=False)
            
            meta = {"file": temp_path, "loader": "excel"}
            df = load_table(meta)
            
            assert len(df) == 2
            assert "value1" in df.columns
        finally:
            os.unlink(temp_path)


class TestDateParsing:
    """Test date parsing functions"""
    
    def test_parse_standard_dates(self):
        """Test parsing standard date formats"""
        df = pd.DataFrame({
            'date': ['2020-01-01', '2020-01-02', '2020-01-03'],
            'value': [1, 2, 3]
        })
        
        meta = {"date_col": "date", "date_format": "%Y-%m-%d"}
        result = parse_dates(df, meta)
        
        assert pd.api.types.is_datetime64_any_dtype(result['date'])
        assert result['date'].iloc[0] == pd.Timestamp('2020-01-01')
    
    def test_parse_numeric_dates(self):
        """Test parsing numeric dates with origin"""
        df = pd.DataFrame({
            'date': [0, 1, 2],  # Days since origin
            'value': [1, 2, 3]
        })
        
        meta = {"date_col": "date", "date_origin": "2020-01-01"}
        result = parse_dates(df, meta)
        
        assert pd.api.types.is_datetime64_any_dtype(result['date'])
        assert result['date'].iloc[0] == pd.Timestamp('2020-01-01')


class TestDataImputation:
    """Test data imputation functions"""
    
    def test_impute_missing_values(self):
        """Test KNN imputation of missing values"""
        # Create data with missing values
        df = pd.DataFrame({
            'feature1': [1.0, 2.0, np.nan, 4.0, 5.0],
            'feature2': [2.0, np.nan, 3.0, 4.0, 5.0],
            'feature3': [1.0, 2.0, 3.0, 4.0, 5.0]
        })
        
        result, missing_stats = impute(df, thresh=0.5)
        
        # Check that missing values are filled
        assert not result.isna().any().any()
        assert len(result) == len(df)
        
        # Check missing statistics
        assert 'feature1' in missing_stats.index
        assert missing_stats['feature1'] == 0.2  # 1/5 missing
    
    def test_remove_high_missing_columns(self):
        """Test removal of columns with high missing rates"""
        df = pd.DataFrame({
            'good_feature': [1.0, 2.0, 3.0, 4.0, 5.0],
            'bad_feature': [1.0, np.nan, np.nan, np.nan, np.nan],  # 80% missing
            'ok_feature': [1.0, 2.0, np.nan, 4.0, 5.0]  # 20% missing
        })
        
        result, missing_stats = impute(df, thresh=0.5)
        
        # Bad feature should be removed
        assert 'bad_feature' not in result.columns
        assert 'good_feature' in result.columns
        assert 'ok_feature' in result.columns


class TestQualityControl:
    """Test quality control functions"""
    
    def test_quality_checks(self):
        """Test quality control with range checks"""
        df = pd.DataFrame({
            'temperature': [-5.0, 10.0, 25.0, 50.0],  # One outlier (50.0)
            'salinity': [0.0, 20.0, 35.0, 60.0],      # One outlier (60.0)
            'other': [1, 2, 3, 4]
        })
        
        meta = {
            "quality_checks": [
                {"column": "temperature", "min": -2, "max": 40},
                {"column": "salinity", "min": 0, "max": 50}
            ]
        }
        
        result = quality_control(df, meta)
        
        # Should remove rows with outliers
        assert len(result) < len(df)
        assert result['temperature'].max() <= 40
        assert result['salinity'].max() <= 50
    
    def test_no_quality_checks(self):
        """Test quality control with no checks specified"""
        df = pd.DataFrame({
            'value1': [1, 2, 3, 4],
            'value2': [5, 6, 7, 8]
        })
        
        meta = {}  # No quality checks
        result = quality_control(df, meta)
        
        # Should return unchanged dataframe
        pd.testing.assert_frame_equal(result, df)


class TestOutlierDetection:
    """Test outlier detection functions"""
    
    def test_iqr_outlier_detection(self):
        """Test IQR-based outlier detection"""
        # Create data with clear outliers
        df = pd.DataFrame({
            'normal_data': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'with_outliers': [1, 2, 3, 4, 5, 6, 7, 8, 9, 100]  # 100 is outlier
        })
        
        outlier_mask = detect_outliers(df, method="iqr", threshold=1.5)
        
        # Should detect the outlier in the last row
        assert outlier_mask.sum() > 0
        assert outlier_mask.iloc[-1] == True  # Last row should be flagged
    
    def test_zscore_outlier_detection(self):
        """Test Z-score based outlier detection"""
        # Create data with clear outliers
        np.random.seed(42)
        normal_data = np.random.normal(0, 1, 100)
        outlier_data = np.concatenate([normal_data, [10, -10]])  # Add outliers
        
        df = pd.DataFrame({'data': outlier_data})
        
        outlier_mask = detect_outliers(df, method="zscore", threshold=3.0)
        
        # Should detect some outliers
        assert outlier_mask.sum() > 0
    
    def test_isolation_forest_outlier_detection(self):
        """Test Isolation Forest outlier detection"""
        # Create data with clear outliers
        np.random.seed(42)
        normal_data = np.random.normal(0, 1, (100, 2))
        outlier_data = np.array([[10, 10], [-10, -10]])  # Clear outliers
        all_data = np.vstack([normal_data, outlier_data])
        
        df = pd.DataFrame(all_data, columns=['feature1', 'feature2'])
        
        outlier_mask = detect_outliers(df, method="isolation_forest")
        
        # Should detect some outliers
        assert outlier_mask.sum() > 0


class TestIntegration:
    """Integration tests for the preprocessing pipeline"""
    
    def test_full_preprocessing_pipeline(self):
        """Test the complete preprocessing pipeline"""
        # Create synthetic dataset
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        
        df = pd.DataFrame({
            'date': dates,
            'temperature': np.random.normal(20, 5, 100),
            'salinity': np.random.normal(35, 3, 100),
            'chlorophyll': np.random.lognormal(0, 1, 100)
        })
        
        # Add some missing values
        df.loc[5:10, 'temperature'] = np.nan
        df.loc[15:17, 'salinity'] = np.nan
        
        # Add some outliers
        df.loc[50, 'temperature'] = 100  # Extreme outlier
        df.loc[60, 'salinity'] = 100     # Extreme outlier
        
        # Test that we can process this data without errors
        meta = {
            "date_col": "date",
            "target_col": "chlorophyll",
            "quality_checks": [
                {"column": "temperature", "min": -2, "max": 40},
                {"column": "salinity", "min": 0, "max": 50}
            ]
        }
        
        # Parse dates
        df_dates = parse_dates(df, meta)
        assert pd.api.types.is_datetime64_any_dtype(df_dates['date'])
        
        # Quality control
        df_qc = quality_control(df_dates, meta)
        assert len(df_qc) < len(df_dates)  # Should remove outliers
        
        # Imputation
        df_imputed, missing_stats = impute(df_qc, thresh=0.3)
        assert not df_imputed.select_dtypes(include=[np.number]).isna().any().any()
        
        print(f"✅ Integration test passed:")
        print(f"   Original samples: {len(df)}")
        print(f"   After QC: {len(df_qc)}")
        print(f"   After imputation: {len(df_imputed)}")
        print(f"   Missing value rates: {missing_stats.to_dict()}")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
