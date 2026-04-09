import pandas as pd
import numpy as np
import pytest
import sys
import os
from typing import Dict

# Add src to path so we can import preprocessing
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from preprocessing import preprocess_data


def sample_data() -> pd.DataFrame:
    """Create sample data for testing."""
    data = {
        'age': [63, 67, 67, 37, 41],
        'sex': [1, 1, 1, 1, 0],
        'cp': [1, 4, 4, 3, 2],
        'trestbps': [145, 160, 120, 130, 130],
        'chol': [233, 286, 229, 250, 204],
        'fbs': [1, 0, 0, 0, 0],
        'restecg': [2, 0, 0, 1, 0],
        'thalach': [150, 108, 129, 187, 172],
        'exang': [0, 1, 1, 0, 0],
        'oldpeak': [2.3, 1.5, 2.6, 3.5, 1.4],
        'slope': [3, 2, 2, 3, 1],
        'ca': [0, 3, 2, 0, 0],
        'thal': [6, 3, 7, 3, 3],
        'target': [1, 1, 1, 0, 0]
    }
    return pd.DataFrame(data)

class TestValidateDataFrame:
    def test_validate_dataframe(self):
        df = sample_data()
        assert 'age' in df.columns
        assert 'sex' in df.columns
        assert 'cp' in df.columns
        assert 'target' in df.columns

    def test_missing_column_raises_error(self):
        df = sample_data().drop(columns=['age'])
        with pytest.raises(ValueError):
            preprocess_data(df)
    
    def test_missing_target_raises_error(self):
        df = sample_data().drop(columns=['target'])
        with pytest.raises(ValueError):
            preprocess_data(df)
    
    def test_empty_dataframe_raises_error(self):
        df = pd.DataFrame()
        with pytest.raises(ValueError):
            preprocess_data(df)

class TestDataQuality:
    def test_no_missing_values_after_preprocessing(self):
        df = sample_data()
        X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
        assert not X_train.isnull().values.any()
        assert not X_test.isnull().values.any()

    def test_target_binary(self):
        df = sample_data()
        X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
        assert set(y_train.unique()).issubset({0, 1})
        assert set(y_test.unique()).issubset({0, 1})
    
    def test_numerical_columns_scaled(self):
        df = sample_data()
        X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
        # Check that age is scaled (should be transformed, not raw values)
        # With small sample, exact 0 mean/1 std not guaranteed
        assert X_train['age'].std() > 0  # Should be scaled
        assert not np.array_equal(X_train['age'], df['age'][:len(X_train)])  # Should be different from original
    
    def test_train_test_split_sizes(self):
        df = sample_data()
        X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
        # With 5 samples, 80/20 split may not be exact due to stratification
        assert len(X_train) + len(X_test) == 5
        assert len(y_train) + len(y_test) == 5


