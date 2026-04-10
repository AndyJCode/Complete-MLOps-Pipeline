import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mlflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, auc, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



def load_data(data_path: str) -> pd.DataFrame:
    """Load dataset from a CSV file."""
    return pd.read_csv(data_path)

#build multiple models based on config parameters
def build_model(config: Dict) -> object:
    """Return a model instance based on configuration."""
    model_type = config.get('model_type', 'logistic_regression')
    random_state = config.get('random_state', 12345)

    if model_type == 'logistic_regression':
        return LogisticRegression(
            C=config.get('lr_C', 1.0),
            solver='liblinear',
            random_state=random_state,
            max_iter=1000
        )
    if model_type == 'random_forest':
        return RandomForestClassifier(
            n_estimators=config.get('rf_n_estimators', 100),
            max_depth=config.get('rf_max_depth', None),
            random_state=random_state,
            bootstrap=config.get('rf_bootstrap', True),
            class_weight=config.get('class_weight', None)
        )
    if model_type == 'gradient_boosting':
        return GradientBoostingClassifier(
            n_estimators=config.get('gb_n_estimators', 100),
            learning_rate=config.get('gb_learning_rate', 0.1),
            max_depth=config.get('gb_max_depth', 3),
            random_state=random_state
        )

    raise ValueError(f"Unsupported model_type: {model_type}")


def handle_missing_values(df: pd.DataFrame, strategy: str = 'median') -> pd.DataFrame:
    """Fill missing values using the selected strategy."""
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    if strategy == 'mean':
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].mean())
    else:
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())

    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode().iloc[0])

    return df


def load_and_prepare_data(config: Dict) -> Tuple[pd.DataFrame, pd.Series, int, List[str], List[str]]:
    """Load the dataset and apply basic preprocessing."""
    df = load_data(config['data_path'])

    if config.get('binary_target', True):
        df['target'] = (df['target'] > 0).astype(int)

    missing_strategy = config.get('handle_missing', 'median')
    if missing_strategy:
        df = handle_missing_values(df, strategy=missing_strategy)

    drop_features = config.get('features_to_drop', []) or []
    df = df.drop(columns=drop_features, errors='ignore')

    numeric_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

    n_rows = len(df)
    X = df.drop('target', axis=1)
    y = df['target']

    return X, y, n_rows, numeric_cols, categorical_cols


def compute_data_version(data_path: str) -> str:
    """Compute a deterministic hash for the dataset file used in the experiment."""
    hash_obj = hashlib.sha256()
    resolved_path = Path(data_path).resolve()
    with open(resolved_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            hash_obj.update(chunk)
    return hash_obj.hexdigest()


def log_config_params(config: Dict) -> None:
    """Log all configuration values as MLflow parameters."""
    for key, value in config.items():
        if isinstance(value, (dict, list)):
            value = json.dumps(value)
        elif value is None:
            value = 'None'
        mlflow.log_param(key, str(value))



def load_config(config_path: str = 'configs/train_config.yaml') -> Dict:
    """Load a training configuration from YAML."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def run_experiment(config_path: str = 'configs/train_config.yaml') -> Optional[str]:
    """Run a single experiment with the given configuration file."""
    from src.evaluation import run_experiment_from_dict  # local import breaks circular dependency
    config = load_config(config_path)
    return run_experiment_from_dict(config)


if __name__ == '__main__':
    run_experiment('configs/train_config.yaml')

