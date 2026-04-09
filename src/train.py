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

    if config.get('handle_missing', 'median'):
        df = handle_missing_values(df, strategy=config['handle_missing'])

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


def run_experiment_from_dict(config: Dict) -> str:
    """Run a single experiment from a configuration dictionary."""
    mlflow.set_experiment(config.get('experiment_name', 'heart-disease-experiments'))
    with mlflow.start_run():
        mlflow.log_param('data_path', config.get('data_path', 'unknown'))
        mlflow.log_param('data_version', compute_data_version(config['data_path']))
        log_config_params(config)

        X, y, n_rows, numeric_cols, categorical_cols = load_and_prepare_data(config)
        mlflow.log_param('n_rows', n_rows)
        mlflow.log_param('n_features', X.shape[1])

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=config['test_size'],
            random_state=config['random_state'],
            stratify=y
        )

        scaler = None
        if config.get('scale_features', True):
            scaler = StandardScaler()
            X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
            X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

        model = build_model(config)
        print(f"\nTraining {config['model_type']}...")
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_prob) if y_prob is not None else float('nan')

        mlflow.log_metric('accuracy', round(accuracy, 4))
        mlflow.log_metric('precision', round(precision, 4))
        mlflow.log_metric('recall', round(recall, 4))
        mlflow.log_metric('f1_score', round(f1, 4))
        if y_prob is not None:
            mlflow.log_metric('auc_roc', round(auc_score, 4))

        os.makedirs('models', exist_ok=True)
        os.makedirs('metrics', exist_ok=True)
        model_path = config.get('model_output_path', 'models/model.pkl')
        scaler_path = config.get('scaler_output_path', 'models/scaler.pkl')

        with open(model_path, 'wb') as f:
            pd.to_pickle(model, f)
        if scaler is not None:
            with open(scaler_path, 'wb') as f:
                pd.to_pickle(scaler, f)

        mlflow.sklearn.log_model(model, 'model')

        config_snapshot = 'config_snapshot.json'
        with open(config_snapshot, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        mlflow.log_artifact(config_snapshot)
        os.remove(config_snapshot)

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_roc': auc_score
        }
        with open('metrics/results.json', 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"\n{'='*50}")
        print(f"Model:     {config['model_type']}")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        if y_prob is not None:
            print(f"AUC-ROC:   {auc_score:.4f}")
        print(f"{'='*50}")

        run_id = mlflow.active_run().info.run_id
        print(f"\nMLflow Run ID: {run_id}")
        print("View this run in the UI: mlflow ui")

        return run_id


def load_config(config_path: str = 'configs/train_config.yaml') -> Dict:
    """Load a training configuration from YAML."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def run_experiment(config_path: str = 'configs/train_config.yaml') -> Optional[str]:
    """Run a single experiment with the given configuration file."""
    config = load_config(config_path)
    return run_experiment_from_dict(config)


if __name__ == '__main__':
    run_experiment('configs/train_config.yaml')

