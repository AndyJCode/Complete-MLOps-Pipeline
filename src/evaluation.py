
import os
import json
from typing import Dict

import mlflow
import pandas as pd

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Import from train (shared utilities)
try:
    from src.train import (build_model, compute_data_version, load_and_prepare_data, log_config_params)
except ModuleNotFoundError:
    from train import (build_model, compute_data_version, load_and_prepare_data, log_config_params)


#MOVE THIS TO EVALUATION.PY
def run_experiment_from_dict(config: Dict) -> str:
    """Run a single experiment from a configuration dictionary."""
    mlflow.set_experiment(config.get('experiment_name', 'heart-disease-experiments'))
    with mlflow.start_run():
        # Define the path once with a fallback to your actual filename
        data_path = config.get('data_path', 'data/heart_combined.csv')
        
        # Use the variable instead of accessing the dict directly
        mlflow.log_param('data_path', data_path)
        mlflow.log_param('data_version', compute_data_version(data_path))
        
        log_config_params(config)

        X, y, n_rows, numeric_cols, categorical_cols = load_and_prepare_data(config)
        mlflow.log_param('n_rows', n_rows)
        mlflow.log_param('n_features', X.shape[1])

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=config.get('test_size', 0.2),
            random_state=config.get('random_state', 12345),
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