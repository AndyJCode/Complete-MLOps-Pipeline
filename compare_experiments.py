import json
from pathlib import Path
from typing import Dict, List

import mlflow

from src.train import run_experiment_from_dict

TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "heart-disease-experiments"
BASE_CONFIG_PATH = "config.json"

# This script runs multiple experiments based on different configurations and compares their results in MLflow.
def load_base_config(path: str = BASE_CONFIG_PATH) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# This function builds a list of experiment configurations by modifying the base configuration with different model types and hyperparameters.
def build_experiment_variants(base_config: Dict) -> List[Dict]:
    base = dict(base_config)
    base.pop("experiment_name", None)

    variants = [
        {   # This variant tests a logistic regression with a specific regularization strength and feature scaling
            "model_type": "logistic_regression",
            "lr_C": 0.5,
            "scale_features": True,
            "features_to_drop": [],
        },
        {   # This variant tests the impact of dropping certain features and using a different regularization strength
            "model_type": "logistic_regression",
            "lr_C": 2.0,
            "scale_features": True,
            "features_to_drop": ["ca", "thal"],
        },
        {# This variant tests a random forest with specific hyperparameters and no feature scaling
            "model_type": "random_forest",
            "rf_n_estimators": 100,
            "rf_max_depth": 10,
            "rf_bootstrap": True,
            "class_weight": None,
            "scale_features": False,
            "features_to_drop": [],
        },
        {   # This variant tests a random forest with more trees, no max depth, and no bootstrapping to see how it affects performance
            "model_type": "random_forest",
            "rf_n_estimators": 200,
            "rf_max_depth": None,
            "rf_bootstrap": False,
            "scale_features": False,
            "features_to_drop": ["thal"],
        },
        {   # This variant tests a gradient boosting model with specific hyperparameters and feature scaling
            "model_type": "gradient_boosting",
            "gb_n_estimators": 150,
            "gb_learning_rate": 0.05,
            "gb_max_depth": 4,
            "scale_features": True,
            "features_to_drop": [],
        },
    ]
    # Build the full list of configs by combining the base config with each variant and adding necessary parameters for MLflow tracking and model output paths.
    configs = []
    for variant in variants:
        config = dict(base)
        config.update(variant)
        config["experiment_name"] = EXPERIMENT_NAME
        config["data_path"] = base_config["data_path"]
        config["random_state"] = base_config.get("random_state", 12345)
        config["test_size"] = base_config.get("test_size", 0.2)
        config["handle_missing"] = base_config.get("handle_missing", "median")
        config["model_output_path"] = f"models/model_{config['model_type']}.pkl"
        config["scaler_output_path"] = f"models/scaler_{config['model_type']}.pkl"
        configs.append(config)

    return configs

# This function runs all the experiments defined in the list of configurations and returns their run IDs for later comparison.
def run_all_experiments(configs: List[Dict]) -> List[str]:
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    run_ids = []
    for index, config in enumerate(configs, start=1):
        print(f"\n=== Running experiment {index}/{len(configs)}: {config['model_type']} ===")
        run_id = run_experiment_from_dict(config)
        run_ids.append(run_id)
    return run_ids


# ANALYZE RESULTS
# This function retrieves the results of all completed runs for the specified experiment from MLflow,
# sorts them by F1 score, and prints the top 5 runs plus the best run in detail.
def compare_experiments(primary_metric: str = "f1_score") -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError(f"Experiment '{EXPERIMENT_NAME}' not found in tracking store.")

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="status = 'FINISHED'",
        order_by=[f"metrics.{primary_metric} DESC"],
        max_results=50,
    )
    if runs.empty:
        print("No completed runs found for the experiment.")
        return

    print("Top 5 Runs by F1 Score:")
    print("=" * 80)
    for i, row in runs.head(5).iterrows():
        print(f"\nRun: {row['run_id'][:8]}...")
        print(f"  Model:    {row['params.model_type']}")
        print(f"  F1:       {row['metrics.f1_score']:.4f}")
        print(f"  Accuracy: {row['metrics.accuracy']:.4f}")
        print(f"  AUC-ROC:  {row['metrics.auc_roc']:.4f}")

    best_run = runs.iloc[0]
    print(f"\n{'=' * 80}")
    print("BEST MODEL")
    print(f"{'=' * 80}")
    print(f"Run ID:     {best_run['run_id']}")
    print(f"Model Type: {best_run['params.model_type']}")
    print(f"F1 Score:   {best_run['metrics.f1_score']:.4f}")
    print(f"Accuracy:   {best_run['metrics.accuracy']:.4f}")
    print(f"AUC-ROC:    {best_run['metrics.auc_roc']:.4f}")

    print(f"\n{'=' * 80}")
    print("Average F1 Score by Model Type:")
    print(f"{'=' * 80}")
    summary = runs.groupby("params.model_type")["metrics.f1_score"].agg(["mean", "max", "count"])
    summary.columns = ["avg_f1", "best_f1", "num_runs"]
    print(summary.sort_values("best_f1", ascending=False).to_string())


if __name__ == "__main__":
    base_config = load_base_config()
    configs = build_experiment_variants(base_config)
    run_all_experiments(configs)
    compare_experiments(primary_metric="f1_score")
