# Heart Disease Prediction MLOps Project

This project demonstrates a complete ML pipeline for heart disease prediction, including data preprocessing, model training, MLflow experiment tracking, drift monitoring with Evidently, and automated testing.

## Project Structure

- `src/` — Python source code
- `configs/` — YAML configuration files
- `tests/` — pytest test suite
- `.github/workflows/` — GitHub Actions CI pipeline
- `requirements.txt` — Python dependencies
- `data/` — raw and processed dataset files
- `reports/` — generated drift monitoring reports
- `mlruns/` and `mlflow.db` — local MLflow tracking artifacts

## Setup

1. Create and activate a Python environment.
2. Install dependencies:
   ```bash
   python -m pip install -r requirements.txt
   ```
3. Initialize DVC and pull data (if configured):
   ```bash
   dvc pull
   ```

## Run training

Train the model using the YAML config:

```bash
python src/train.py
```

## Compare experiments

Run the experiment comparison script to identify the best run:

```bash
python compare_experiments.py
```

## Run drift monitoring

Generate drift reports from the production simulation:

```bash
python src/monitor_drift.py
```

## Run tests

Execute the pytest suite:

```bash
python -m pytest tests/ -v
```

## Notes

- The project uses MLflow for experiment tracking and logs hyperparameters, data version hash, metrics, and model artifacts.
- Drift monitoring is implemented with Evidently and saves HTML reports to `reports/`.
- The training script reads hyperparameters and file paths from `configs/train_config.yaml`.
