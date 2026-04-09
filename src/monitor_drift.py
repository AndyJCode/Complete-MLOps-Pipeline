import os
from pathlib import Path
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

# Configuration
DRIFT_SHARE_WARNING = 0.20    # warn if more than 20% of features drift
DRIFT_SHARE_CRITICAL = 0.40   # fail if more than 40% of features drift


def load_data(data_path: str) -> pd.DataFrame:
    """Load dataset from a CSV file."""
    return pd.read_csv(data_path)


def create_reference_and_production(df: pd.DataFrame):
    """Split data into reference and simulated production batches."""
    df = df.sample(frac=1, random_state=12345).reset_index(drop=True)
    split_index = int(len(df) * 0.8)
    reference = df.iloc[:split_index].copy()
    remaining = df.iloc[split_index:].copy()

    batch_size = max(1, len(remaining) // 3)
    month1 = remaining.iloc[:batch_size].copy()
    month2 = remaining.iloc[batch_size:2 * batch_size].copy()
    month3 = remaining.iloc[2 * batch_size:].copy()

    return reference, month1, month2, month3


def introduce_drift(df: pd.DataFrame, drift_type: str) -> pd.DataFrame:
    """Introduce synthetic drift into the dataset."""
    df = df.copy()
    if drift_type == 'covariate':
        df['age'] = df['age'] + 5
    elif drift_type == 'label':
        flip_indices = df.sample(frac=0.2, random_state=12345).index
        df.loc[flip_indices, 'target'] = 1 - df.loc[flip_indices, 'target']
    elif drift_type == 'concept':
        df['target'] = ((df['age'] > 60) & (df['target'] == 0)).astype(int) + \
                       ((df['age'] <= 60) & (df['target'] == 1)).astype(int)
    return df


def generate_drift_report(reference: pd.DataFrame, current: pd.DataFrame, output_path: Path) -> dict:
    """Generate a drift report and save it as HTML."""
    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(reference_data=reference, current_data=current)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(output_path))

    summary = snapshot.dict()["metrics"][0]["result"]
    return summary


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "data" / "raw" / "heart_combined.csv"
    df = load_data(data_path)

    reference, month1, month2, month3 = create_reference_and_production(df)
    month_batches = {
        "month1": introduce_drift(month1, 'covariate'),
        "month2": introduce_drift(month2, 'label'),
        "month3": introduce_drift(month3, 'concept'),
    }

    reports_dir = project_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    exit_code = 0
    for batch_name, batch_data in month_batches.items():
        output_path = reports_dir / f"drift_report_{batch_name}.html"
        summary = generate_drift_report(reference, batch_data, output_path)

        share = summary.get("share_of_drifted_columns", 0)
        dataset_drift = summary.get("dataset_drift", False)

        print(f"\nBatch: {batch_name}")
        print(f"Feature drift share: {share:.3f}")
        print(f"Dataset drift detected: {dataset_drift}")
        print(f"Report saved to: {output_path}")

        if share >= DRIFT_SHARE_CRITICAL:
            exit_code = 1

    if exit_code == 1:
        print("\nCritical drift detected. Exiting with failure.")
    else:
        print("\nDrift monitoring completed. No critical drift detected.")

    raise SystemExit(exit_code)
