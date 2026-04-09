import pandas as pd
from pathlib import Path


def combine_heart_data() -> None:
    """Combine multiple heart disease datasets into a single CSV file.
    
    Reads processed data files from data/raw/heart+disease/ and combines them
    into data/raw/heart_combined.csv.
    """
    project_root = Path(__file__).resolve().parents[1]
    columns = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
    ]

    # Load the raw data from the heart+disease folder.
    source_dir = project_root / "data" / "raw" / "heart+disease"
    data_files = [
        "processed.cleveland.data",
        "processed.hungarian.data",
        "processed.switzerland.data",
        "processed.va.data"
    ]

    frames = []
    for filename in data_files:
        path = source_dir / filename
        if path.exists():
            df = pd.read_csv(path, names=columns, na_values="?")
            frames.append(df)

    if not frames:
        raise FileNotFoundError(
            f"No Heart Disease source files found in {source_dir}."
        )

    combined_df = pd.concat(frames, ignore_index=True)
    output_path = project_root / "data" / "raw" / "heart_combined.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(output_path, index=False)

    print(f"Combined {len(combined_df)} rows into {output_path}")


if __name__ == "__main__":
    combine_heart_data()
