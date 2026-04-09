import pandas as pd


def load_dataset() -> pd.DataFrame:
    return pd.read_csv("data/raw/heart_combined.csv")


def test_required_columns_exist():
    df = load_dataset()
    expected_columns = {
        "age",
        "sex",
        "cp",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalach",
        "exang",
        "oldpeak",
        "slope",
        "ca",
        "thal",
        "target",
    }
    assert expected_columns.issubset(df.columns)


def test_target_values_are_valid():
    df = load_dataset()
    assert set(df["target"].dropna().unique()).issubset({0, 1, 2, 3, 4})


def test_numeric_columns_have_reasonable_ranges():
    df = load_dataset()
    assert df["age"].between(20, 100).all()
    # if chol is not in range, it may indicate data quality issues or outliers that could affect model performance
    chol = df["chol"].replace(0, float("nan"))
    invalid = chol.dropna()[~chol.dropna().between(0, 600)]

    # Allow tiny fraction of bad data
    assert len(invalid) / len(chol.dropna()) < 0.01, f"Too many outliers:\n{invalid}"

    assert df["trestbps"].dropna().between(80, 250).all()

