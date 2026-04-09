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
    assert df["chol"].between(100, 500).all()
    assert df["trestbps"].between(80, 250).all()
