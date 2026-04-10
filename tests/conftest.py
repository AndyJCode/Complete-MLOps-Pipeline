import numpy as np
import pandas as pd
import pytest
from pathlib import Path

DATA_PATH = "data/heart_combined.csv"


@pytest.fixture(scope="session", autouse=True)
def ensure_data_file():
    """Generate synthetic heart disease data if the real file is absent (e.g. CI)."""
    path = Path(DATA_PATH)
    if path.exists():
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    n = 300
    df = pd.DataFrame({
        "age":      rng.integers(30, 80, n),
        "sex":      rng.integers(0, 2, n),
        "cp":       rng.integers(0, 4, n),
        "trestbps": rng.integers(90, 200, n),
        "chol":     rng.integers(150, 400, n),
        "fbs":      rng.integers(0, 2, n),
        "restecg":  rng.integers(0, 3, n),
        "thalach":  rng.integers(70, 200, n),
        "exang":    rng.integers(0, 2, n),
        "oldpeak":  rng.uniform(0, 6, n).round(1),
        "slope":    rng.integers(0, 3, n),
        "ca":       rng.integers(0, 4, n),
        "thal":     rng.integers(0, 4, n),
        "target":   rng.integers(0, 2, n),
    })
    df.to_csv(path, index=False)
