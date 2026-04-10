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
    n = 500

    age      = rng.integers(30, 80, n).astype(float)
    sex      = rng.integers(0, 2, n).astype(float)
    cp       = rng.integers(0, 4, n).astype(float)
    trestbps = rng.integers(90, 200, n).astype(float)
    chol     = rng.integers(150, 400, n).astype(float)
    fbs      = rng.integers(0, 2, n).astype(float)
    restecg  = rng.integers(0, 3, n).astype(float)
    thalach  = rng.integers(70, 200, n).astype(float)
    exang    = rng.integers(0, 2, n).astype(float)
    oldpeak  = rng.uniform(0, 6, n).round(1)
    slope    = rng.integers(0, 3, n).astype(float)
    ca       = rng.integers(0, 4, n).astype(float)
    thal     = rng.integers(0, 4, n).astype(float)

    # Derive target from a simple rule so the model can learn a real signal
    signal = (age > 55).astype(int) + (cp > 1).astype(int) + (exang == 1).astype(int) + (ca > 1).astype(int)
    target = (signal >= 2).astype(int)

    df = pd.DataFrame({
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps,
        "chol": chol, "fbs": fbs, "restecg": restecg, "thalach": thalach,
        "exang": exang, "oldpeak": oldpeak, "slope": slope, "ca": ca,
        "thal": thal, "target": target,
    })
    df.to_csv(path, index=False)
