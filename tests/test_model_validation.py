import pandas as pd
from src.preprocessing import preprocess_data
from src.train import build_model


def test_model_predict_returns_binary_labels():
    X_train, X_test, y_train, y_test, _ = preprocess_data("data/raw/heart_combined.csv")
    model = build_model({
        "model_type": "logistic_regression",
        "lr_C": 1.0,
        "random_state": 12345,
    })
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    assert predictions.shape == (len(X_test),)
    assert set(predictions).issubset({0, 1})


def test_random_forest_model_meets_minimum_accuracy():
    X_train, X_test, y_train, y_test, _ = preprocess_data("data/raw/heart_combined.csv")
    model = build_model({
        "model_type": "random_forest",
        "rf_n_estimators": 50,
        "rf_max_depth": 5,
        "random_state": 12345,
    })
    model.fit(X_train, y_train)
    accuracy = (model.predict(X_test) == y_test).mean()

    assert accuracy >= 0.70
