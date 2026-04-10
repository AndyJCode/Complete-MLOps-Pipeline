import pandas as pd
import numpy as np
from typing import Union
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def preprocess_data(data: Union[str, pd.DataFrame] = 'data/heart_combined.csv', binary_target: bool = True):
    """Preprocess heart disease data for modeling.
    
    Args:
        data: Path to CSV file or DataFrame
        binary_target: Convert target to binary (0/1) if True
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, scaler)
        
    Raises:
        ValueError: If data is empty or missing required columns
    """
    # Load data if path provided
    if isinstance(data, str):
        df = pd.read_csv(data)
    else:
        df = data.copy()
    
    # Validate input
    if df.empty:
        raise ValueError("Input data is empty")
    
    required_cols = ['age', 'sex', 'cp', 'target']  # Minimum required
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Define column types
    numerical_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    
    # Handle missing values
    for col in numerical_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode().iloc[0])
    

    # In preprocessing
    df["chol"] = df["chol"].replace(0, np.nan)
    df = df[(df["chol"].isna()) | ((df["chol"] >= 0) & (df["chol"] <= 600))]
    df["trestbps"] = df["trestbps"].replace(0, np.nan)
    df = df[(df["trestbps"].isna()) | ((df["trestbps"] >= 0) & (df["trestbps"] <= 250))]
    #fill with median after removing outliers
    df["chol"] = df["chol"].fillna(df["chol"].median())
    df["trestbps"] = df["trestbps"].fillna(df["trestbps"].median())


    # Convert target to binary if specified (0 = no disease, 1-4 = disease)
    if binary_target:
        df['target'] = (df['target'] > 0).astype(int)
    
    # Separate features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Scale numerical features
    scaler = StandardScaler()
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    
    # Split into train and test sets
    stratify_param = y if len(df) >= 10 else None  # Only stratify if enough samples
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=12345, stratify=stratify_param
    )
    
    return X_train, X_test, y_train, y_test, scaler
