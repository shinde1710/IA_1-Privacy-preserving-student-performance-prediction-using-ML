"""
src/data_preprocessing.py
--------------------------
Data preprocessing module for Student Performance dataset.
Handles:
- Dataset loading and validation
- Target construction (G3 >= 10 -> Pass (1), Fail (0))
- Leakage prevention (strictly drops G1, G2, G3 from features)
- Categorical encoding and feature standardization
- Stratified train/test splitting
"""

import os
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, List
from sklearn.model_selection import train_test_split


def load_dataset(csv_path: str) -> pd.DataFrame:
    """
    Loads student performance dataset from CSV.
    The UCI student dataset typically uses semicolon ';' delimiter.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset file not found at: {csv_path}")
    
    # Try semicolon delimiter first (official UCI format), fallback to comma
    try:
        df = pd.read_csv(csv_path, sep=';')
        if df.shape[1] == 1:
            df = pd.read_csv(csv_path, sep=',')
    except Exception:
        df = pd.read_csv(csv_path, sep=',')
        
    return df


def inspect_dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculates essential descriptive statistics for documentation.
    """
    summary = {
        "num_rows": int(df.shape[0]),
        "num_cols": int(df.shape[1]),
        "missing_values": int(df.isnull().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
        "pass_count": int((df['G3'] >= 10).sum()) if 'G3' in df.columns else None,
        "fail_count": int((df['G3'] < 10).sum()) if 'G3' in df.columns else None,
        "columns": list(df.columns)
    }
    return summary


def preprocess_student_data(
    df: pd.DataFrame,
    exclude_earlier_grades: bool = True,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str]]:
    """
    Prepares student performance data for machine learning while avoiding target leakage.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw student dataset.
    exclude_earlier_grades : bool, default=True
        If True, drops G1 (period 1 grade) and G2 (period 2 grade) in addition to G3.
        In realistic educational intervention, predictions must be made early before
        exam grades are known. Including G1/G2 causes severe target leakage.
    test_size : float, default=0.2
        Fraction of data reserved for testing.
    random_state : int, default=42
        Reproducibility seed.
        
    Returns:
    --------
    X_train, X_test, y_train, y_test, feature_names
    """
    data = df.copy()
    
    # Binary target definition: Pass (1) if G3 >= 10, Fail (0) if G3 < 10
    if 'G3' not in data.columns:
        raise ValueError("Target column 'G3' not found in dataset.")
        
    y = (data['G3'] >= 10).astype(int)
    
    # Drop target and leaky features
    cols_to_drop = ['G3']
    if exclude_earlier_grades:
        cols_to_drop.extend(['G1', 'G2'])
        
    X_raw = data.drop(columns=[c for c in cols_to_drop if c in data.columns])
    
    # Identify categorical and numeric columns
    categorical_cols = X_raw.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
    numeric_cols = X_raw.select_dtypes(include=[np.number]).columns.tolist()
    
    # One-hot encode categorical features (drop_first=True to avoid collinearity)
    X_encoded = pd.get_dummies(X_raw, columns=categorical_cols, drop_first=True, dtype=int)
    feature_names = list(X_encoded.columns)
    
    # Stratified split to maintain pass/fail ratio across train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    return X_train, X_test, y_train, y_test, feature_names


if __name__ == "__main__":
    import os
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "student-mat.csv")
    df = load_dataset(data_path)
    summary = inspect_dataset_summary(df)
    print("Dataset Summary:", summary)
    X_tr, X_te, y_tr, y_te, f_names = preprocess_student_data(df)
    print(f"X_train shape: {X_tr.shape}, X_test shape: {X_te.shape}")
    print(f"Number of features after encoding: {len(f_names)}")
