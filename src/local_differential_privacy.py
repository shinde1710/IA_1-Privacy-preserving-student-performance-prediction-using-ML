"""
src/local_differential_privacy.py
---------------------------------
Experiment C: Local Differential Privacy (LDP) implementation on student study time.

Mathematical Foundation:
- Target Feature: 'studytime' (weekly study hours category)
- Valid Range [a, b]: [1, 4]
- Global Sensitivity: Delta_f = b - a = 4 - 1 = 3
- Mechanism: Client-side Laplace Mechanism with Post-Processing Clipping
    Z_i ~ Laplace(0, Delta_f / epsilon)
    x_priv = clip(x_i + Z_i, 1.0, 4.0)
- Formal Guarantee:
    Satisfies epsilon-Local Differential Privacy (LDP) prior to post-processing.
    By the Post-Processing Theorem (Dwork & Roth, 2014), clipping the perturbed 
    value back into the valid domain [1, 4] preserves epsilon-LDP unconditionally.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error
from baseline_model import evaluate_classifier


FEATURE_NAME = "studytime"
FEATURE_MIN = 1.0
FEATURE_MAX = 4.0
SENSITIVITY = FEATURE_MAX - FEATURE_MIN  # 3.0


def apply_laplace_ldp(
    values: pd.Series,
    epsilon: float,
    sensitivity: float = SENSITIVITY,
    clip_min: float = FEATURE_MIN,
    clip_max: float = FEATURE_MAX,
    random_state: int = 42
) -> Tuple[np.ndarray, float, float]:
    """
    Applies mathematically rigorous client-side Laplace perturbation under Local DP.
    
    Parameters:
    -----------
    values : pd.Series
        Original raw attribute values.
    epsilon : float
        Privacy budget parameter (lower = stronger privacy, higher noise).
    sensitivity : float
        L1 sensitivity of the attribute (b - a).
    clip_min : float
        Lower bound of attribute domain.
    clip_max : float
        Upper bound of attribute domain.
    random_state : int
        Seed for reproducible noise generation.
        
    Returns:
    --------
    perturbed_values, mae_distortion, rmse_distortion
    """
    if epsilon <= 0:
        raise ValueError(f"Privacy budget epsilon must be strictly positive, got {epsilon}")
        
    rng = np.random.default_rng(random_state)
    scale = sensitivity / epsilon
    
    # Generate independent Laplace noise for each student
    noise = rng.laplace(loc=0.0, scale=scale, size=len(values))
    
    # Add noise locally (before transmission)
    noisy_raw = values.to_numpy() + noise
    
    # Post-processing: clip to bounded domain [clip_min, clip_max]
    # Post-processing preserves epsilon-LDP by Dwork & Roth (2014)
    perturbed = np.clip(noisy_raw, clip_min, clip_max)
    
    mae = float(mean_absolute_error(values, perturbed))
    rmse = float(np.sqrt(mean_squared_error(values, perturbed)))
    
    return perturbed, mae, rmse


def run_ldp_experiments(
    df: pd.DataFrame,
    epsilons: List[float] = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
    test_size: float = 0.2,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Executes LDP experiments across a range of privacy budgets.
    Measures feature distortion and downstream Decision Tree utility.
    """
    data = df.copy()
    y = (data['G3'] >= 10).astype(int)
    
    # Exclude target and earlier grades to prevent leakage
    cols_to_drop = ['G3', 'G1', 'G2']
    X_base = data.drop(columns=[c for c in cols_to_drop if c in data.columns])
    
    results = []
    
    for eps in epsilons:
        perturbed_feature, mae, rmse = apply_laplace_ldp(
            X_base[FEATURE_NAME],
            epsilon=eps,
            random_state=random_state
        )
        
        # Inject locally perturbed feature into feature set
        X_perturbed = X_base.copy()
        X_perturbed[FEATURE_NAME] = perturbed_feature
        
        # Encode categoricals
        categorical_cols = X_perturbed.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
        X_encoded = pd.get_dummies(X_perturbed, columns=categorical_cols, drop_first=True, dtype=int)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )
        
        # Train downstream Decision Tree
        dt = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=random_state)
        dt.fit(X_train, y_train)
        
        metrics = evaluate_classifier(dt, X_test, y_test, model_name=f"LDP DT (eps={eps})")
        
        results.append({
            "epsilon": eps,
            "privacy_level": "Strong" if eps < 1.0 else ("Moderate" if eps <= 2.0 else "Weak"),
            "mae_distortion": mae,
            "rmse_distortion": rmse,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "macro_f1": metrics["macro_f1"]
        })
        
    df_results = pd.DataFrame(results)
    return df_results


if __name__ == "__main__":
    from data_preprocessing import load_dataset
    
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "student-mat.csv")
    df = load_dataset(data_path)
    
    print("--- Running Local Differential Privacy Experiments on 'studytime' ---")
    ldp_df = run_ldp_experiments(df)
    print(ldp_df.to_string(index=False))
