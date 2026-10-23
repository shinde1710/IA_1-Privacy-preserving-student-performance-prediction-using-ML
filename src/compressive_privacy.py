"""
src/compressive_privacy.py
--------------------------
Module 5: Compressive Privacy for Machine Learning.

Demonstrates the core conceptual and empirical principles of Compressive Privacy:
- Objective: Learn a representation that retains predictive utility for student performance (G3 >= 10)
  while actively suppressing information about a sensitive demographic attribute (e.g., 'sex').
- Comparison:
  1. Full Feature Space (Baseline representation)
  2. Standard PCA Representation (explains why standard dimensionality reduction is NOT privacy-preserving)
  3. Compressive Privacy Projection (orthogonal subspace projection removing sensitive correlation)
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


def evaluate_compressive_privacy(
    df: pd.DataFrame,
    sensitive_col: str = "sex",
    n_pca_components: int = 10,
    test_size: float = 0.2,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Evaluates target prediction utility vs sensitive attribute leakage across:
    (1) Full Feature Representation
    (2) Standard PCA Representation
    (3) Compressive Privacy (Orthogonal Projection) Representation
    """
    data = df.copy()
    y_target = (data['G3'] >= 10).astype(int)
    
    # Sensitive attribute: 1 for female ('F'), 0 for male ('M')
    s_sensitive = (data[sensitive_col] == 'F').astype(int)
    
    # Input features: remove target, exam grades, and explicitly remove sensitive_col
    cols_to_drop = ['G1', 'G2', 'G3', sensitive_col]
    X_raw = data.drop(columns=[c for c in cols_to_drop if c in data.columns])
    
    cat_cols = X_raw.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
    X_encoded = pd.get_dummies(X_raw, columns=cat_cols, drop_first=True, dtype=float)
    
    # Stratified train/test split
    X_tr, X_te, y_tr, y_te, s_tr, s_te = train_test_split(
        X_encoded, y_target, s_sensitive,
        test_size=test_size,
        random_state=random_state,
        stratify=y_target
    )
    
    majority_s_chance = float(max(s_te.mean(), 1.0 - s_te.mean()))
    
    # -------------------------------------------------------------
    # Representation 1: Full Features (Sensitive column dropped from inputs)
    # -------------------------------------------------------------
    clf_y_full = LogisticRegression(max_iter=1000, random_state=random_state)
    clf_y_full.fit(X_tr, y_tr)
    pred_y_full = clf_y_full.predict(X_te)
    acc_y_full = float(accuracy_score(y_te, pred_y_full))
    f1_y_full = float(f1_score(y_te, pred_y_full, zero_division=0))
    
    clf_s_full = LogisticRegression(max_iter=1000, random_state=random_state)
    clf_s_full.fit(X_tr, s_tr)
    pred_s_full = clf_s_full.predict(X_te)
    leak_full = float(accuracy_score(s_te, pred_s_full))
    
    # -------------------------------------------------------------
    # Representation 2: Standard PCA (Dimensionality Reduction)
    # -------------------------------------------------------------
    pca = PCA(n_components=n_pca_components, random_state=random_state)
    X_tr_pca = pca.fit_transform(X_tr)
    X_te_pca = pca.transform(X_te)
    
    clf_y_pca = LogisticRegression(max_iter=1000, random_state=random_state)
    clf_y_pca.fit(X_tr_pca, y_tr)
    pred_y_pca = clf_y_pca.predict(X_te_pca)
    acc_y_pca = float(accuracy_score(y_te, pred_y_pca))
    f1_y_pca = float(f1_score(y_te, pred_y_pca, zero_division=0))
    
    clf_s_pca = LogisticRegression(max_iter=1000, random_state=random_state)
    clf_s_pca.fit(X_tr_pca, s_tr)
    pred_s_pca = clf_s_pca.predict(X_te_pca)
    leak_pca = float(accuracy_score(s_te, pred_s_pca))
    
    # -------------------------------------------------------------
    # Representation 3: Compressive Privacy (Orthogonal Subspace Projection)
    # Projects feature space onto the nullspace of the sensitive attribute's direction
    # -------------------------------------------------------------
    v = clf_s_full.coef_[0]
    v_norm = np.linalg.norm(v)
    v_unit = v / v_norm if v_norm > 0 else v
    proj_matrix = np.eye(X_tr.shape[1]) - np.outer(v_unit, v_unit)
    
    X_tr_comp = X_tr.values @ proj_matrix
    X_te_comp = X_te.values @ proj_matrix
    
    clf_y_comp = LogisticRegression(max_iter=1000, random_state=random_state)
    clf_y_comp.fit(X_tr_comp, y_tr)
    pred_y_comp = clf_y_comp.predict(X_te_comp)
    acc_y_comp = float(accuracy_score(y_te, pred_y_comp))
    f1_y_comp = float(f1_score(y_te, pred_y_comp, zero_division=0))
    
    clf_s_comp = LogisticRegression(max_iter=1000, random_state=random_state)
    clf_s_comp.fit(X_tr_comp, s_tr)
    pred_s_comp = clf_s_comp.predict(X_te_comp)
    leak_comp = float(accuracy_score(s_te, pred_s_comp))
    
    records = [
        {
            "Representation": "1. Full Feature Space",
            "Target_Accuracy": acc_y_full,
            "Target_F1": f1_y_full,
            "Sensitive_Leakage_Accuracy": leak_full,
            "Baseline_Chance_Leakage": majority_s_chance,
            "Privacy_Protection": "None (Full correlation accessible)"
        },
        {
            "Representation": f"2. Standard PCA (k={n_pca_components})",
            "Target_Accuracy": acc_y_pca,
            "Target_F1": f1_y_pca,
            "Sensitive_Leakage_Accuracy": leak_pca,
            "Baseline_Chance_Leakage": majority_s_chance,
            "Privacy_Protection": "Inadequate (Preserves sensitive variance)"
        },
        {
            "Representation": "3. Compressive Privacy Projection",
            "Target_Accuracy": acc_y_comp,
            "Target_F1": f1_y_comp,
            "Sensitive_Leakage_Accuracy": leak_comp,
            "Baseline_Chance_Leakage": majority_s_chance,
            "Privacy_Protection": "Enhanced (Sensitive correlation suppressed)"
        }
    ]
    
    return pd.DataFrame(records)


if __name__ == "__main__":
    from data_preprocessing import load_dataset
    
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "student-mat.csv")
    df = load_dataset(data_path)
    print("--- Evaluating Compressive Privacy Representation Learning ---")
    comp_df = evaluate_compressive_privacy(df)
    print(comp_df.to_string(index=False))
