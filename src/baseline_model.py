"""
src/baseline_model.py
---------------------
Baseline Machine Learning Model training, cross-validation, and evaluation.
Primary model: Decision Tree Classifier (transparent, explainable for viva).
Secondary baseline: Logistic Regression (linear reference).
Supports:
- Train/test single-split evaluation
- 5-Fold Stratified Cross-Validation (mean +/- std)
- Feature importance extraction
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


def train_decision_tree(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    max_depth: int = 3,
    min_samples_leaf: int = 5,
    random_state: int = 42
) -> DecisionTreeClassifier:
    """
    Trains a pruned Decision Tree classifier.
    
    A controlled depth (max_depth=3) prevents overfitting on tabular educational data
    and produces an easily interpretable decision structure suitable for academic viva.
    """
    dt = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state
    )
    dt.fit(X_train, y_train)
    return dt


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    max_iter: int = 1000,
    random_state: int = 42
) -> LogisticRegression:
    """
    Trains a baseline Logistic Regression model for linear comparison.
    """
    lr = LogisticRegression(
        max_iter=max_iter,
        random_state=random_state
    )
    lr.fit(X_train, y_train)
    return lr


def evaluate_classifier(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "Model"
) -> Dict[str, Any]:
    """
    Evaluates a trained classifier across key standard performance metrics:
    Accuracy, Precision, Recall, F1-score, and Confusion Matrix.
    """
    y_pred = model.predict(X_test)
    
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    metrics = {
        "model_name": model_name,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "confusion_matrix": cm.tolist()
    }
    
    return metrics


def evaluate_with_cross_validation(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Performs Stratified K-Fold Cross Validation for both Decision Tree and Logistic Regression.
    Reports mean and standard deviation to assess baseline stability on small datasets.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    models = {
        "Decision Tree (d=3)": DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=random_state),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=random_state)
    }
    
    cv_records = []
    
    for name, clf in models.items():
        accs, precs, recs, f1s = [], [], [], []
        
        for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
            X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
            
            clf.fit(X_tr, y_tr)
            y_pred = clf.predict(X_val)
            
            accs.append(accuracy_score(y_val, y_pred))
            precs.append(precision_score(y_val, y_pred, zero_division=0))
            recs.append(recall_score(y_val, y_pred, zero_division=0))
            f1s.append(f1_score(y_val, y_pred, zero_division=0))
            
        cv_records.append({
            "Model": name,
            "Accuracy_Mean": float(np.mean(accs)),
            "Accuracy_Std": float(np.std(accs)),
            "Precision_Mean": float(np.mean(precs)),
            "Precision_Std": float(np.std(precs)),
            "Recall_Mean": float(np.mean(recs)),
            "Recall_Std": float(np.std(recs)),
            "F1_Mean": float(np.mean(f1s)),
            "F1_Std": float(np.std(f1s))
        })
        
    return pd.DataFrame(cv_records)


def get_feature_importances(
    dt_model: DecisionTreeClassifier,
    feature_names: list
) -> pd.DataFrame:
    """
    Extracts and ranks feature importances from the Decision Tree model.
    """
    df_imp = pd.DataFrame({
        "feature": feature_names,
        "importance": dt_model.feature_importances_
    }).sort_values(by="importance", ascending=False).reset_index(drop=True)
    
    return df_imp


if __name__ == "__main__":
    from data_preprocessing import load_dataset, preprocess_student_data
    
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "student-mat.csv")
    df = load_dataset(data_path)
    X_tr, X_te, y_tr, y_te, f_names = preprocess_student_data(df)
    
    # Single split evaluation
    dt = train_decision_tree(X_tr, y_tr)
    metrics_dt = evaluate_classifier(dt, X_te, y_te, "Baseline Decision Tree")
    print("Decision Tree Metrics (Single Split):", metrics_dt)
    
    # Cross Validation evaluation
    y_full = (df['G3'] >= 10).astype(int)
    X_full_raw = df.drop(columns=['G1', 'G2', 'G3'])
    cat_cols = X_full_raw.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
    X_full_enc = pd.get_dummies(X_full_raw, columns=cat_cols, drop_first=True, dtype=int)
    
    cv_df = evaluate_with_cross_validation(X_full_enc, y_full, n_splits=5)
    print("\n=== 5-Fold Stratified Cross Validation Results ===")
    print(cv_df.to_string(index=False))
