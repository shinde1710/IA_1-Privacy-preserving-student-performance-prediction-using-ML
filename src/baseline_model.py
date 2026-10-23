"""
src/baseline_model.py
---------------------
Baseline Machine Learning Model training and evaluation.
Primary model: Decision Tree Classifier (transparent, explainable for viva).
Secondary baseline: Logistic Regression (linear reference).
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
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
    # cm layout: [[TN, FP], [FN, TP]]
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
    
    dt = train_decision_tree(X_tr, y_tr)
    metrics_dt = evaluate_classifier(dt, X_te, y_te, "Baseline Decision Tree")
    print("Decision Tree Metrics:", metrics_dt)
    
    lr = train_logistic_regression(X_tr, y_tr)
    metrics_lr = evaluate_classifier(lr, X_te, y_te, "Baseline Logistic Regression")
    print("Logistic Regression Metrics:", metrics_lr)
    
    imp = get_feature_importances(dt, f_names)
    print("\nTop 5 Important Features:\n", imp.head(5))
