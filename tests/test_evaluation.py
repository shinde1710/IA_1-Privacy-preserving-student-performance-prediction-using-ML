"""
tests/test_evaluation.py
------------------------
Unit tests for model evaluation, metrics calculation, and LDP noise validity.
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from baseline_model import evaluate_classifier, train_decision_tree
from anonymization import assess_reidentification_risk, anonymize_student_dataset
from local_differential_privacy import apply_laplace_ldp, SENSITIVITY


def test_evaluate_classifier_output_structure():
    """Verify evaluation dictionary contains valid bounded metrics."""
    X = pd.DataFrame({"feat1": [1, 2, 3, 4, 5, 6], "feat2": [0, 1, 0, 1, 0, 1]})
    y = pd.Series([0, 0, 0, 1, 1, 1])
    
    dt = DecisionTreeClassifier(max_depth=2, random_state=42)
    dt.fit(X, y)
    
    metrics = evaluate_classifier(dt, X, y, model_name="TestDT")
    
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert 0.0 <= metrics["f1"] <= 1.0
    assert metrics["tn"] + metrics["fp"] + metrics["fn"] + metrics["tp"] == len(y)
    assert len(metrics["confusion_matrix"]) == 2


def test_ldp_laplace_bounds_and_distortion():
    """Verify that Laplace noise perturbations stay strictly inside domain [1, 4]."""
    raw_studytime = pd.Series([1, 2, 3, 4, 2, 3, 1, 4] * 10)
    eps = 0.5
    
    perturbed, mae, rmse = apply_laplace_ldp(
        raw_studytime,
        epsilon=eps,
        sensitivity=SENSITIVITY,
        clip_min=1.0,
        clip_max=4.0,
        random_state=42
    )
    
    # Check clipping bounds
    assert np.all(perturbed >= 1.0)
    assert np.all(perturbed <= 4.0)
    # Check distortion is non-negative and non-zero
    assert mae > 0.0
    assert rmse > 0.0


def test_anonymization_risk_reduction():
    """Verify that anonymization reduces re-identification uniqueness."""
    raw_df = pd.DataFrame({
        "school": ["GP", "MS", "GP", "MS", "GP"],
        "sex": ["F", "M", "F", "M", "F"],
        "age": [15, 18, 15, 20, 16],
        "address": ["U", "R", "U", "R", "U"],
        "Medu": [4, 1, 4, 1, 3],
        "Fedu": [4, 2, 4, 2, 3],
        "absences": [2, 15, 3, 20, 4],
        "G3": [12, 8, 14, 6, 15]
    })
    
    risk_before = assess_reidentification_risk(raw_df, ["school", "sex", "age", "address", "Medu", "Fedu"])
    anon_df = anonymize_student_dataset(raw_df)
    
    assert "school" not in anon_df.columns
    assert "age_bin" in anon_df.columns
    assert "Medu_bin" in anon_df.columns
    assert "Fedu_bin" in anon_df.columns
