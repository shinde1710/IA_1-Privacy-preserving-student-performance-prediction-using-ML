"""
tests/test_risk_analysis.py
---------------------------
Unit tests for Privacy Risk Analysis and Inference Risk evaluations.
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from privacy_risk_analysis import evaluate_attribute_inference_attack, evaluate_membership_inference_risk
from compressive_privacy import evaluate_compressive_privacy


@pytest.fixture
def mock_student_dataset():
    """Generates synthetic student dataset for fast unit test execution."""
    rng = np.random.default_rng(42)
    n = 60
    return pd.DataFrame({
        "school": ["GP"] * 30 + ["MS"] * 30,
        "sex": ["F", "M"] * 30,
        "age": rng.integers(15, 20, size=n),
        "address": ["U", "R"] * 30,
        "famsize": ["GT3", "LE3"] * 30,
        "Pstatus": ["T"] * 50 + ["A"] * 10,
        "Medu": rng.integers(0, 5, size=n),
        "Fedu": rng.integers(0, 5, size=n),
        "studytime": rng.integers(1, 5, size=n),
        "failures": rng.integers(0, 3, size=n),
        "romantic": ["no", "yes"] * 30,
        "Dalc": rng.integers(1, 4, size=n),
        "absences": rng.integers(0, 20, size=n),
        "G1": rng.integers(5, 18, size=n),
        "G2": rng.integers(5, 18, size=n),
        "G3": rng.integers(5, 18, size=n)
    })


def test_attribute_inference_attack(mock_student_dataset):
    """Verify attribute inference attack returns structured dataframe with valid accuracy."""
    res = evaluate_attribute_inference_attack(mock_student_dataset, sensitive_col="romantic", test_size=0.3)
    summary = res["attack_summary"]
    
    assert "attack_accuracy" in summary.columns
    assert np.all(summary["attack_accuracy"] >= 0.0)
    assert np.all(summary["attack_accuracy"] <= 1.0)
    assert len(summary) >= 3


def test_membership_inference_risk(mock_student_dataset):
    """Verify membership inference returns bounded accuracy."""
    mia_df = evaluate_membership_inference_risk(mock_student_dataset, test_size=0.5)
    
    assert "MIA_Attack_Accuracy" in mia_df.columns
    assert np.all(mia_df["MIA_Attack_Accuracy"] >= 0.0)
    assert np.all(mia_df["MIA_Attack_Accuracy"] <= 1.0)


def test_compressive_privacy(mock_student_dataset):
    """Verify compressive privacy evaluation runs and computes target and leakage metrics."""
    comp_df = evaluate_compressive_privacy(mock_student_dataset, sensitive_col="sex", n_pca_components=4, test_size=0.3)
    
    assert "Target_Accuracy" in comp_df.columns
    assert "Sensitive_Leakage_Accuracy" in comp_df.columns
    assert len(comp_df) == 3
