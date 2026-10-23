"""
tests/test_preprocessing.py
---------------------------
Unit tests for data preprocessing and leakage prevention.
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from data_preprocessing import load_dataset, inspect_dataset_summary, preprocess_student_data


@pytest.fixture
def sample_raw_data():
    """Returns sample raw dataframe simulating UCI student performance format."""
    data = {
        "school": ["GP", "GP", "MS", "MS", "GP"],
        "sex": ["F", "M", "F", "M", "F"],
        "age": [15, 17, 16, 18, 15],
        "address": ["U", "R", "U", "R", "U"],
        "famsize": ["GT3", "LE3", "GT3", "LE3", "GT3"],
        "Pstatus": ["T", "A", "T", "T", "A"],
        "Medu": [4, 1, 2, 3, 4],
        "Fedu": [4, 2, 2, 3, 2],
        "studytime": [2, 1, 3, 2, 4],
        "failures": [0, 1, 0, 2, 0],
        "absences": [4, 10, 2, 16, 0],
        "G1": [12, 8, 14, 7, 15],
        "G2": [13, 7, 15, 6, 16],
        "G3": [14, 8, 15, 5, 17]  # Targets: 1, 0, 1, 0, 1
    }
    return pd.DataFrame(data)


def test_leakage_prevention(sample_raw_data):
    """Verify that G1, G2, and G3 are strictly excluded from features."""
    X_train, X_test, y_train, y_test, f_names = preprocess_student_data(
        sample_raw_data, exclude_earlier_grades=True, test_size=0.4, random_state=42
    )
    
    assert "G3" not in X_train.columns
    assert "G1" not in X_train.columns
    assert "G2" not in X_train.columns
    assert "G3" not in X_test.columns
    assert "G1" not in X_test.columns
    assert "G2" not in X_test.columns
    assert "G3" not in f_names


def test_target_binarization(sample_raw_data):
    """Verify that target G3 >= 10 is mapped to 1, and G3 < 10 to 0."""
    X_train, X_test, y_train, y_test, _ = preprocess_student_data(
        sample_raw_data, test_size=0.4, random_state=42
    )
    
    assert set(y_train.unique()).issubset({0, 1})
    assert set(y_test.unique()).issubset({0, 1})
    assert len(y_train) + len(y_test) == len(sample_raw_data)


def test_real_dataset_loading():
    """Verify that the official UCI dataset loads cleanly without nulls."""
    csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "student-mat.csv"))
    if os.path.exists(csv_path):
        df = load_dataset(csv_path)
        assert len(df) == 395
        assert df.isnull().sum().sum() == 0
