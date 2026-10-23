"""
tests/test_central_dp.py
------------------------
Unit tests for Central Differential Privacy implementation.
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from central_differential_privacy import CentralDPLogisticRegression


def test_central_dp_prediction_bounds():
    """Verify that Central DP Logistic Regression outputs valid class predictions."""
    X = np.random.randn(50, 5)
    y = np.random.randint(0, 2, size=50)
    
    clf = CentralDPLogisticRegression(epsilon=1.0, C=1.0, random_state=42)
    clf.fit(X, y)
    
    preds = clf.predict(X)
    probs = clf.predict_proba(X)
    
    assert set(np.unique(preds)).issubset({0, 1})
    assert probs.shape == (50, 2)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
    assert np.allclose(np.sum(probs, axis=1), 1.0)


def test_central_dp_noise_injection():
    """Verify that different random seeds produce different perturbed weights."""
    X = np.random.randn(60, 4)
    y = np.random.randint(0, 2, size=60)
    
    clf1 = CentralDPLogisticRegression(epsilon=0.5, C=1.0, random_state=42).fit(X, y)
    clf2 = CentralDPLogisticRegression(epsilon=0.5, C=1.0, random_state=99).fit(X, y)
    
    assert not np.allclose(clf1.coef_, clf2.coef_)
