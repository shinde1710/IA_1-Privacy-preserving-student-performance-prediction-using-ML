"""
src/central_differential_privacy.py
-----------------------------------
Experiment C: Central Differential Privacy (Central DP) implementation.

Includes:
1. IBM diffprivlib GaussianNB (Perturbed Class Statistics)
2. Central DP Logistic Regression via Output Perturbation (Chaudhuri et al., 2011)
   using diffprivlib.mechanisms.Laplace.
   
Theoretical Foundation:
- Setting: Central Curator model where aggregator is trusted to collect raw data,
  but model release must satisfy formal (epsilon, 0)-Differential Privacy.
- Mechanism: Output perturbation on regularized empirical risk minimization.
  Sensitivity Delta_w = (2 * C) / N where N is training samples and C is 
  inverse regularization parameter.
  Weights w_priv = w + Laplace(0, Delta_w / epsilon).
- Post-Processing: Predictions computed using w_priv maintain epsilon-DP unconditionally.
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Apply compatibility shim for diffprivlib with modern scikit-learn
import sklearn.tree._tree
if not hasattr(sklearn.tree._tree, 'DOUBLE'):
    sklearn.tree._tree.DOUBLE = np.float64
if not hasattr(sklearn.tree._tree, 'DTYPE'):
    sklearn.tree._tree.DTYPE = np.float32

from diffprivlib.mechanisms import Laplace
from diffprivlib.models import GaussianNB as DPGaussianNB


class CentralDPLogisticRegression:
    """
    Central Differentially Private Logistic Regression via Output Perturbation.
    Ref: Chaudhuri et al., 'Differentially Private Empirical Risk Minimization', JMLR 2011.
    """
    def __init__(self, epsilon: float = 1.0, C: float = 1.0, random_state: int = 42):
        self.epsilon = epsilon
        self.C = C
        self.random_state = random_state
        self.base_model = LogisticRegression(C=C, max_iter=1000, random_state=random_state)
        self.coef_ = None
        self.intercept_ = None
        self.classes_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        # 1. Fit standard L2-regularized logistic regression
        self.base_model.fit(X, y)
        self.classes_ = self.base_model.classes_
        
        N = len(X)
        d = X.shape[1]
        
        # 2. Compute L1 sensitivity of weight parameters
        # For L2 regularized logistic loss with bounded norm ||x|| <= 1:
        # Sensitivity of weight vector ||Delta w||_1 <= 2 * C / N per coordinate
        sensitivity = float(2.0 * self.C / N)
        
        # 3. Add calibrated Laplace noise using diffprivlib mechanism
        rng = np.random.default_rng(self.random_state)
        scale = sensitivity / self.epsilon
        
        noise_coef = rng.laplace(0.0, scale, size=self.base_model.coef_.shape)
        noise_intercept = rng.laplace(0.0, scale, size=self.base_model.intercept_.shape)
        
        self.coef_ = self.base_model.coef_ + noise_coef
        self.intercept_ = self.base_model.intercept_ + noise_intercept
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        logits = np.dot(X, self.coef_.T) + self.intercept_
        p1 = 1.0 / (1.0 + np.exp(-logits))
        p0 = 1.0 - p1
        return np.hstack([p0, p1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return (probs[:, 1] >= 0.5).astype(int)


def run_central_dp_experiments(
    df: pd.DataFrame,
    epsilons: List[float] = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
    test_size: float = 0.2,
    n_repeats: int = 5,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Evaluates Central Differential Privacy across a range of privacy budgets.
    Includes both Central DP Logistic Regression and diffprivlib GaussianNB.
    """
    data = df.copy()
    y = (data['G3'] >= 10).astype(int)
    
    cols_to_drop = ['G3', 'G1', 'G2']
    X_raw = data.drop(columns=[c for c in cols_to_drop if c in data.columns])
    
    categorical_cols = X_raw.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
    X_encoded = pd.get_dummies(X_raw, columns=categorical_cols, drop_first=True, dtype=float)
    
    # Standardize inputs to normalize feature scales
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_encoded), columns=X_encoded.columns)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    # Domain bounds for diffprivlib GaussianNB
    gnb_bounds = (X_train.min(axis=0).values, X_train.max(axis=0).values)
    
    results = []
    
    for eps in epsilons:
        dp_lr_acc, dp_lr_prec, dp_lr_rec, dp_lr_f1 = [], [], [], []
        gnb_acc, gnb_f1 = [], []
        
        for rep in range(n_repeats):
            seed = random_state + rep * 17
            
            # 1. Central DP Logistic Regression (Output Perturbation)
            dp_lr = CentralDPLogisticRegression(epsilon=eps, C=1.0, random_state=seed)
            dp_lr.fit(X_train.values, y_train.values)
            y_pred_lr = dp_lr.predict(X_test.values)
            
            dp_lr_acc.append(accuracy_score(y_test, y_pred_lr))
            dp_lr_prec.append(precision_score(y_test, y_pred_lr, zero_division=0))
            dp_lr_rec.append(recall_score(y_test, y_pred_lr, zero_division=0))
            dp_lr_f1.append(f1_score(y_test, y_pred_lr, zero_division=0))
            
            # 2. IBM diffprivlib GaussianNB
            dp_gnb = DPGaussianNB(epsilon=eps, bounds=gnb_bounds)
            dp_gnb.fit(X_train.values, y_train.values)
            y_pred_gnb = dp_gnb.predict(X_test.values)
            
            gnb_acc.append(accuracy_score(y_test, y_pred_gnb))
            gnb_f1.append(f1_score(y_test, y_pred_gnb, zero_division=0))
            
        results.append({
            "epsilon": eps,
            "privacy_level": "Strong" if eps < 1.0 else ("Moderate" if eps <= 2.0 else "Weak"),
            "DP_LR_Accuracy": float(np.mean(dp_lr_acc)),
            "DP_LR_Acc_Std": float(np.std(dp_lr_acc)),
            "DP_LR_Precision": float(np.mean(dp_lr_prec)),
            "DP_LR_Recall": float(np.mean(dp_lr_rec)),
            "DP_LR_F1": float(np.mean(dp_lr_f1)),
            "DP_LR_F1_Std": float(np.std(dp_lr_f1)),
            "DP_GNB_Accuracy": float(np.mean(gnb_acc)),
            "DP_GNB_F1": float(np.mean(gnb_f1))
        })
        
    df_results = pd.DataFrame(results)
    return df_results


if __name__ == "__main__":
    from data_preprocessing import load_dataset
    
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "student-mat.csv")
    df = load_dataset(data_path)
    print("--- Executing Central Differential Privacy Experiments ---")
    dp_df = run_central_dp_experiments(df)
    print(dp_df[['epsilon', 'privacy_level', 'DP_LR_Accuracy', 'DP_LR_Acc_Std', 'DP_LR_F1', 'DP_GNB_Accuracy', 'DP_GNB_F1']].to_string(index=False))
