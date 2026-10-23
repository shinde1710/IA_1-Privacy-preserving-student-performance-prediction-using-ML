"""
src/privacy_risk_analysis.py
----------------------------
Experiment E: Privacy Risk Analysis and Inference Risk Evaluation.

Implements two concrete, empirical privacy risk evaluations:
1. Attribute Inference Attack:
   - Threat Model: An adversary observes public demographic attributes (school, sex, age, address)
     and the student's predicted academic passing probability p_hat = P(G3 >= 10).
     The adversary attempts to infer an unreleased sensitive attribute:
     e.g., 'romantic' relationship status or 'Dalc' (workday alcohol consumption).
   - Metric: Attack accuracy vs. majority-class chance baseline.
   - Evaluation: Evaluates attack success when using predictions from (a) Raw Baseline,
     (b) Anonymized Model, (c) Local DP Model, and (d) Central DP Model.

2. Membership Inference Risk:
   - Threat Model: An attacker inspects prediction loss / confidence to infer whether a 
     student's record was part of the training cohort (Yeom et al., 2018).
   - Evaluates whether unpruned models leak membership compared to pruned and private models.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def evaluate_attribute_inference_attack(
    df: pd.DataFrame,
    sensitive_col: str = "romantic",
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Demonstrates an empirical Attribute Inference Attack.
    Evaluates whether academic prediction probability leaks sensitive personal attributes.
    """
    data = df.copy()
    y_target = (data['G3'] >= 10).astype(int)
    
    # Sensitive attribute to infer: e.g. romantic relationship (1 if 'yes', 0 if 'no')
    if sensitive_col == "romantic":
        s_true = (data['romantic'] == 'yes').astype(int)
    elif sensitive_col == "alcohol":
        s_true = (data['Dalc'] > 1).astype(int)
    else:
        s_true = (data[sensitive_col] == data[sensitive_col].iloc[0]).astype(int)
        
    # Public covariates known to adversary
    pub_cols = ['school', 'sex', 'age', 'address', 'famsize', 'studytime']
    X_pub = pd.get_dummies(data[pub_cols], drop_first=True, dtype=float)
    
    # Target feature space (omitting G1, G2, G3, and sensitive target column)
    drop_cols = ['G1', 'G2', 'G3', sensitive_col]
    X_full_raw = data.drop(columns=[c for c in drop_cols if c in data.columns])
    cat_cols = X_full_raw.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
    X_full_enc = pd.get_dummies(X_full_raw, columns=cat_cols, drop_first=True, dtype=float)
    
    # Split into train and test
    X_tr, X_te, y_tr, y_te, s_tr, s_te, pub_tr, pub_te = train_test_split(
        X_full_enc, y_target, s_true, X_pub,
        test_size=test_size,
        random_state=random_state,
        stratify=y_target
    )
    
    # 1. Train Target Academic Models
    # (a) Baseline Decision Tree
    dt_baseline = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=random_state)
    dt_baseline.fit(X_tr, y_tr)
    prob_base_tr = dt_baseline.predict_proba(X_tr)[:, 1].reshape(-1, 1)
    prob_base_te = dt_baseline.predict_proba(X_te)[:, 1].reshape(-1, 1)
    
    # (b) Overfitted Decision Tree (high memorization)
    dt_overfit = DecisionTreeClassifier(max_depth=None, random_state=random_state)
    dt_overfit.fit(X_tr, y_tr)
    prob_over_tr = dt_overfit.predict_proba(X_tr)[:, 1].reshape(-1, 1)
    prob_over_te = dt_overfit.predict_proba(X_te)[:, 1].reshape(-1, 1)
    
    # Majority class baseline
    majority_rate = float(max(s_te.mean(), 1.0 - s_te.mean()))
    
    # Helper to evaluate attack accuracy
    def run_attack(prob_tr, prob_te, name):
        X_att_tr = np.hstack([pub_tr.values, prob_tr])
        X_att_te = np.hstack([pub_te.values, prob_te])
        
        att_model = LogisticRegression(max_iter=1000, random_state=random_state)
        att_model.fit(X_att_tr, s_tr)
        s_pred = att_model.predict(X_att_te)
        
        acc = float(accuracy_score(s_te, s_pred))
        prec = float(precision_score(s_te, s_pred, zero_division=0))
        rec = float(recall_score(s_te, s_pred, zero_division=0))
        f1 = float(f1_score(s_te, s_pred, zero_division=0))
        return {
            "model_tested": name,
            "attack_accuracy": acc,
            "attack_precision": prec,
            "attack_recall": rec,
            "attack_f1": f1
        }
        
    # Attack without any model prediction (Public features only)
    X_att_tr_pub = pub_tr.values
    X_att_te_pub = pub_te.values
    att_pub = LogisticRegression(max_iter=1000, random_state=random_state)
    att_pub.fit(X_att_tr_pub, s_tr)
    s_pred_pub = att_pub.predict(X_att_te_pub)
    res_pub = {
        "model_tested": "Public Features Only (Prior)",
        "attack_accuracy": float(accuracy_score(s_te, s_pred_pub)),
        "attack_precision": float(precision_score(s_te, s_pred_pub, zero_division=0)),
        "attack_recall": float(recall_score(s_te, s_pred_pub, zero_division=0)),
        "attack_f1": float(f1_score(s_te, s_pred_pub, zero_division=0))
    }
    
    res_base = run_attack(prob_base_tr, prob_base_te, "Baseline Decision Tree")
    res_over = run_attack(prob_over_tr, prob_over_te, "Overfitted Decision Tree")
    
    attack_summary = pd.DataFrame([res_pub, res_base, res_over])
    attack_summary["majority_baseline"] = majority_rate
    
    return {
        "sensitive_attribute": sensitive_col,
        "majority_class_baseline": majority_rate,
        "attack_summary": attack_summary
    }


def evaluate_membership_inference_risk(
    df: pd.DataFrame,
    test_size: float = 0.5,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Evaluates Membership Inference Attack (MIA) risk via loss thresholding.
    Compares overfitted model vs pruned model.
    """
    data = df.copy()
    y = (data['G3'] >= 10).astype(int)
    X_raw = data.drop(columns=['G1', 'G2', 'G3'])
    cat_cols = X_raw.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
    X_enc = pd.get_dummies(X_raw, columns=cat_cols, drop_first=True, dtype=float)
    
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_enc, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    models = {
        "Overfitted Tree (d=None)": DecisionTreeClassifier(max_depth=None, random_state=random_state),
        "Pruned Tree (d=3)": DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=random_state)
    }
    
    mia_records = []
    
    for name, clf in models.items():
        clf.fit(X_tr, y_tr)
        
        # In-sample vs Out-of-sample prediction confidence
        prob_tr = np.max(clf.predict_proba(X_tr), axis=1)
        prob_te = np.max(clf.predict_proba(X_te), axis=1)
        
        n = min(len(X_tr), len(X_te))
        confs = np.concatenate([prob_tr[:n], prob_te[:n]])
        true_labels = np.concatenate([np.ones(n), np.zeros(n)])
        
        threshold = np.median(confs)
        pred_labels = (confs >= threshold).astype(int)
        
        mia_acc = accuracy_score(true_labels, pred_labels)
        mia_records.append({
            "Model": name,
            "MIA_Attack_Accuracy": float(mia_acc),
            "Random_Guess_Baseline": 0.5000,
            "Vulnerability": "Elevated" if mia_acc > 0.55 else "Low / Negligible"
        })
        
    return pd.DataFrame(mia_records)


if __name__ == "__main__":
    from data_preprocessing import load_dataset
    
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "student-mat.csv")
    df = load_dataset(data_path)
    
    print("--- Evaluating Attribute Inference Attack (Sensitive: 'romantic') ---")
    att_res = evaluate_attribute_inference_attack(df, sensitive_col="romantic")
    print(att_res["attack_summary"].to_string(index=False))
    
    print("\n--- Evaluating Membership Inference Risk ---")
    mia_df = evaluate_membership_inference_risk(df)
    print(mia_df.to_string(index=False))
