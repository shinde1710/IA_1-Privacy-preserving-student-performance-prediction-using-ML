"""
src/anonymization.py
--------------------
Experiment B: Data Anonymization, Quasi-Identifier Analysis, and k-Anonymity Assessment.

Implements:
- Identification of Quasi-Identifiers (QIs)
- Quantitative re-identification risk assessment (uniqueness rate & equivalence classes)
- Generalization and suppression transformations
- Downstream machine learning utility comparison
- Rigorous discussion of k-anonymity limitations (homogeneity & background knowledge attacks)
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from baseline_model import evaluate_classifier


DEFAULT_QUASI_IDENTIFIERS = [
    'school', 'sex', 'age', 'address', 'Medu', 'Fedu'
]


def assess_reidentification_risk(
    df: pd.DataFrame,
    quasi_identifiers: List[str]
) -> Dict[str, Any]:
    """
    Measures re-identification vulnerability over a specified set of quasi-identifiers.
    
    Computes:
    - Total equivalence classes
    - Minimum equivalence class size (empirical k)
    - Number and percentage of unique individuals (k=1)
    - Median equivalence class size
    """
    existing_qis = [c for c in quasi_identifiers if c in df.columns]
    if not existing_qis:
        raise ValueError(f"None of the quasi-identifiers {quasi_identifiers} exist in the DataFrame.")
        
    grouped = df.groupby(existing_qis, observed=False).size()
    # Filter classes that have actual student counts
    active_groups = grouped[grouped > 0]
    
    unique_count = int((active_groups == 1).sum())
    total_records = len(df)
    min_k = int(active_groups.min())
    median_k = float(active_groups.median())
    
    return {
        "quasi_identifiers": existing_qis,
        "num_records": total_records,
        "num_equivalence_classes": len(active_groups),
        "min_class_size_k": min_k,
        "unique_individuals_count": unique_count,
        "uniqueness_percentage": float((unique_count / total_records) * 100),
        "median_class_size": median_k
    }


def anonymize_student_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies de-identification, generalization, and suppression:
    1. Direct identifiers: None in raw UCI dataset (confirmed).
    2. Suppression: Drops 'school' (institutional identifier).
    3. Generalization:
       - 'age' -> grouped into '15-16', '17-18', '19+'
       - 'Medu' & 'Fedu' -> grouped into 'Basic (0-2)' and 'Higher (3-4)'
       - 'absences' -> categorized into 'Low (0-4)', 'Moderate (5-10)', 'High (>10)'
    """
    anon = df.copy()
    
    # 1. Suppress institutional identifier
    if 'school' in anon.columns:
        anon = anon.drop(columns=['school'])
        
    # 2. Generalize Age
    if 'age' in anon.columns:
        anon['age_bin'] = pd.cut(
            anon['age'],
            bins=[14, 16, 18, 25],
            labels=['15-16', '17-18', '19+']
        ).astype(str)
        anon = anon.drop(columns=['age'])
        
    # 3. Generalize Parental Education
    if 'Medu' in anon.columns:
        anon['Medu_bin'] = pd.cut(
            anon['Medu'],
            bins=[-1, 2, 4],
            labels=['Basic (0-2)', 'Higher (3-4)']
        ).astype(str)
        anon = anon.drop(columns=['Medu'])
        
    if 'Fedu' in anon.columns:
        anon['Fedu_bin'] = pd.cut(
            anon['Fedu'],
            bins=[-1, 2, 4],
            labels=['Basic (0-2)', 'Higher (3-4)']
        ).astype(str)
        anon = anon.drop(columns=['Fedu'])
        
    # 4. Generalize Absences
    if 'absences' in anon.columns:
        anon['absences_bin'] = pd.cut(
            anon['absences'],
            bins=[-1, 4, 10, 100],
            labels=['Low (0-4)', 'Moderate (5-10)', 'High (>10)']
        ).astype(str)
        anon = anon.drop(columns=['absences'])
        
    return anon


def evaluate_anonymized_model(
    anon_df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Trains and evaluates a Decision Tree on the anonymized feature representation.
    """
    data = anon_df.copy()
    y = (data['G3'] >= 10).astype(int)
    
    cols_to_drop = ['G3']
    for g in ['G1', 'G2']:
        if g in data.columns:
            cols_to_drop.append(g)
            
    X_raw = data.drop(columns=cols_to_drop)
    
    categorical_cols = X_raw.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
    X_encoded = pd.get_dummies(X_raw, columns=categorical_cols, drop_first=True, dtype=int)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    dt = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=random_state)
    dt.fit(X_train, y_train)
    
    metrics = evaluate_classifier(dt, X_test, y_test, model_name="Anonymized (Generalized) Decision Tree")
    return metrics, X_encoded


if __name__ == "__main__":
    from data_preprocessing import load_dataset
    
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "student-mat.csv")
    df = load_dataset(data_path)
    
    print("--- Original Re-identification Risk ---")
    orig_risk = assess_reidentification_risk(df, DEFAULT_QUASI_IDENTIFIERS)
    for k, v in orig_risk.items():
        print(f"  {k}: {v}")
        
    anon_df = anonymize_student_dataset(df)
    anon_qis = ['sex', 'age_bin', 'address', 'Medu_bin', 'Fedu_bin']
    print("\n--- Anonymized Re-identification Risk ---")
    anon_risk = assess_reidentification_risk(anon_df, anon_qis)
    for k, v in anon_risk.items():
        print(f"  {k}: {v}")
        
    metrics, _ = evaluate_anonymized_model(anon_df)
    print("\n--- Downstream Anonymized Model Metrics ---")
    for k, v in metrics.items():
        if k != "confusion_matrix":
            print(f"  {k}: {v}")
