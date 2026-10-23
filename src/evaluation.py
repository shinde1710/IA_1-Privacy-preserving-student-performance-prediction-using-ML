"""
src/evaluation.py
-----------------
Comprehensive evaluation and visualization module for the Privacy-Preserving
Student Performance Prediction project.

Generates:
1. results/metrics.csv - Detailed performance metrics across models.
2. results/privacy_utility_comparison.csv - Privacy-utility trade-off table.
3. figures/class_distribution.png - Pass/Fail class distribution.
4. figures/confusion_matrix.png - Baseline Decision Tree confusion matrix.
5. figures/model_comparison.png - Metric comparison across baseline & privacy models.
6. figures/feature_distortion.png - Privacy budget epsilon vs. MAE distortion.
7. figures/privacy_utility_tradeoff.png - Epsilon vs. Downstream F1 & Accuracy.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List

from data_preprocessing import load_dataset, preprocess_student_data
from baseline_model import (
    train_decision_tree,
    train_logistic_regression,
    evaluate_classifier,
    get_feature_importances
)
from anonymization import (
    assess_reidentification_risk,
    anonymize_student_dataset,
    evaluate_anonymized_model,
    DEFAULT_QUASI_IDENTIFIERS
)
from local_differential_privacy import (
    apply_laplace_ldp,
    run_ldp_experiments,
    FEATURE_NAME
)


# Configure clean, publication-grade aesthetics
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8


def plot_class_distribution(y: pd.Series, output_path: str):
    """
    Plots the target class distribution (Pass vs Fail).
    """
    counts = y.value_counts().sort_index()
    labels = ["Fail (G3 < 10)", "Pass (G3 >= 10)"]
    percentages = (counts / len(y)) * 100
    
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    bars = ax.bar(labels, counts, color=['#e74c3c', '#2ecc71'], width=0.45, edgecolor='#2c3e50', linewidth=1.2)
    
    for bar, pct, cnt in zip(bars, percentages, counts):
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + 6,
            f"{cnt} ({pct:.1f}%)",
            ha='center', va='bottom', fontsize=11, fontweight='bold', color='#2c3e50'
        )
        
    ax.set_ylim(0, max(counts) * 1.18)
    ax.set_title("Student Mathematics Target Distribution (G3)", fontsize=13, fontweight='bold', pad=15)
    ax.set_ylabel("Student Count", fontsize=11)
    ax.set_xlabel("Academic Outcome Category", fontsize=11)
    
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"[OK] Saved: {output_path}")


def plot_confusion_matrix(cm: np.ndarray, output_path: str, title: str = "Baseline Decision Tree Confusion Matrix"):
    """
    Plots a clear, professional heatmap of the confusion matrix.
    """
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    labels = ["Fail (0)", "Pass (1)"]
    
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', cbar=False,
        xticklabels=labels, yticklabels=labels,
        annot_kws={"size": 14, "weight": "bold"}, ax=ax
    )
    
    ax.set_title(title, fontsize=12, fontweight='bold', pad=12)
    ax.set_ylabel("True Academic Label", fontsize=11, fontweight='bold')
    ax.set_xlabel("Predicted Label", fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"[OK] Saved: {output_path}")


def plot_model_comparison(comparison_df: pd.DataFrame, output_path: str):
    """
    Plots a multi-bar chart comparing Accuracy, Precision, Recall, and F1 across methods.
    """
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    
    methods = comparison_df['Method'].tolist()
    x = np.arange(len(methods))
    width = 0.2
    
    ax.bar(x - 1.5*width, comparison_df['Accuracy'], width, label='Accuracy', color='#3498db')
    ax.bar(x - 0.5*width, comparison_df['Precision'], width, label='Precision', color='#9b59b6')
    ax.bar(x + 0.5*width, comparison_df['Recall'], width, label='Recall', color='#2ecc71')
    ax.bar(x + 1.5*width, comparison_df['F1-score'], width, label='F1-score', color='#e67e22')
    
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Performance Comparison: Baseline vs. Privacy-Preserving Approaches", fontsize=12, fontweight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha='right', fontsize=9.5)
    ax.set_ylim(0, 1.05)
    ax.legend(loc='lower right', frameon=True)
    
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"[OK] Saved: {output_path}")


def plot_feature_distortion(ldp_df: pd.DataFrame, output_path: str):
    """
    Plots privacy budget epsilon vs. feature distortion (MAE and RMSE).
    """
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    
    ax.plot(ldp_df['epsilon'], ldp_df['mae_distortion'], marker='o', linewidth=2.2, color='#e74c3c', label='Mean Absolute Error (MAE)')
    ax.plot(ldp_df['epsilon'], ldp_df['rmse_distortion'], marker='s', linewidth=2.0, linestyle='--', color='#8e44ad', label='Root Mean Squared Error (RMSE)')
    
    ax.set_xscale('log')
    ax.set_xlabel("Privacy Budget (Epsilon, log scale)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Distortion Error in 'studytime'", fontsize=11, fontweight='bold')
    ax.set_title("Local DP: Privacy Budget vs. Feature Distortion", fontsize=12, fontweight='bold', pad=12)
    ax.legend(loc='upper right', frameon=True)
    
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"[OK] Saved: {output_path}")


def plot_privacy_utility_tradeoff(ldp_df: pd.DataFrame, output_path: str):
    """
    Plots privacy budget epsilon vs. downstream model performance (Accuracy and F1-score).
    """
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    
    ax.plot(ldp_df['epsilon'], ldp_df['accuracy'], marker='^', linewidth=2.2, color='#2980b9', label='Downstream Accuracy')
    ax.plot(ldp_df['epsilon'], ldp_df['f1'], marker='D', linewidth=2.2, color='#27ae60', label='Downstream F1-score')
    
    ax.set_xscale('log')
    ax.set_xlabel("Privacy Budget (Epsilon, log scale)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Classification Performance Metric", fontsize=11, fontweight='bold')
    ax.set_title("Privacy-Utility Trade-off under Local Differential Privacy", fontsize=12, fontweight='bold', pad=12)
    ax.set_ylim(0.5, 0.9)
    ax.legend(loc='lower right', frameon=True)
    
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"[OK] Saved: {output_path}")


def execute_all_experiments(base_dir: str):
    """
    Orchestrates all experiments, saves metrics, figures, and comparison tables.
    """
    data_path = os.path.join(base_dir, "data", "student-mat.csv")
    results_dir = os.path.join(base_dir, "results")
    figures_dir = os.path.join(base_dir, "figures")
    
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    
    # 1. Load Data
    df = load_dataset(data_path)
    y_raw = (df['G3'] >= 10).astype(int)
    
    # Plot Class Distribution
    plot_class_distribution(y_raw, os.path.join(figures_dir, "class_distribution.png"))
    
    # 2. Baseline Model
    X_train, X_test, y_train, y_test, f_names = preprocess_student_data(df)
    
    dt_baseline = train_decision_tree(X_train, y_train)
    m_dt = evaluate_classifier(dt_baseline, X_test, y_test, "Baseline Decision Tree")
    
    lr_baseline = train_logistic_regression(X_train, y_train)
    m_lr = evaluate_classifier(lr_baseline, X_test, y_test, "Baseline Logistic Regression")
    
    plot_confusion_matrix(
        np.array(m_dt['confusion_matrix']),
        os.path.join(figures_dir, "confusion_matrix.png")
    )
    
    # 3. Anonymization Model
    anon_df = anonymize_student_dataset(df)
    m_anon, _ = evaluate_anonymized_model(anon_df)
    
    # 4. Local Differential Privacy Experiments
    ldp_results = run_ldp_experiments(df)
    plot_feature_distortion(ldp_results, os.path.join(figures_dir, "feature_distortion.png"))
    plot_privacy_utility_tradeoff(ldp_results, os.path.join(figures_dir, "privacy_utility_tradeoff.png"))
    
    # Select key LDP checkpoints: Strong (eps=0.5), Moderate (eps=1.0), Weak (eps=5.0)
    ldp_05 = ldp_results[ldp_results['epsilon'] == 0.5].iloc[0]
    ldp_10 = ldp_results[ldp_results['epsilon'] == 1.0].iloc[0]
    ldp_50 = ldp_results[ldp_results['epsilon'] == 5.0].iloc[0]
    
    # 5. Compile Privacy-Utility Comparison Table
    comparison_records = [
        {
            "Method": "Baseline Decision Tree",
            "Privacy mechanism or transformation": "None (Raw identifiable features)",
            "Accuracy": round(m_dt['accuracy'], 4),
            "Precision": round(m_dt['precision'], 4),
            "Recall": round(m_dt['recall'], 4),
            "F1-score": round(m_dt['f1'], 4),
            "Feature distortion, where applicable": "None (0.000)",
            "Limitations": "High re-identification risk (27.6% unique QI signatures); susceptible to membership inference."
        },
        {
            "Method": "Baseline Logistic Regression",
            "Privacy mechanism or transformation": "None (Raw features, linear model)",
            "Accuracy": round(m_lr['accuracy'], 4),
            "Precision": round(m_lr['precision'], 4),
            "Recall": round(m_lr['recall'], 4),
            "F1-score": round(m_lr['f1'], 4),
            "Feature distortion, where applicable": "None (0.000)",
            "Limitations": "No privacy safeguards; weights reveal linear sensitivity to input demographics."
        },
        {
            "Method": "Anonymized Decision Tree",
            "Privacy mechanism or transformation": "Generalization (age, Medu, Fedu, absences) & suppression (school)",
            "Accuracy": round(m_anon['accuracy'], 4),
            "Precision": round(m_anon['precision'], 4),
            "Recall": round(m_anon['recall'], 4),
            "F1-score": round(m_anon['f1'], 4),
            "Feature distortion, where applicable": "Information loss via discretization",
            "Limitations": "Reduces unique signatures from 27.6% to 1.3%, but vulnerable to homogeneity/background knowledge attacks; no formal DP guarantee."
        },
        {
            "Method": "LDP Decision Tree (eps=0.5)",
            "Privacy mechanism or transformation": "Local Laplace mechanism on 'studytime' (scale=6.0, clipped [1,4])",
            "Accuracy": round(ldp_05['accuracy'], 4),
            "Precision": round(ldp_05['precision'], 4),
            "Recall": round(ldp_05['recall'], 4),
            "F1-score": round(ldp_05['f1'], 4),
            "Feature distortion, where applicable": f"MAE: {ldp_05['mae_distortion']:.4f} (RMSE: {ldp_05['rmse_distortion']:.4f})",
            "Limitations": "Significant noise on studytime; feature-level guarantee only; aggregator is untrusted for this feature."
        },
        {
            "Method": "LDP Decision Tree (eps=1.0)",
            "Privacy mechanism or transformation": "Local Laplace mechanism on 'studytime' (scale=3.0, clipped [1,4])",
            "Accuracy": round(ldp_10['accuracy'], 4),
            "Precision": round(ldp_10['precision'], 4),
            "Recall": round(ldp_10['recall'], 4),
            "F1-score": round(ldp_10['f1'], 4),
            "Feature distortion, where applicable": f"MAE: {ldp_10['mae_distortion']:.4f} (RMSE: {ldp_10['rmse_distortion']:.4f})",
            "Limitations": "Moderate privacy; balanced reconstruction error; protects single attribute against untrusted central server."
        },
        {
            "Method": "LDP Decision Tree (eps=5.0)",
            "Privacy mechanism or transformation": "Local Laplace mechanism on 'studytime' (scale=0.6, clipped [1,4])",
            "Accuracy": round(ldp_50['accuracy'], 4),
            "Precision": round(ldp_50['precision'], 4),
            "Recall": round(ldp_50['recall'], 4),
            "F1-score": round(ldp_50['f1'], 4),
            "Feature distortion, where applicable": f"MAE: {ldp_50['mae_distortion']:.4f} (RMSE: {ldp_50['rmse_distortion']:.4f})",
            "Limitations": "Weak privacy protection (high epsilon allows probabilistic attribute reconstruction)."
        }
    ]
    
    comp_df = pd.DataFrame(comparison_records)
    comp_csv = os.path.join(results_dir, "privacy_utility_comparison.csv")
    comp_df.to_csv(comp_csv, index=False)
    print(f"[OK] Saved: {comp_csv}")
    
    # 6. Save Detailed Metrics CSV
    all_metrics = [
        {"model": "Baseline Decision Tree", **m_dt},
        {"model": "Baseline Logistic Regression", **m_lr},
        {"model": "Anonymized Decision Tree", **m_anon}
    ]
    for _, row in ldp_results.iterrows():
        all_metrics.append({
            "model": f"LDP_studytime_eps_{row['epsilon']}",
            "accuracy": row['accuracy'],
            "precision": row['precision'],
            "recall": row['recall'],
            "f1": row['f1'],
            "macro_f1": row['macro_f1'],
            "mae_distortion": row['mae_distortion'],
            "rmse_distortion": row['rmse_distortion']
        })
    metrics_df = pd.DataFrame(all_metrics)
    metrics_csv = os.path.join(results_dir, "metrics.csv")
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"[OK] Saved: {metrics_csv}")
    
    # Plot Model Comparison Bar Chart
    plot_model_comparison(comp_df, os.path.join(figures_dir, "model_comparison.png"))
    
    return comp_df, metrics_df


if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    comp, mets = execute_all_experiments(base_dir)
    print("\n=== EXPERIMENTAL SUMMARY TABLE ===")
    print(comp[['Method', 'Accuracy', 'Precision', 'Recall', 'F1-score', 'Feature distortion, where applicable']])
