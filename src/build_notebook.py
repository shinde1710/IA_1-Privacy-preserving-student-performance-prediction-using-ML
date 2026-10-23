"""
build_notebook.py
Generates the complete, runnable Jupyter Notebook for IA 1.
"""

import json
import os

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Privacy-Preserving Student Performance Prediction Using Machine Learning\n",
                "### Course: Security and Privacy for Machine Learning | Internal Assessment 1 (IA 1)\n",
                "**Author:** Aryan (Semester VII B.Tech)\n",
                "**Primary Model:** Decision Tree Classifier (with Logistic Regression baseline comparison)\n",
                "**Dataset:** UCI Student Performance (Mathematics - `student-mat.csv`)\n",
                "\n",
                "---\n",
                "## Project Overview & Objectives\n",
                "1. **Problem:** Predict whether a secondary school student will pass ($G3 \\ge 10$) or fail ($G3 < 10$) using demographic, social, and academic attributes.\n",
                "2. **Target Leakage Prevention:** Early period grades ($G1, G2$) and the final score ($G3$) are strictly excluded from input features to simulate realistic early-stage educational intervention.\n",
                "3. **Privacy Experiments:**\n",
                "   - **Experiment A:** Baseline transparent Decision Tree and linear Logistic Regression.\n",
                "   - **Experiment B:** Data Anonymization (de-identification, generalization, and $k$-anonymity risk assessment).\n",
                "   - **Experiment C:** Local Differential Privacy (LDP) using the client-side bounded Laplace mechanism on `studytime`.\n",
                "   - **Experiment D:** Empirical privacy–utility trade-off analysis across privacy budgets $\\epsilon \\in [0.1, 10.0]$."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 1: Environment Setup & Library Imports"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "import sys\n",
                "import numpy as np\n",
                "import pandas as pd\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "\n",
                "# Modeling and Metrics\n",
                "from sklearn.model_selection import train_test_split\n",
                "from sklearn.tree import DecisionTreeClassifier, plot_tree\n",
                "from sklearn.linear_model import LogisticRegression\n",
                "from sklearn.metrics import (\n",
                "    accuracy_score, precision_score, recall_score, f1_score,\n",
                "    confusion_matrix, classification_report, mean_absolute_error, mean_squared_error\n",
                ")\n",
                "\n",
                "# Plot styling\n",
                "plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')\n",
                "plt.rcParams['font.sans-serif'] = 'Arial'\n",
                "print('All libraries successfully loaded!')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 2: Dataset Loading and Provenance Inspection\n",
                "We load `student-mat.csv` from the UCI Machine Learning Repository (Cortez & Silva, 2008)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "data_path = os.path.join('..', 'data', 'student-mat.csv')\n",
                "if not os.path.exists(data_path):\n",
                "    # Fallback to local path if running from project root\n",
                "    data_path = os.path.join('data', 'student-mat.csv')\n",
                "\n",
                "df = pd.read_csv(data_path, sep=';')\n",
                "print(f'Dataset Dimensions: {df.shape[0]} students, {df.shape[1]} attributes')\n",
                "print(f'Missing Values: {df.isnull().sum().sum()}')\n",
                "print(f'Duplicate Rows: {df.duplicated().sum()}')\n",
                "df.head(3)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 3: Target Construction & Target Leakage Prevention\n",
                "- Target: Binary outcome where $Pass = 1$ if $G3 \\ge 10$, and $Fail = 0$ if $G3 < 10$.\n",
                "- **Leakage Prevention:** Period 1 ($G1$) and Period 2 ($G2$) grades are strongly correlated with final grade $G3$. If included, models achieve deceptively high accuracy ($>90\\%$) but cannot be used for early-stage intervention before examinations occur. Thus, $G1, G2, G3$ are completely omitted from features."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "df['pass_target'] = (df['G3'] >= 10).astype(int)\n",
                "pass_cnt = df['pass_target'].sum()\n",
                "fail_cnt = len(df) - pass_cnt\n",
                "print(f'Target Distribution: Pass = {pass_cnt} ({pass_cnt/len(df)*100:.1f}%), Fail = {fail_cnt} ({fail_cnt/len(df)*100:.1f}%)')\n",
                "\n",
                "# Plot target distribution\n",
                "fig, ax = plt.subplots(figsize=(6, 4))\n",
                "bars = ax.bar(['Fail (G3 < 10)', 'Pass (G3 >= 10)'], [fail_cnt, pass_cnt], color=['#e74c3c', '#2ecc71'], width=0.4)\n",
                "ax.set_title('Target Outcome Distribution (G3)', fontsize=12, fontweight='bold')\n",
                "ax.set_ylabel('Number of Students')\n",
                "for bar in bars:\n",
                "    yval = bar.get_height()\n",
                "    ax.text(bar.get_x() + bar.get_width()/2, yval + 5, f'{yval} ({yval/len(df)*100:.1f}%)', ha='center', fontweight='bold')\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 4: Experiment A - Baseline Student Performance Prediction\n",
                "We train an explainable **Decision Tree Classifier** (pruned with `max_depth=3`) and a **Logistic Regression** baseline using stratified 80/20 train/test splitting."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Feature matrix without G1, G2, G3\n",
                "X_raw = df.drop(columns=['G1', 'G2', 'G3', 'pass_target'])\n",
                "y = df['pass_target']\n",
                "\n",
                "# One-Hot Encoding for categorical features\n",
                "categorical_cols = X_raw.select_dtypes(include=['object', 'category', 'string']).columns.tolist()\n",
                "X_encoded = pd.get_dummies(X_raw, columns=categorical_cols, drop_first=True, dtype=int)\n",
                "\n",
                "# Stratified split\n",
                "X_train, X_test, y_train, y_test = train_test_split(\n",
                "    X_encoded, y, test_size=0.2, random_state=42, stratify=y\n",
                ")\n",
                "\n",
                "# Train Decision Tree\n",
                "dt_baseline = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42)\n",
                "dt_baseline.fit(X_train, y_train)\n",
                "y_pred_dt = dt_baseline.predict(X_test)\n",
                "\n",
                "# Train Logistic Regression\n",
                "lr_baseline = LogisticRegression(max_iter=1000, random_state=42)\n",
                "lr_baseline.fit(X_train, y_train)\n",
                "y_pred_lr = lr_baseline.predict(X_test)\n",
                "\n",
                "print('--- Baseline Decision Tree ---')\n",
                "print(f'Accuracy : {accuracy_score(y_test, y_pred_dt):.4f}')\n",
                "print(f'Precision: {precision_score(y_test, y_pred_dt):.4f}')\n",
                "print(f'Recall   : {recall_score(y_test, y_pred_dt):.4f}')\n",
                "print(f'F1-Score : {f1_score(y_test, y_pred_dt):.4f}')\n",
                "\n",
                "print('\\n--- Baseline Logistic Regression ---')\n",
                "print(f'Accuracy : {accuracy_score(y_test, y_pred_lr):.4f}')\n",
                "print(f'Precision: {precision_score(y_test, y_pred_lr):.4f}')\n",
                "print(f'Recall   : {recall_score(y_test, y_pred_lr):.4f}')\n",
                "print(f'F1-Score : {f1_score(y_test, y_pred_lr):.4f}')"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Plot Decision Tree Confusion Matrix\n",
                "cm = confusion_matrix(y_test, y_pred_dt)\n",
                "fig, ax = plt.subplots(figsize=(5, 4))\n",
                "sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,\n",
                "            xticklabels=['Fail (0)', 'Pass (1)'], yticklabels=['Fail (0)', 'Pass (1)'],\n",
                "            annot_kws={'size': 14, 'weight': 'bold'}, ax=ax)\n",
                "ax.set_title('Baseline Decision Tree Confusion Matrix', fontweight='bold')\n",
                "ax.set_ylabel('True Label')\n",
                "ax.set_xlabel('Predicted Label')\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 5: Experiment B - Data Anonymization & Re-identification Risk Assessment\n",
                "In Module 4 (Privacy-Preserving Data Mining), we examine how quasi-identifiers (`school, sex, age, address, Medu, Fedu`) allow re-identifying individual students via linkage attacks.\n",
                "- We assess equivalence classes and uniqueness percentage before and after generalization."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "qis = ['school', 'sex', 'age', 'address', 'Medu', 'Fedu']\n",
                "grouped_raw = df.groupby(qis, observed=False).size()\n",
                "active_raw = grouped_raw[grouped_raw > 0]\n",
                "unique_raw = (active_raw == 1).sum()\n",
                "\n",
                "print('=== Raw Dataset Re-identification Risk ===')\n",
                "print(f'Total Equivalence Classes: {len(active_raw)}')\n",
                "print(f'Uniquely Identifiable Students (k=1): {unique_raw} ({unique_raw/len(df)*100:.1f}%)')\n",
                "print(f'Minimum Class Size (k): {active_raw.min()}')\n",
                "\n",
                "# Apply Generalization & Suppression\n",
                "df_anon = df.copy()\n",
                "df_anon = df_anon.drop(columns=['school'])  # Suppress school identity\n",
                "df_anon['age_bin'] = pd.cut(df_anon['age'], bins=[14, 16, 18, 25], labels=['15-16', '17-18', '19+']).astype(str)\n",
                "df_anon['Medu_bin'] = pd.cut(df_anon['Medu'], bins=[-1, 2, 4], labels=['Basic (0-2)', 'Higher (3-4)']).astype(str)\n",
                "df_anon['Fedu_bin'] = pd.cut(df_anon['Fedu'], bins=[-1, 2, 4], labels=['Basic (0-2)', 'Higher (3-4)']).astype(str)\n",
                "df_anon['absences_bin'] = pd.cut(df_anon['absences'], bins=[-1, 4, 10, 100], labels=['Low (0-4)', 'Moderate (5-10)', 'High (>10)']).astype(str)\n",
                "df_anon = df_anon.drop(columns=['age', 'Medu', 'Fedu', 'absences'])\n",
                "\n",
                "qis_anon = ['sex', 'age_bin', 'address', 'Medu_bin', 'Fedu_bin']\n",
                "grouped_anon = df_anon.groupby(qis_anon, observed=False).size()\n",
                "active_anon = grouped_anon[grouped_anon > 0]\n",
                "unique_anon = (active_anon == 1).sum()\n",
                "\n",
                "print('\\n=== Anonymized Dataset Re-identification Risk ===')\n",
                "print(f'Total Equivalence Classes: {len(active_anon)}')\n",
                "print(f'Uniquely Identifiable Students (k=1): {unique_anon} ({unique_anon/len(df)*100:.1f}%)')\n",
                "print(f'Median Class Size: {active_anon.median():.1f}')"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Train Downstream Decision Tree on Anonymized Data\n",
                "X_anon_raw = df_anon.drop(columns=['G1', 'G2', 'G3', 'pass_target'])\n",
                "cat_anon = X_anon_raw.select_dtypes(include=['object', 'category', 'string']).columns.tolist()\n",
                "X_anon_enc = pd.get_dummies(X_anon_raw, columns=cat_anon, drop_first=True, dtype=int)\n",
                "\n",
                "X_tr_a, X_te_a, y_tr_a, y_te_a = train_test_split(\n",
                "    X_anon_enc, y, test_size=0.2, random_state=42, stratify=y\n",
                ")\n",
                "dt_anon = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42)\n",
                "dt_anon.fit(X_tr_a, y_tr_a)\n",
                "y_pred_anon = dt_anon.predict(X_te_a)\n",
                "\n",
                "print('--- Anonymized Decision Tree Performance ---')\n",
                "print(f'Accuracy : {accuracy_score(y_te_a, y_pred_anon):.4f}')\n",
                "print(f'Precision: {precision_score(y_te_a, y_pred_anon):.4f}')\n",
                "print(f'Recall   : {recall_score(y_te_a, y_pred_anon):.4f}')\n",
                "print(f'F1-Score : {f1_score(y_te_a, y_pred_anon):.4f}')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 6: Experiment C - Local Differential Privacy (LDP)\n",
                "Under Local DP (Module 3), data is perturbed directly at the student client device before being transmitted to the school server. The server is untrusted.\n",
                "- Target feature: `studytime` with bounded domain $[1, 4]$. Sensitivity $\\Delta f = 4 - 1 = 3$.\n",
                "- Noise: $Z_i \\sim \\text{Laplace}(0, 3 / \\epsilon)$.\n",
                "- Perturbation: $x_{priv} = \\text{clip}(x_i + Z_i, 1.0, 4.0)$ preserves $\\epsilon$-LDP by the post-processing theorem."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "epsilons = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]\n",
                "ldp_records = []\n",
                "\n",
                "for eps in epsilons:\n",
                "    rng = np.random.default_rng(42)\n",
                "    noise = rng.laplace(0, 3.0 / eps, size=len(df))\n",
                "    perturbed_studytime = np.clip(df['studytime'] + noise, 1.0, 4.0)\n",
                "    mae = mean_absolute_error(df['studytime'], perturbed_studytime)\n",
                "    rmse = np.sqrt(mean_squared_error(df['studytime'], perturbed_studytime))\n",
                "    \n",
                "    # Build dataset with perturbed studytime\n",
                "    X_p = X_raw.copy()\n",
                "    X_p['studytime'] = perturbed_studytime\n",
                "    X_p_enc = pd.get_dummies(X_p, columns=categorical_cols, drop_first=True, dtype=int)\n",
                "    \n",
                "    X_tr_p, X_te_p, y_tr_p, y_te_p = train_test_split(\n",
                "        X_p_enc, y, test_size=0.2, random_state=42, stratify=y\n",
                "    )\n",
                "    \n",
                "    dt_ldp = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42)\n",
                "    dt_ldp.fit(X_tr_p, y_tr_p)\n",
                "    pred_ldp = dt_ldp.predict(X_te_p)\n",
                "    \n",
                "    ldp_records.append({\n",
                "        'epsilon': eps,\n",
                "        'MAE_distortion': mae,\n",
                "        'RMSE_distortion': rmse,\n",
                "        'Accuracy': accuracy_score(y_te_p, pred_ldp),\n",
                "        'Precision': precision_score(y_te_p, pred_ldp),\n",
                "        'Recall': recall_score(y_te_p, pred_ldp),\n",
                "        'F1-score': f1_score(y_te_p, pred_ldp)\n",
                "    })\n",
                "\n",
                "ldp_df = pd.DataFrame(ldp_records)\n",
                "print(ldp_df.to_string(index=False))"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 7: Experiment D - Privacy–Utility Visualizations\n",
                "We visualize:\n",
                "1. Privacy budget $\\epsilon$ vs. Feature distortion error (MAE and RMSE).\n",
                "2. Privacy budget $\\epsilon$ vs. Downstream classification performance."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))\n",
                "\n",
                "# Distortion Plot\n",
                "ax1.plot(ldp_df['epsilon'], ldp_df['MAE_distortion'], marker='o', color='#e74c3c', linewidth=2, label='MAE Distortion')\n",
                "ax1.plot(ldp_df['epsilon'], ldp_df['RMSE_distortion'], marker='s', linestyle='--', color='#8e44ad', linewidth=2, label='RMSE Distortion')\n",
                "ax1.set_xscale('log')\n",
                "ax1.set_title('Feature Distortion vs. Privacy Budget (Epsilon)', fontweight='bold')\n",
                "ax1.set_xlabel('Epsilon (log scale)')\n",
                "ax1.set_ylabel('Distortion in studytime')\n",
                "ax1.legend()\n",
                "\n",
                "# Utility Plot\n",
                "ax2.plot(ldp_df['epsilon'], ldp_df['Accuracy'], marker='^', color='#2980b9', linewidth=2, label='Accuracy')\n",
                "ax2.plot(ldp_df['epsilon'], ldp_df['F1-score'], marker='D', color='#27ae60', linewidth=2, label='F1-score')\n",
                "ax2.set_xscale('log')\n",
                "ax2.set_title('Downstream Model Utility vs. Epsilon', fontweight='bold')\n",
                "ax2.set_xlabel('Epsilon (log scale)')\n",
                "ax2.set_ylabel('Performance Metric')\n",
                "ax2.set_ylim(0.5, 0.9)\n",
                "ax2.legend()\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 8: Comprehensive Privacy–Utility Summary Table\n",
                "Final synthesis comparing Baseline, Anonymized, and LDP models across privacy parameters, distortion, and operational limitations."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "summary_csv = os.path.join('..', 'results', 'privacy_utility_comparison.csv')\n",
                "if not os.path.exists(summary_csv):\n",
                "    summary_csv = os.path.join('results', 'privacy_utility_comparison.csv')\n",
                "\n",
                "comp_df = pd.read_csv(summary_csv)\n",
                "comp_df"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbformat": 4,
            "nbformat_minor": 2,
            "version": "3.12.3"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

out_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "notebooks", "IA1_Student_Performance_Privacy.ipynb"))
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2)

print(f"[OK] Generated Notebook at: {out_path}")
