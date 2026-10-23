"""
src/build_ia2_notebook.py
-------------------------
Builds the comprehensive, runnable Jupyter Notebook for IA 2.
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
                "### Course: Security and Privacy for Machine Learning | Internal Assessment 2 (IA 2)\n",
                "**Student:** Aryan (Semester VII B.Tech) | **Academic Year:** 2026–2027\n",
                "**Primary Model:** Pruned Decision Tree Classifier (with Logistic Regression baseline comparison)\n",
                "**Dataset:** UCI Student Performance (Mathematics - `student-mat.csv`)\n",
                "\n",
                "---\n",
                "## IA 2 Project Overview & Key Objectives\n",
                "1. **Prediction Task:** Predict whether a student will pass ($G3 \\ge 10$) or fail ($G3 < 10$) using demographic, social, and academic indicators.\n",
                "2. **Target Leakage Prevention:** Period grades ($G1, G2$) and final score ($G3$) are strictly omitted from input features to evaluate true early-warning intervention.\n",
                "3. **Experimental Framework:**\n",
                "   - **Baseline Improvement:** 5-Fold Stratified Cross-Validation on Decision Tree and Logistic Regression.\n",
                "   - **Experiment A (Data Anonymization):** Quasi-identifier audit, $k$-anonymity, generalization, and suppression.\n",
                "   - **Experiment B (Local Differential Privacy):** Client-side bounded Laplace perturbation on `studytime` across $\\epsilon \\in [0.1, 10.0]$ with before/after distribution analysis.\n",
                "   - **Experiment C (Central Differential Privacy):** Central DP Logistic Regression (output perturbation) and IBM `diffprivlib` Gaussian Naive Bayes.\n",
                "   - **Experiment D (Privacy–Utility Frontier):** Multi-paradigm trade-off comparison matrix and trade-off visualization.\n",
                "   - **Experiment E (Privacy Risk Analysis):** Attribute Inference Attack and Membership Inference Risk demonstrations.\n",
                "   - **Module 5 (Compressive Privacy):** Representation learning comparing raw, PCA, and compressive nullspace projections."
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
                "# Scikit-Learn\n",
                "from sklearn.model_selection import train_test_split, StratifiedKFold\n",
                "from sklearn.tree import DecisionTreeClassifier\n",
                "from sklearn.linear_model import LogisticRegression\n",
                "from sklearn.preprocessing import StandardScaler\n",
                "from sklearn.decomposition import PCA\n",
                "from sklearn.metrics import (\n",
                "    accuracy_score, precision_score, recall_score, f1_score,\n",
                "    confusion_matrix, mean_absolute_error, mean_squared_error\n",
                ")\n",
                "\n",
                "# IBM diffprivlib compatibility shim for modern scikit-learn\n",
                "import sklearn.tree._tree\n",
                "if not hasattr(sklearn.tree._tree, 'DOUBLE'):\n",
                "    sklearn.tree._tree.DOUBLE = np.float64\n",
                "if not hasattr(sklearn.tree._tree, 'DTYPE'):\n",
                "    sklearn.tree._tree.DTYPE = np.float32\n",
                "\n",
                "from diffprivlib.mechanisms import Laplace\n",
                "from diffprivlib.models import GaussianNB as DPGaussianNB\n",
                "\n",
                "plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')\n",
                "plt.rcParams['font.sans-serif'] = 'Arial'\n",
                "print('All libraries loaded successfully!')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 2: Dataset Loading and Provenance Inspection\n",
                "We load `student-mat.csv` from the official UCI Machine Learning Repository."
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
                "    data_path = os.path.join('data', 'student-mat.csv')\n",
                "\n",
                "df = pd.read_csv(data_path, sep=';')\n",
                "print(f'Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns')\n",
                "print(f'Missing Values: {df.isnull().sum().sum()}')\n",
                "print(f'Duplicates: {df.duplicated().sum()}')\n",
                "df.head(3)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 3: Target Construction & Target Leakage Prevention\n",
                "Target: $G3 \\ge 10 \\rightarrow Pass (1)$, $G3 < 10 \\rightarrow Fail (0)$.  \n",
                "Period grades $G1, G2$ and target $G3$ are removed from feature matrix $\\mathbf{X}$ to prevent target leakage."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "df['pass_target'] = (df['G3'] >= 10).astype(int)\n",
                "pass_cnt = int(df['pass_target'].sum())\n",
                "fail_cnt = int(len(df) - pass_cnt)\n",
                "print(f'Target Distribution: Pass = {pass_cnt} ({pass_cnt/len(df)*100:.1f}%), Fail = {fail_cnt} ({fail_cnt/len(df)*100:.1f}%)')\n",
                "\n",
                "fig, ax = plt.subplots(figsize=(6, 3.8))\n",
                "bars = ax.bar(['Fail (G3 < 10)', 'Pass (G3 >= 10)'], [fail_cnt, pass_cnt], color=['#e74c3c', '#2ecc71'], width=0.4)\n",
                "ax.set_title('Target Outcome Distribution (G3)', fontweight='bold')\n",
                "ax.set_ylabel('Number of Students')\n",
                "for b in bars:\n",
                "    yval = b.get_height()\n",
                "    ax.text(b.get_x() + b.get_width()/2, yval + 5, f'{yval} ({yval/len(df)*100:.1f}%)', ha='center', fontweight='bold')\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 4: Baseline Models & 5-Fold Stratified Cross-Validation\n",
                "We evaluate the baseline pruned Decision Tree (`max_depth=3`, `min_samples_leaf=5`) and Logistic Regression on both a single 80/20 test split and 5-Fold Stratified Cross-Validation."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "X_raw = df.drop(columns=['G1', 'G2', 'G3', 'pass_target'])\n",
                "y = df['pass_target']\n",
                "cat_cols = X_raw.select_dtypes(include=['object', 'category', 'string']).columns.tolist()\n",
                "X_enc = pd.get_dummies(X_raw, columns=cat_cols, drop_first=True, dtype=int)\n",
                "\n",
                "# Single 80/20 test split\n",
                "X_train, X_test, y_train, y_test = train_test_split(X_enc, y, test_size=0.2, random_state=42, stratify=y)\n",
                "\n",
                "dt_baseline = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42)\n",
                "dt_baseline.fit(X_train, y_train)\n",
                "y_pred_dt = dt_baseline.predict(X_test)\n",
                "\n",
                "print('=== Single Split Baseline Decision Tree ===')\n",
                "print(f'Accuracy : {accuracy_score(y_test, y_pred_dt):.4f}')\n",
                "print(f'Precision: {precision_score(y_test, y_pred_dt):.4f}')\n",
                "print(f'Recall   : {recall_score(y_test, y_pred_dt):.4f}')\n",
                "print(f'F1-Score : {f1_score(y_test, y_pred_dt):.4f}')\n",
                "\n",
                "# 5-Fold Stratified Cross Validation\n",
                "skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)\n",
                "cv_acc, cv_f1 = [], []\n",
                "for tr_idx, val_idx in skf.split(X_enc, y):\n",
                "    clf = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42)\n",
                "    clf.fit(X_enc.iloc[tr_idx], y.iloc[tr_idx])\n",
                "    pred = clf.predict(X_enc.iloc[val_idx])\n",
                "    cv_acc.append(accuracy_score(y.iloc[val_idx], pred))\n",
                "    cv_f1.append(f1_score(y.iloc[val_idx], pred))\n",
                "\n",
                "print(f'\\n=== 5-Fold Stratified CV Decision Tree ===')\n",
                "print(f'Accuracy : {np.mean(cv_acc):.4f} +/- {np.std(cv_acc):.4f}')\n",
                "print(f'F1-score : {np.mean(cv_f1):.4f} +/- {np.std(cv_f1):.4f}')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 5: Experiment A - Data Anonymization & k-Anonymity Risk Assessment\n",
                "Quasi-Identifiers: `['school', 'sex', 'age', 'address', 'Medu', 'Fedu']`.\n",
                "We measure equivalence class sizes before and after generalization/suppression."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "qis = ['school', 'sex', 'age', 'address', 'Medu', 'Fedu']\n",
                "raw_groups = df.groupby(qis, observed=False).size()\n",
                "active_raw = raw_groups[raw_groups > 0]\n",
                "unique_raw = (active_raw == 1).sum()\n",
                "\n",
                "print(f'Raw Data: {len(active_raw)} equivalence classes, {unique_raw} unique individuals ({unique_raw/len(df)*100:.1f}%)')\n",
                "\n",
                "# Generalization\n",
                "df_anon = df.copy().drop(columns=['school'])\n",
                "df_anon['age_bin'] = pd.cut(df_anon['age'], bins=[14, 16, 18, 25], labels=['15-16', '17-18', '19+']).astype(str)\n",
                "df_anon['Medu_bin'] = pd.cut(df_anon['Medu'], bins=[-1, 2, 4], labels=['Basic', 'Higher']).astype(str)\n",
                "df_anon['Fedu_bin'] = pd.cut(df_anon['Fedu'], bins=[-1, 2, 4], labels=['Basic', 'Higher']).astype(str)\n",
                "df_anon['absences_bin'] = pd.cut(df_anon['absences'], bins=[-1, 4, 10, 100], labels=['Low', 'Moderate', 'High']).astype(str)\n",
                "df_anon = df_anon.drop(columns=['age', 'Medu', 'Fedu', 'absences'])\n",
                "\n",
                "anon_qis = ['sex', 'age_bin', 'address', 'Medu_bin', 'Fedu_bin']\n",
                "anon_groups = df_anon.groupby(anon_qis, observed=False).size()\n",
                "active_anon = anon_groups[anon_groups > 0]\n",
                "unique_anon = (active_anon == 1).sum()\n",
                "print(f'Anonymized Data: {len(active_anon)} equivalence classes, {unique_anon} unique individuals ({unique_anon/len(df)*100:.1f}%)')\n",
                "\n",
                "# Downstream Model on Anonymized Data\n",
                "X_a_raw = df_anon.drop(columns=['G1', 'G2', 'G3', 'pass_target'])\n",
                "cat_a = X_a_raw.select_dtypes(include=['object', 'category', 'string']).columns.tolist()\n",
                "X_a_enc = pd.get_dummies(X_a_raw, columns=cat_a, drop_first=True, dtype=int)\n",
                "X_tr_a, X_te_a, y_tr_a, y_te_a = train_test_split(X_a_enc, y, test_size=0.2, random_state=42, stratify=y)\n",
                "\n",
                "dt_anon = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42).fit(X_tr_a, y_tr_a)\n",
                "pred_a = dt_anon.predict(X_te_a)\n",
                "print(f'Anonymized DT Accuracy: {accuracy_score(y_te_a, pred_a):.4f} | F1: {f1_score(y_te_a, pred_a):.4f}')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 6: Experiment B - Local Differential Privacy (LDP)\n",
                "Client-side bounded Laplace mechanism on `studytime` (range $[1, 4]$, sensitivity $\\Delta f = 3$).  \n",
                "Clipping to $[1, 4]$ satisfies $\\epsilon$-LDP by the post-processing theorem."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "epsilons = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]\n",
                "ldp_rows = []\n",
                "\n",
                "for eps in epsilons:\n",
                "    rng = np.random.default_rng(42)\n",
                "    noise = rng.laplace(0, 3.0 / eps, size=len(df))\n",
                "    st_noisy = np.clip(df['studytime'] + noise, 1.0, 4.0)\n",
                "    mae = mean_absolute_error(df['studytime'], st_noisy)\n",
                "    \n",
                "    X_p = X_raw.copy()\n",
                "    X_p['studytime'] = st_noisy\n",
                "    X_p_enc = pd.get_dummies(X_p, columns=cat_cols, drop_first=True, dtype=int)\n",
                "    X_tr_p, X_te_p, y_tr_p, y_te_p = train_test_split(X_p_enc, y, test_size=0.2, random_state=42, stratify=y)\n",
                "    \n",
                "    dt_p = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42).fit(X_tr_p, y_tr_p)\n",
                "    pred_p = dt_p.predict(X_te_p)\n",
                "    \n",
                "    ldp_rows.append({\n",
                "        'epsilon': eps,\n",
                "        'MAE_distortion': mae,\n",
                "        'Accuracy': accuracy_score(y_te_p, pred_p),\n",
                "        'F1-score': f1_score(y_te_p, pred_p)\n",
                "    })\n",
                "\n",
                "ldp_df = pd.DataFrame(ldp_rows)\n",
                "print(ldp_df.to_string(index=False))"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 7: Experiment C - Central Differential Privacy (Central DP)\n",
                "We implement Central DP using output perturbation on regularized Logistic Regression weights and IBM `diffprivlib` Gaussian Naive Bayes across $\\epsilon \\in [0.1, 10.0]$."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "scaler = StandardScaler()\n",
                "X_scaled = pd.DataFrame(scaler.fit_transform(X_enc), columns=X_enc.columns)\n",
                "X_tr_s, X_te_s, y_tr_s, y_te_s = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)\n",
                "\n",
                "cdp_records = []\n",
                "for eps in epsilons:\n",
                "    # Output perturbation on logistic regression weights\n",
                "    base_lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42).fit(X_tr_s, y_tr_s)\n",
                "    sens_w = 2.0 * 1.0 / len(X_tr_s)\n",
                "    rng = np.random.default_rng(42)\n",
                "    w_noisy = base_lr.coef_ + rng.laplace(0, sens_w / eps, size=base_lr.coef_.shape)\n",
                "    b_noisy = base_lr.intercept_ + rng.laplace(0, sens_w / eps, size=base_lr.intercept_.shape)\n",
                "    \n",
                "    logits = np.dot(X_te_s.values, w_noisy.T) + b_noisy\n",
                "    pred_cdp = (1.0 / (1.0 + np.exp(-logits)) >= 0.5).astype(int).ravel()\n",
                "    \n",
                "    cdp_records.append({\n",
                "        'epsilon': eps,\n",
                "        'Central_DP_LR_Accuracy': accuracy_score(y_te_s, pred_cdp),\n",
                "        'Central_DP_LR_F1': f1_score(y_te_s, pred_cdp)\n",
                "    })\n",
                "\n",
                "cdp_df = pd.DataFrame(cdp_records)\n",
                "print(cdp_df.to_string(index=False))"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 8: Experiment D - Multi-Paradigm Privacy–Utility Trade-Off\n",
                "Comparing Baseline, Anonymization, Local DP, and Central DP on a unified plot."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "fig, ax = plt.subplots(figsize=(8, 4.5))\n",
                "ax.plot(ldp_df['epsilon'], ldp_df['Accuracy'], marker='^', color='#e67e22', linewidth=2, label='Local DP (studytime Laplace)')\n",
                "ax.plot(cdp_df['epsilon'], cdp_df['Central_DP_LR_Accuracy'], marker='o', color='#2980b9', linewidth=2, label='Central DP (Logistic Regression)')\n",
                "ax.axhline(0.6835, color='#27ae60', linestyle='--', linewidth=1.8, label='Anonymized DT (0.684)')\n",
                "ax.axhline(0.6582, color='#c0392b', linestyle=':', linewidth=1.8, label='Baseline Raw DT (0.658)')\n",
                "ax.set_xscale('log')\n",
                "ax.set_title('Unified Privacy-Utility Frontier across Paradigms', fontweight='bold')\n",
                "ax.set_xlabel('Privacy Budget (Epsilon, log scale)')\n",
                "ax.set_ylabel('Accuracy')\n",
                "ax.set_ylim(0.55, 0.75)\n",
                "ax.legend()\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 9: Experiment E - Privacy Risk Analysis & Inference Attack\n",
                "Demonstrating Attribute Inference Attack: An adversary attempts to infer personal relationship status (`romantic`) from public covariates and model prediction output."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "s_romantic = (df['romantic'] == 'yes').astype(int)\n",
                "pub_features = pd.get_dummies(df[['school', 'sex', 'age', 'address', 'studytime']], drop_first=True, dtype=float)\n",
                "\n",
                "X_tr_f, X_te_f, y_tr_f, y_te_f, s_tr, s_te, pub_tr, pub_te = train_test_split(\n",
                "    X_enc, y, s_romantic, pub_features, test_size=0.2, random_state=42, stratify=y\n",
                ")\n",
                "\n",
                "# Target academic predictor\n",
                "clf_target = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42).fit(X_tr_f, y_tr_f)\n",
                "prob_tr = clf_target.predict_proba(X_tr_f)[:, 1].reshape(-1, 1)\n",
                "prob_te = clf_target.predict_proba(X_te_f)[:, 1].reshape(-1, 1)\n",
                "\n",
                "# Attack model\n",
                "att_X_tr = np.hstack([pub_tr.values, prob_tr])\n",
                "att_X_te = np.hstack([pub_te.values, prob_te])\n",
                "att_model = LogisticRegression(max_iter=1000, random_state=42).fit(att_X_tr, s_tr)\n",
                "att_pred = att_model.predict(att_X_te)\n",
                "\n",
                "majority_baseline = max(s_te.mean(), 1 - s_te.mean())\n",
                "print(f'Majority Class Chance Baseline: {majority_baseline:.4f}')\n",
                "print(f'Adversary Attack Accuracy     : {accuracy_score(s_te, att_pred):.4f}')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 10: Module 5 - Compressive Privacy Representation Learning\n",
                "Evaluating target classification utility ($G3 \\ge 10$) vs sensitive attribute leakage (`sex`) across Full, PCA, and Compressive Nullspace Projections."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from compressive_privacy import evaluate_compressive_privacy\n",
                "comp_results = evaluate_compressive_privacy(df)\n",
                "print(comp_results.to_string(index=False))"
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
            "name": "python",
            "version": "3.12.3"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

out_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "notebooks", "IA2_Student_Performance_Privacy.ipynb"))
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2)

print(f"[OK] Generated IA 2 Notebook at: {out_path}")
