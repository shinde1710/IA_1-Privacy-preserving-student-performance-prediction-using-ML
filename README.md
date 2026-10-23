# Privacy-Preserving Student Performance Prediction Using Machine Learning
### Subject: Security and Privacy for Machine Learning | Internal Assessment 2 (IA 2)
**Student:** Aryan (Semester VII B.Tech) | **Academic Year:** 2026–2027  

---

## 1. Project Overview & IA 2 Objectives
This project investigates the empirical trade-offs between privacy guarantees and downstream prediction utility in educational data mining. We predict whether a student will pass or fail mathematics using the benchmark **UCI Student Performance Dataset** (`student-mat.csv`, $N=395$).

### Key Features & IA 2 Advances
1. **Target Construction:** Binarized final grade $G3$ ($Pass \ge 10$, $Fail < 10$).
2. **Leakage Prevention:** Strictly excludes intermediate grades $G1, G2$ and target $G3$ from features, ensuring legitimate early-semester prediction before exams take place.
3. **Cross-Validation Stability:** Evaluates baseline pruned Decision Tree and Logistic Regression across 5-Fold Stratified Cross-Validation (Decision Tree: Accuracy $69.87\% \pm 3.14\%$, F1 $79.28\% \pm 2.34\%$).
4. **Experiment A (Data Anonymization):** Audited quasi-identifiers (`school, sex, age, address, Medu, Fedu`). Suppressing `school` and generalizing `age`, `Medu`, `Fedu`, and `absences` reduced the re-identification uniqueness rate from **$27.59\%$ down to $1.27\%$** ($95.4\%$ reduction) while improving accuracy to $68.35\%$.
5. **Experiment B (Local Differential Privacy):** Client-side bounded Laplace mechanism on `studytime` (domain $[1, 4]$, $\Delta f = 3$) satisfying $\epsilon$-LDP via post-processing clipping across $\epsilon \in [0.1, 10.0]$, with feature distortion and probability distribution auditing.
6. **Experiment C (Central Differential Privacy):** Implemented Output Perturbation Logistic Regression and IBM `diffprivlib` Gaussian Naive Bayes satisfying $(\epsilon, 0)$-Central DP.
7. **Experiment D (Privacy–Utility Trade-Off):** Formulated multi-paradigm comparison matrices and multi-curve trade-off plots.
8. **Experiment E (Privacy Risk Analysis):** Executed an empirical Attribute Inference Attack demonstrating that overfitted models leak sensitive relationship status with $70.89\%$ accuracy, whereas pruned models neutralize leakage to baseline chance ($68.35\%$).
9. **Module 5 (Compressive Privacy):** Demonstrated representation learning where orthogonal nullspace projection maintains $69.62\%$ accuracy while proving why standard PCA fails to preserve privacy.

---

## 2. Project Directory Structure

```text
student_performance_privacy/
├── data/
│   ├── README.md                          # Dataset documentation, license & provenance
│   └── student-mat.csv                    # Official UCI dataset (395 records, 33 attributes)
├── notebooks/
│   ├── IA1_Student_Performance_Privacy.ipynb # IA 1 walkthrough notebook
│   └── IA2_Student_Performance_Privacy.ipynb # Complete IA 2 interactive notebook
├── src/
│   ├── download_data.py                   # Dataset download & extraction script
│   ├── data_preprocessing.py              # Leakage-free preprocessing & stratified split
│   ├── baseline_model.py                  # Pruned Decision Tree, Logistic Regression & 5-Fold CV
│   ├── anonymization.py                   # k-Anonymity audit, generalization & evaluation
│   ├── local_differential_privacy.py      # Client-side Laplace mechanism & distortion metrics
│   ├── central_differential_privacy.py    # Output perturbation & diffprivlib GaussianNB
│   ├── privacy_risk_analysis.py           # Attribute Inference & Membership Inference attacks
│   ├── compressive_privacy.py             # Orthogonal projection vs PCA representation learning
│   ├── evaluation.py                      # Master pipeline orchestrator & figure generator
│   ├── build_ia2_notebook.py              # IA 2 Jupyter Notebook builder
│   ├── generate_ia2_docx_report.py        # Word report generator with embedded figures
│   └── package_project.py                 # Distribution packaging script
├── results/
│   ├── metrics.csv                        # IA 1 metrics log
│   ├── privacy_utility_comparison.csv     # IA 1 comparison table
│   ├── metrics_ia2.csv                    # IA 2 comprehensive metrics log
│   ├── privacy_utility_comparison_ia2.csv # IA 2 consolidated multi-paradigm comparison
│   ├── cross_validation_results.csv       # 5-Fold Stratified Cross-Validation metrics
│   ├── central_dp_results.csv             # Central DP performance across epsilons
│   ├── inference_risk_results.csv         # Attribute inference attack results
│   └── compressive_privacy_results.csv    # Target F1 vs sensitive leakage comparison
├── figures/
│   ├── class_distribution.png             # Target distribution (Pass: 265, Fail: 130)
│   ├── confusion_matrix.png               # Baseline Decision Tree confusion matrix
│   ├── cross_validation_metrics.png       # 5-Fold Cross-Validation performance stability
│   ├── model_comparison.png               # Multi-metric comparison bar chart
│   ├── feature_distortion.png             # Epsilon vs MAE/RMSE distortion on studytime
│   ├── feature_distribution_ldp.png       # studytime distribution before vs after LDP
│   ├── central_dp_tradeoff.png            # Central DP accuracy across epsilons
│   ├── inference_risk_analysis.png        # Attribute inference attack accuracy
│   └── privacy_utility_tradeoff_ia2.png   # Unified multi-paradigm privacy-utility frontier
├── tests/
│   ├── test_preprocessing.py              # Leakage prevention & data validation unit tests
│   ├── test_evaluation.py                 # Model evaluation & LDP bounds unit tests
│   ├── test_central_dp.py                 # Central DP bounds & noise injection tests
│   └── test_risk_analysis.py              # Privacy risk & compressive privacy unit tests
├── reports/
│   ├── IA2_Final_Report.md                # Complete 20-section technical report (Markdown)
│   ├── IA2_Final_Report.docx              # Formatted Word report (910 KB with embedded figures)
│   ├── IA2_VIVA_PREPARATION_GUIDE.md      # 10 Essential Viva Questions with simple answers
│   ├── IA1_Preliminary_Report.md          # IA 1 preliminary report archive
│   └── IA1_Preliminary_Report.docx        # IA 1 Word report archive
├── README.md                              # Complete setup & execution guide
└── requirements.txt                       # Tested Python dependencies
```

---

## 3. Quickstart & Execution Guide

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run All Experiments & Generate Figures
To run baseline models, 5-fold cross-validation, anonymization, Local DP, Central DP, inference attacks, compressive privacy, and update all figures and CSV files:
```bash
python src/evaluation.py
```

### Step 3: Run Automated Unit Tests
To verify data validation, leakage prevention, privacy bounds, and inference risk functions:
```bash
pytest tests/
```
*(All 11 tests pass with 100% coverage).*

### Step 4: Launch Interactive Jupyter Notebook
```bash
jupyter notebook notebooks/IA2_Student_Performance_Privacy.ipynb
```
*(You can also upload this notebook directly to **Google Colab**).*

### Step 5: Generate Formatted Word Report (.docx)
To regenerate the Word document with embedded high-resolution figures:
```bash
python src/generate_ia2_docx_report.py
```

### Step 6: Package Project ZIP Archive
```bash
python src/package_project.py
```

---

## 4. Key Experimental Results

### Consolidated Privacy–Utility Comparison Matrix (IA 2)

| Method | Privacy Mechanism / Transformation | Accuracy | Precision | Recall | F1-Score | Feature Distortion |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Baseline Decision Tree** | None (Raw features) | **0.6582** | 0.6857 | 0.9057 | **0.7805** | None (0.000) |
| **Baseline Logistic Reg** | None (Linear model) | **0.6582** | 0.7167 | 0.8113 | **0.7611** | None (0.000) |
| **Anonymized Decision Tree** | Generalization (age, Medu, Fedu, absences) & suppression (school) | **0.6835** | 0.7333 | 0.8302 | **0.7788** | Discretization info loss |
| **Local DP DT ($\epsilon=0.5$)** | Client-side Laplace on `studytime` (scale=6.0, clipped $[1,4]$) | **0.6582** | 0.6857 | 0.9057 | **0.7805** | MAE: 1.2905 |
| **Local DP DT ($\epsilon=1.0$)** | Client-side Laplace on `studytime` (scale=3.0, clipped $[1,4]$) | **0.6582** | 0.6857 | 0.9057 | **0.7805** | MAE: 1.1030 |
| **Central DP LR ($\epsilon=0.5$)** | Output perturbation on weights via Laplace | **0.6405** | 0.6857 | 0.8679 | **0.7473** | Weight noise ($\Delta_w=0.0063$) |
| **Central DP LR ($\epsilon=1.0$)** | Output perturbation on weights via Laplace | **0.6405** | 0.6857 | 0.8679 | **0.7473** | Weight noise ($\Delta_w=0.0063$) |

### 5-Fold Stratified Cross-Validation Stability
- **Decision Tree ($d=3$):** Accuracy = $0.6987 \pm 0.0314$, F1 = $0.7928 \pm 0.0234$
- **Logistic Regression:** Accuracy = $0.6937 \pm 0.0490$, F1 = $0.7888 \pm 0.0342$

### Attribute Inference Attack Results
- **Majority Chance Baseline:** $68.35\%$
- **Public Features Only:** $68.35\%$ (F1 = 0.1379)
- **Baseline Pruned Tree Output:** $68.35\%$ (F1 = 0.1379) — zero leakage above prior.
- **Overfitted Tree Output:** $70.89\%$ (Precision = 66.67%, F1 = 0.2581) — leaks sensitive relationship status!

---

## 5. Mapping to All Five Syllabus Modules
- **Module 1 (Privacy Considerations in ML):** Target leakage prevention (dropping $G1, G2$), FERPA context, and empirical Attribute Inference Attack demonstration.
- **Module 2 (Differential Privacy for ML):** Output perturbation Central DP on Logistic Regression weights and IBM `diffprivlib` GaussianNB across $\epsilon \in [0.1, 10.0]$.
- **Module 3 (Local Differential Privacy for ML):** Client-side bounded Laplace perturbation on `studytime` with distribution and distortion plots.
- **Module 4 (Privacy-Preserving Data Mining):** Quasi-identifier audit reducing uniqueness from $27.59\%$ to $1.27\%$ via generalization and suppression.
- **Module 5 (Compressive Privacy for ML):** Empirical demonstration of orthogonal nullspace projection retaining $69.62\%$ accuracy while proving why standard PCA fails to preserve privacy.

---

## 6. Viva Preparation
Consult `reports/IA2_VIVA_PREPARATION_GUIDE.md` for 10 concise, beginner-friendly answers covering:
- Differences between Central DP and Local DP in practice.
- Mathematical derivation of output perturbation sensitivity.
- How overfitted models leak sensitive attributes in inference attacks.
- Why standard PCA fails as a compressive privacy tool.
- Cross-validation baseline stability.
