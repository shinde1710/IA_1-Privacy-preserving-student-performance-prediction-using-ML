# Privacy-Preserving Student Performance Prediction Using Machine Learning
### Subject: Security and Privacy for Machine Learning | Internal Assessment 1 (IA 1)
**Student:** Aryan (Semester VII B.Tech)  
**Academic Year:** 2026–2027  

---

## 1. Project Overview
Educational institutions collect rich student data (study time, parental education, attendance, past failures) to identify students at risk of academic failure. However, training conventional machine learning models directly on raw records poses significant privacy threats:
- **Linkage Attacks:** Combining demographic quasi-identifiers with external voter or school rosters can re-identify individual students.
- **Untrusted Aggregators:** Cloud platforms or school servers can inspect intimate lifestyle habits (e.g., alcohol use, parental cohabitation).
- **Target Leakage:** Naive models rely on early exam grades ($G1, G2$), which inflates accuracy ($>90\%$) but prevents true early intervention before exams occur.

This project implements a **leakage-free machine learning pipeline** that compares:
1. **Baseline Decision Tree Classifier** (interpretable, pruned to depth 3 for viva explainability)
2. **Baseline Logistic Regression** (linear comparative model)
3. **Data Anonymization** ($k$-anonymity audit, generalization of age and education, suppression of school)
4. **Local Differential Privacy (LDP)** (client-side bounded Laplace mechanism on `studytime` across $\epsilon \in [0.1, 10.0]$)

---

## 2. Project Directory Structure

```text
student_performance_privacy/
├── data/
│   ├── README.md                      # Dataset documentation, license & provenance
│   └── student-mat.csv                # UCI Mathematics student dataset (395 records)
├── notebooks/
│   └── IA1_Student_Performance_Privacy.ipynb # Interactive, runnable walkthrough notebook
├── src/
│   ├── download_data.py               # Dataset download & extraction script
│   ├── data_preprocessing.py          # Leakage-free preprocessing & stratified split
│   ├── baseline_model.py              # Decision Tree & Logistic Regression training
│   ├── anonymization.py               # k-Anonymity audit, generalization & evaluation
│   ├── local_differential_privacy.py  # Client-side Laplace mechanism & distortion metrics
│   ├── evaluation.py                  # Experiment runner, CSV logger & figure generator
│   ├── build_notebook.py              # Jupyter Notebook generator
│   └── generate_docx_report.py        # Word report generator with embedded figures
├── results/
│   ├── metrics.csv                    # Detailed metric logs across all models
│   └── privacy_utility_comparison.csv # Consolidated privacy-utility comparison table
├── figures/
│   ├── class_distribution.png         # Target distribution (Pass: 265, Fail: 130)
│   ├── confusion_matrix.png           # Baseline Decision Tree confusion matrix
│   ├── model_comparison.png           # Multi-metric comparison bar chart
│   ├── feature_distortion.png         # Epsilon vs. Feature MAE/RMSE distortion
│   └── privacy_utility_tradeoff.png   # Epsilon vs. Downstream F1 & Accuracy
├── tests/
│   ├── test_preprocessing.py          # Leakage prevention & data validation tests
│   └── test_evaluation.py             # Model evaluation & LDP bounds tests
├── reports/
│   ├── IA1_Preliminary_Report.md      # Comprehensive 17-section technical report (Markdown)
│   ├── IA1_Preliminary_Report.docx    # Formatted Word report with embedded figures
│   └── VIVA_PREPARATION_GUIDE.md      # 12 Essential Viva Questions with simple answers
├── README.md                          # Project guide and instructions
└── requirements.txt                   # Python dependencies
```

---

## 3. Quickstart & Execution Guide

### Step 1: Install Dependencies
Open a terminal in the project directory and install the required packages:
```bash
pip install -r requirements.txt
```

### Step 2: Download Dataset
If `student-mat.csv` is not already present in `data/`, run:
```bash
python src/download_data.py
```
*(This automatically downloads and extracts the official UCI archive).*

### Step 3: Run All Experiments & Generate Figures
To run baseline models, anonymization, and LDP experiments, and update all figures and CSV files:
```bash
python src/evaluation.py
```

### Step 4: Run Automated Unit Tests
To verify data validation, leakage prevention, and privacy bounds:
```bash
pytest tests/
```
*(Expected output: 6 passed).*

### Step 5: Launch Jupyter Notebook
To run the interactive notebook in VS Code or Jupyter:
```bash
jupyter notebook notebooks/IA1_Student_Performance_Privacy.ipynb
```
*(You can also upload this notebook directly to **Google Colab**).*

### Step 6: Generate Word Report (.docx)
To regenerate the formatted Word document with updated figures:
```bash
python src/generate_docx_report.py
```

---

## 4. Key Experimental Results

### Privacy–Utility Comparison Summary Table

| Method | Privacy Mechanism / Transformation | Accuracy | Precision | Recall | F1-Score | Feature Distortion |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Baseline Decision Tree** | None (Raw features) | **0.6582** | 0.6857 | 0.9057 | **0.7805** | None (0.000) |
| **Baseline Logistic Regression** | None (Linear model) | **0.6582** | 0.7167 | 0.8113 | **0.7611** | None (0.000) |
| **Anonymized Decision Tree** | Generalization (age, Medu, Fedu, absences) & suppression (school) | **0.6835** | 0.7333 | 0.8302 | **0.7788** | Discretization info loss |
| **LDP Decision Tree ($\epsilon=0.5$)** | Local Laplace on `studytime` (scale=6.0, clipped $[1,4]$) | **0.6582** | 0.6857 | 0.9057 | **0.7805** | MAE: 1.2905 (RMSE: 1.5750) |
| **LDP Decision Tree ($\epsilon=1.0$)** | Local Laplace on `studytime` (scale=3.0, clipped $[1,4]$) | **0.6582** | 0.6857 | 0.9057 | **0.7805** | MAE: 1.1030 (RMSE: 1.3880) |
| **LDP Decision Tree ($\epsilon=5.0$)** | Local Laplace on `studytime` (scale=0.6, clipped $[1,4]$) | **0.6582** | 0.6857 | 0.9057 | **0.7805** | MAE: 0.4520 (RMSE: 0.6295) |

### Key Findings
1. **Re-Identification Vulnerability:** In the raw dataset, 109 students (**27.59%**) were completely unique ($k=1$) under quasi-identifiers (`school, sex, age, address, Medu, Fedu`).
2. **Anonymization Success:** Coarse generalization reduced uniqueness to **5 students (1.27%)**—a **95.4% drop in re-identification vulnerability**—while slightly improving accuracy to **68.35%** by regularizing tree splits.
3. **Local DP Distortion:** Adding Laplace noise locally to `studytime` showed that error drops from $\text{MAE}=1.4770$ at $\epsilon=0.1$ down to $\text{MAE}=0.2403$ at $\epsilon=10.0$.
4. **Target Leakage Prevention:** Excluding $G1$ and $G2$ establishes an honest, realistic early-prediction baseline of $\sim 66\%$ accuracy and $\sim 78\%$ F1-score.

---

## 5. Mapping to All Five Syllabus Modules
- **Module 1 (Privacy Considerations in ML):** Audited sensitive educational attributes and prevented target leakage by removing $G1$ and $G2$.
- **Module 2 (Differential Privacy for ML):** Derived Laplace noise scale $\lambda = \Delta f / \epsilon$ and studied privacy budget allocation.
- **Module 3 (Local Differential Privacy for ML):** Implemented client-side bounded perturbation on `studytime` under an untrusted aggregator threat model.
- **Module 4 (Privacy-Preserving Data Mining):** Conducted $k$-anonymity risk analysis and applied generalization and suppression.
- **Module 5 (Compressive Privacy for ML):** Demonstrated feature representation compression where discretized features preserved $77.88\%$ F1 utility while suppressing granular specifics.

---

## 6. Viva Preparation
Review `reports/VIVA_PREPARATION_GUIDE.md` for 12 concise, beginner-friendly answers explaining:
- Why Decision Trees were selected and how pruning prevents overfitting.
- Why $G1$ and $G2$ cause target leakage and must be removed.
- Difference between Direct Identifiers and Quasi-Identifiers.
- Weaknesses of $k$-anonymity (homogeneity and background knowledge attacks).
- What $\epsilon$ means in Differential Privacy and why clipping is valid under post-processing.
- Central DP vs. Local DP.

---

## 7. License and Attribution
- **Dataset:** UCI Machine Learning Repository (Cortez & Silva, 2008), licensed under Creative Commons Attribution 4.0 International (CC BY 4.0).
- **Code:** Created for academic use in Semester VII B.Tech (Security and Privacy for Machine Learning).
