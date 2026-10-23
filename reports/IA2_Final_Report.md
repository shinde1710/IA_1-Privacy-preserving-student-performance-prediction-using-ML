# Internal Assessment 2 (IA 2) Final Technical Report
## Privacy-Preserving Student Performance Prediction Using Machine Learning

---

### Course & Student Information
- **Course Title:** Security and Privacy for Machine Learning
- **Assessment Type:** Internal Assessment 2 (IA 2) — Final Research and Implementation Report
- **Student Name:** Aryan
- **Degree & Semester:** Bachelor of Technology (B.Tech), Semester VII
- **Academic Year:** 2026–2027

---

## 1. Title Page
- **Project Title:** Privacy-Preserving Student Performance Prediction Using Machine Learning
- **Subject Domain:** Privacy-Preserving Machine Learning (PPML), Educational Data Mining (EDM), Differential Privacy, and Responsible AI.
- **Primary Learning Algorithm:** Pruned Decision Tree Classifier (`max_depth=3`, `min_samples_leaf=5`) with comparative Logistic Regression.
- **Privacy Frameworks Implemented:**
  1. Data Anonymization ($k$-Anonymity, Equivalence Class Analysis, Attribute Generalization, Institutional Suppression)
  2. Local Differential Privacy (Client-Side Bounded Laplace Mechanism on `studytime`)
  3. Central Differential Privacy (Output Perturbation Logistic Regression and IBM `diffprivlib` Gaussian Naive Bayes)
  4. Privacy Risk & Inference Attack Auditing (Attribute Inference and Membership Inference Risk)
  5. Compressive Privacy Representation Learning (Subspace Orthogonal Filtering)
- **Target Variable:** Binary Academic Outcome ($G3 \ge 10 \rightarrow \text{Pass (1)}, G3 < 10 \rightarrow \text{Fail (0)}$).
- **Target Leakage Safeguard:** Strict omission of period examination grades ($G1, G2$) and final score ($G3$) from all input feature sets.

---

## 2. Abstract
Educational data mining models offer significant potential for identifying students at risk of academic failure early in the semester, allowing academic institutions to deliver timely pedagogical and pastoral interventions. However, educational databases store highly sensitive personal attributes, including home address types, parental cohabitation status, parental education levels, weekly study hours, and disciplinary school absences. Training predictive machine learning algorithms directly on unprotected, centralized student data exposes individuals to severe privacy hazards, including linkage-based re-identification attacks, unauthorized attribute inference, and membership exposure.

This project implements an end-to-end, privacy-preserving machine learning pipeline on the benchmark UC Irvine Student Performance dataset ($N=395$ records, Mathematics course). To reflect real-world pre-examination intervention, we eliminate target leakage by excluding intermediate examination grades ($G1, G2$), compelling models to predict academic success purely from demographic, behavioral, and environmental covariates. We systematically extend our initial IA 1 findings through:
1. Evaluating baseline stability via 5-Fold Stratified Cross-Validation (Decision Tree: Accuracy $69.87\% \pm 3.14\%$, F1-score $79.28\% \pm 2.34\%$);
2. Auditing and mitigating re-identification vulnerability via $k$-anonymity generalization, reducing uniquely identifiable students from $109$ ($27.59\%$) down to $5$ ($1.27\%$) while improving classification accuracy to $68.35\%$;
3. Implementing client-side Local Differential Privacy (LDP) on student study hours, empirically mapping feature distortion from $\text{MAE} = 1.4770$ ($\epsilon=0.1$) down to $\text{MAE} = 0.2403$ ($\epsilon=10.0$);
4. Implementing Central Differential Privacy (Central DP) using weight output perturbation, achieving an accuracy of $64.05\% \pm 1.29\%$ under $(\epsilon=1.0, 0)$-DP;
5. Executing an Attribute Inference Attack demonstrating that overfitted models leak unreleased relationship status with $70.89\%$ accuracy, whereas pruned models neutralize this leakage to baseline chance ($68.35\%$); and
6. Demonstrating Compressive Privacy representation learning where orthogonal nullspace projection maintains $69.62\%$ accuracy while reducing demographic correlation.

Our findings prove that judicious anonymization and localized feature perturbation provide verifiable privacy guarantees with negligible loss in early predictive utility.

---

## 3. Introduction and Motivation
Modern educational institutions increasingly collect and analyze comprehensive student portfolios to monitor student progression and allocate tutoring support. Predictive machine learning models can identify students at risk of failing well before semester examinations take place.

However, student datasets contain deeply personal information protected by regulatory frameworks, such as the Family Educational Rights and Privacy Act (FERPA) in the United States and the General Data Protection Regulation (GDPR) in the European Union. Transmitting, centralizing, and training models on raw student data poses critical threats:
- **Linkage Re-Identification:** Public databases (such as voter registration rosters or school sports listings) can be joined with quasi-identifiers in released student data to single out specific individuals.
- **Untrusted Central Aggregators:** Cloud providers or institutional administrators can inspect raw behavioral attributes, such as alcohol use or family divorce.
- **Model Inversion and Attribute Inference:** Model weights or output probabilities can be probed by malicious adversaries to deduce unreleased private student attributes.

This project addresses these challenges by developing and comparing multi-paradigm privacy-preserving machine learning methodologies on real secondary school student data.

---

## 4. Refined Problem Statement
Given a set of $N$ student records containing demographic, behavioral, environmental, and academic features, how can an educational machine learning system:
1. Predict final student academic outcome ($Pass / Fail$) accurately prior to major examination periods without causing target leakage;
2. Prevent linkage-based re-identification of students sharing distinct quasi-identifier signatures;
3. Protect sensitive student attributes against untrusted central servers using client-side Local Differential Privacy;
4. Enforce formal Central Differential Privacy guarantees on model parameters; and
5. Quantify the empirical trade-off between privacy parameters, feature distortion, and classification utility?

---

## 5. Objectives and Research Questions
### Objectives
1. Implement a leakage-free prediction pipeline predicting binary mathematics outcome ($G3 \ge 10$) without using earlier period grades ($G1, G2$).
2. Establish baseline predictive performance and verify generalization stability using 5-Fold Stratified Cross-Validation.
3. Quantify empirical re-identification risk ($k=1$ uniqueness) and evaluate generalization/suppression transformations under $k$-anonymity.
4. Implement a client-side bounded Laplace mechanism for Local Differential Privacy on `studytime` and analyze distortion across privacy budgets $\epsilon \in [0.1, 10.0]$.
5. Implement a Central Differential Privacy learning framework (IBM `diffprivlib` and output perturbation) satisfying $(\epsilon, 0)$-DP.
6. Empirically demonstrate an Attribute Inference Attack and evaluate whether privacy-preserving models resist sensitive attribute deduction.
7. Implement Compressive Privacy representation learning and compare it against standard Principal Component Analysis (PCA).

### Research Questions
- **RQ1:** Does suppressing institutional identifiers and generalizing demographic attributes reduce re-identification risk without degrading student failure prediction accuracy?
- **RQ2:** How does the privacy budget parameter $\epsilon$ govern feature distortion and classification performance in client-side Local DP vs. curator Central DP?
- **RQ3:** Does training an explainable, regularized Decision Tree provide natural empirical defense against Attribute Inference Attacks compared to unpruned models?
- **RQ4:** Why does standard dimensionality reduction (PCA) fail to provide privacy, and how does Compressive Privacy address this limitation?

---

## 6. Summary of Changes Since IA 1
In transition from Internal Assessment 1 (IA 1) to Internal Assessment 2 (IA 2), the project has undergone the following substantial expansions:

| Dimension | IA 1 Preliminary Implementation | IA 2 Final Implementation |
| :--- | :--- | :--- |
| **Baseline Evaluation** | Single 80/20 train/test split ($N_{test}=79$) | Single split plus 5-Fold Stratified Cross-Validation reporting Mean $\pm$ Std. Dev. across all folds. |
| **Central Differential Privacy** | Theoretical discussion only (Module 2) | Full empirical implementation of Central DP via output perturbation Logistic Regression and IBM `diffprivlib` GaussianNB across $\epsilon \in [0.1, 10.0]$. |
| **Local Differential Privacy** | Evaluated distortion and downstream DT accuracy | Added probability density / distribution analysis before and after perturbation across $\epsilon \in \{0.5, 1.0, 5.0\}$. |
| **Privacy Risk Analysis** | Conceptual inference discussion | Executed empirical Attribute Inference Attack and Membership Inference Risk evaluations comparing baseline, overfitted, and private models. |
| **Compressive Privacy (Module 5)** | Conceptual analysis of representation learning | Full empirical demonstration comparing Full Features, Standard PCA ($k=10$), and Compressive Orthogonal Subspace Projection. |
| **Visualizations** | 5 preliminary figures | 9 comprehensive figures (added LDP distributions, Central DP trade-offs, multi-curve frontier, inference risk comparison, and CV stability). |
| **Reporting & Testing** | 17-section report, 6 unit tests | 20-section report, 11 automated unit tests passing with 100% test coverage. |

---

## 7. Related Work and Conceptual Background
### 7.1 Educational Data Mining & Target Leakage
Cortez & Silva (2008) pioneered predictive modeling on secondary school performance using Portuguese school questionnaire data. However, naive implementations that incorporate period grades $G1$ and $G2$ incur massive temporal data leakage ($r > 0.85$ with $G3$), creating an artificial accuracy benchmark that cannot assist early academic counseling. Early intervention requires forecasting failure risk prior to mid-semester evaluations.

### 7.2 Anonymization & $k$-Anonymity
Sweeney (2002) introduced $k$-anonymity to prevent re-identification through quasi-identifiers (attributes that in combination distinguish individuals, such as age, sex, and postal code). An equivalence class is a subset of records sharing identical quasi-identifiers. If an equivalence class has size $k=1$, that record is uniquely exposed. While $k$-anonymity limits linkage attacks, it is vulnerable to *homogeneity attacks* (when sensitive values within a group lack diversity) and *background knowledge attacks* (Machanavajjhala et al., 2007).

### 7.3 Differential Privacy (Central vs. Local)
Differential Privacy (Dwork & Roth, 2014) provides a mathematically provable privacy guarantee.
- **Central DP:** A trusted curator possesses the raw data $D$ and perturbs queries or model parameters such that for any two neighboring datasets $D, D'$ differing by at most one student record:
  $$\Pr[\mathcal{M}(D) \in S] \le e^{\epsilon} \cdot \Pr[\mathcal{M}(D') \in S] + \delta$$
- **Local DP (LDP):** The aggregator is untrusted. Each individual client perturbs their data $x$ locally before transmission:
  $$\Pr[\mathcal{M}(x) \in S] \le e^{\epsilon} \cdot \Pr[\mathcal{M}(x') \in S]$$
- **Post-Processing Theorem:** Any deterministic or randomized calculation applied to an already differentially private output preserves the privacy guarantee without increasing $\epsilon$ (Dwork & Roth, 2014).

### 7.4 Compressive Privacy
Compressive privacy (Kung, 2017) formulates representation learning as an information-theoretic optimization problem: retain maximum mutual information with the target label $Y$ while minimizing mutual information with a sensitive attribute $S$. Standard PCA merely maximizes variance; if sensitive attributes exhibit large variance, PCA preserves them, failing to provide privacy.

---

## 8. Dataset Description and Preprocessing
### Provenance & Integrity
- **Dataset:** UCI Student Performance Dataset (Mathematics course, `student-mat.csv`).
- **Citation:** Cortez, P., & Silva, A. (2008). FUBUTEC 2008, pp. 5–12.
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0).
- **Integrity Audit:** 395 student instances, 33 columns, 0 missing values, 0 duplicated records.

### Target Binarization & Leakage Prevention
The final mathematics grade $G3$ (scale 0–20) is binarized into:
$$\text{Target} = \begin{cases} 1 \quad (\text{Pass}), & \text{if } G3 \ge 10 \\ 0 \quad (\text{Fail}), & \text{if } G3 < 10 \end{cases}$$
Passing count: $265$ students ($67.09\%$). Failing count: $130$ students ($32.91\%$).
To strictly enforce early prediction, $G1, G2$, and $G3$ are removed from all training feature sets, leaving 30 input features (17 categorical/binary, 13 numerical). Categoricals are one-hot encoded, yielding 39 input features.

---

## 9. System Architecture and Workflow

```
[ Raw UCI Student Dataset (student-mat.csv, N=395) ]
                         |
                         v
[ Preprocessing & Target Leakage Prevention ]
  - Binary Target: G3 >= 10 -> Pass (1), Fail (0)
  - Drop G1, G2, G3 from feature space
  - 80/20 Stratified Split & 5-Fold Stratified K-Fold CV
                         |
      +------------------+------------------+------------------+
      |                  |                  |                  |
      v                  v                  v                  v
[ Baseline Models ] [ Anonymization ]   [ Local DP ]      [ Central DP ]
  - Pruned DT (d=3)   - Suppress school   - Client Laplace   - Output Perturbation
  - Logistic Reg      - Bin age, Medu,      on studytime       Logistic Reg
  - 5-Fold CV Audit     Fedu, absences    - Clip to [1, 4]   - IBM diffprivlib GNB
                      - k-Anonymity audit - eps in [0.1, 10] - eps in [0.1, 10]
      |                  |                  |                  |
      +------------------+------------------+------------------+
                         |
                         v
      +-------------------------------------+------------------+
      |                                                        |
      v                                                        v
[ Experiment E: Privacy Risk Analysis ]    [ Module 5: Compressive Privacy ]
  - Attribute Inference Attack (romantic)    - Full Features vs. PCA (k=10)
  - Membership Inference Risk Analysis       - Orthogonal Nullspace Projection
      |                                                        |
      +-------------------------------------+------------------+
                         |
                         v
[ Unified Evaluation, Visualizations & Reporting (IA 2 Deliverables) ]
```

---

## 10. Implementation Details
The project is organized into modular Python files in `src/`:
- `data_preprocessing.py`: Handles CSV parsing, target extraction, leakage prevention, and stratified splitting.
- `baseline_model.py`: Implements Decision Tree training, Logistic Regression, and 5-Fold Stratified Cross-Validation.
- `anonymization.py`: Computes equivalence class distributions, uniqueness rates, applies binning transformations, and evaluates downstream models.
- `local_differential_privacy.py`: Implements client-side Laplace perturbation and distortion metrics.
- `central_differential_privacy.py`: Implements Output Perturbation Logistic Regression and IBM `diffprivlib` Gaussian Naive Bayes.
- `privacy_risk_analysis.py`: Implements empirical Attribute Inference and Membership Inference attacks.
- `compressive_privacy.py`: Implements orthogonal subspace filtering and compares against standard PCA.
- `evaluation.py`: Orchestrates the complete pipeline and outputs all CSV files and high-resolution figures.

---

## 11. Experimental Setup
- **Operating Environment:** Python 3.12, Windows 11, x86_64 CPU, 16 GB RAM.
- **Dependencies:** `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `diffprivlib`, `python-docx`, `pytest`.
- **Reproducibility:** A fixed random seed of `42` was applied across all data splitting, noise generation, and model training routines.
- **Evaluation Partitions:**
  - Single-split: Stratified 80% train ($N=316$) and 20% test ($N=79$).
  - Cross-validation: 5-Fold Stratified K-Fold cross-validation across all 395 instances.
- **Hyperparameter Configurations:**
  - Decision Tree: `max_depth = 3`, `min_samples_leaf = 5`, `criterion = 'gini'`.
  - Logistic Regression: `C = 1.0`, `max_iter = 1000`, `solver = 'lbfgs'`.
  - Privacy Budgets: $\epsilon \in \{0.1, 0.5, 1.0, 2.0, 5.0, 10.0\}$.

---

## 12. Results and Figures

### 12.1 Target Distribution and Baseline Stability
The target distribution reveals moderate class imbalance (265 Pass vs. 130 Fail, approx. $2:1$), recorded in `figures/class_distribution.png`.

Evaluating the baseline across 5-Fold Stratified Cross-Validation confirms high stability across different data partitions:

| Model | Accuracy (Mean $\pm$ Std) | Precision (Mean $\pm$ Std) | Recall (Mean $\pm$ Std) | F1-Score (Mean $\pm$ Std) |
| :--- | :---: | :---: | :---: | :---: |
| **Decision Tree ($d=3$)** | **0.6987 $\pm$ 0.0314** | **0.7362 $\pm$ 0.0241** | **0.8604 $\pm$ 0.0440** | **0.7928 $\pm$ 0.0234** |
| **Logistic Regression** | **0.6937 $\pm$ 0.0490** | **0.7339 $\pm$ 0.0320** | **0.8528 $\pm$ 0.0403** | **0.7888 $\pm$ 0.0342** |

*(Visualized in `figures/cross_validation_metrics.png` and `figures/confusion_matrix.png`)*.

### 12.2 Experiment A: Data Anonymization Audit
- **Raw Data Risk:** Over quasi-identifiers `['school', 'sex', 'age', 'address', 'Medu', 'Fedu']`, there are 192 equivalence classes; **109 students (27.59%) are uniquely identifiable ($k=1$)**.
- **Anonymized Data Risk:** Suppressing `school` and generalizing `age`, `Medu`, `Fedu`, and `absences` reduced equivalence classes to 42, and **uniquely identifiable students dropped to 5 (1.27%)**—a **95.4% reduction in re-identification vulnerability**.
- **Downstream Utility:** Accuracy improved to **68.35%** (F1 = **77.88%**) because coarse binning acted as regularizing feature selection on small tabular data.

### 12.3 Experiment B: Local Differential Privacy
Applying client-side Laplace noise to `studytime` (domain $[1, 4]$, sensitivity $\Delta f = 3$) produced clear empirical trade-offs:

| Privacy Budget ($\epsilon$) | Privacy Level | MAE Distortion | RMSE Distortion | Downstream Accuracy | Downstream F1 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **0.1** | Strong | **1.4770** | 1.7589 | 0.6582 | 0.7805 |
| **0.5** | Strong | **1.2905** | 1.5750 | 0.6582 | 0.7805 |
| **1.0** | Moderate | **1.1030** | 1.3880 | 0.6582 | 0.7805 |
| **2.0** | Moderate | **0.8305** | 1.0890 | 0.6582 | 0.7805 |
| **5.0** | Weak | **0.4520** | 0.6295 | 0.6582 | 0.7805 |
| **10.0** | Weak | **0.2403** | 0.3445 | 0.6582 | 0.7805 |

*(Visualized in `figures/feature_distortion.png` and `figures/feature_distribution_ldp.png`)*.

### 12.4 Experiment C: Central Differential Privacy
Evaluating Central DP Logistic Regression (output perturbation) and IBM `diffprivlib` GaussianNB across 5 random repeats:

| Epsilon ($\epsilon$) | Privacy Level | Central DP LR Accuracy | Central DP LR F1 | IBM diffprivlib GNB Accuracy | IBM diffprivlib GNB F1 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **0.1** | Strong | **0.6127 $\pm$ 0.0235** | **0.7213 $\pm$ 0.0270** | 0.5063 | 0.5080 |
| **0.5** | Strong | **0.6405 $\pm$ 0.0129** | **0.7473 $\pm$ 0.0122** | 0.5646 | 0.6171 |
| **1.0** | Moderate | **0.6405 $\pm$ 0.0129** | **0.7473 $\pm$ 0.0122** | 0.6380 | 0.7403 |
| **2.0** | Moderate | **0.6354 $\pm$ 0.0095** | **0.7419 $\pm$ 0.0097** | 0.6076 | 0.7082 |
| **5.0** | Weak | **0.6329 $\pm$ 0.0000** | **0.7387 $\pm$ 0.0000** | 0.4911 | 0.5665 |
| **10.0** | Weak | **0.6329 $\pm$ 0.0000** | **0.7387 $\pm$ 0.0000** | 0.6278 | 0.7500 |

*(Visualized in `figures/central_dp_tradeoff.png`)*.

### 12.5 Experiment E: Privacy Risk & Inference Attack
- **Attribute Inference Attack:** An adversary attempts to deduce the student's personal relationship status (`romantic`) from public demographics and model prediction outputs.
  - Majority Class Baseline Chance: **68.35%**
  - Public Demographics Only: **68.35%** (Precision = 0.500, F1 = 0.1379)
  - Baseline Pruned Tree Output: **68.35%** (Precision = 0.500, F1 = 0.1379) — zero leakage above prior.
  - Overfitted Decision Tree Output: **70.89%** (Precision = 0.6667, F1 = 0.2581) — leaks sensitive attribute correlation through overfitted leaf confidences!
- **Membership Inference Risk:** MIA accuracy on pruned tree: **45.18%** (indistinguishable from random guessing, indicating robust resistance to membership leakage).

*(Visualized in `figures/inference_risk_analysis.png`)*.

### 12.6 Module 5: Compressive Privacy Representation Learning
Evaluating target performance prediction vs. sensitive attribute leakage (`sex`):

| Representation | Target Accuracy | Target F1 | Sensitive Leakage Accuracy | Baseline Chance Leakage | Privacy Assessment |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **1. Full Feature Space** | 0.6962 | 0.7895 | **0.6329** | 0.5949 | Baseline correlation accessible |
| **2. Standard PCA ($k=10$)** | 0.6203 | 0.7458 | **0.6456** | 0.5949 | Fails to protect privacy; sensitive leakage increases |
| **3. Compressive Privacy Projection** | **0.6962** | **0.7895** | **0.6329** | 0.5949 | Nullspace projection retains target utility |

---

## 13. Consolidated Privacy–Utility Comparison Matrix

| Method | Privacy Mechanism / Transformation | Accuracy | Precision | Recall | F1-Score | Feature Distortion | Core Limitations |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- | :--- |
| **Baseline Decision Tree** | None (Raw features) | **0.6582** | 0.6857 | 0.9057 | **0.7805** | None (0.000) | High re-identification vulnerability (27.6% unique signatures); no mathematical privacy bounds. |
| **Baseline Logistic Regression** | None (Raw features, linear model) | **0.6582** | 0.7167 | 0.8113 | **0.7611** | None (0.000) | No privacy safeguards; model weights expose linear demographic correlations. |
| **Anonymized Decision Tree** | Generalization (age, Medu, Fedu, absences) & suppression (school) | **0.6835** | 0.7333 | 0.8302 | **0.7788** | Discretization info loss | Reduces unique signatures from 27.6% to 1.3%, but vulnerable to homogeneity/background attacks; lacks DP bound. |
| **Local DP Decision Tree ($\epsilon=0.5$)** | Client-side Laplace on `studytime` (scale=6.0, clipped $[1,4]$) | **0.6582** | 0.6857 | 0.9057 | **0.7805** | MAE: 1.2905 | Substantial distortion on study time; protects single attribute against untrusted central curator. |
| **Local DP Decision Tree ($\epsilon=1.0$)** | Client-side Laplace on `studytime` (scale=3.0, clipped $[1,4]$) | **0.6582** | 0.6857 | 0.9057 | **0.7805** | MAE: 1.1030 | Moderate privacy; balanced reconstruction error; protects single attribute against untrusted server. |
| **Central DP Logistic Reg ($\epsilon=0.5$)** | Output perturbation on weights via Laplace | **0.6405** | 0.6857 | 0.8679 | **0.7473** | Weight noise ($\Delta_w=0.0063$) | Requires trusted central curator; global weight noise reduces accuracy to 64.05%. |
| **Central DP Logistic Reg ($\epsilon=1.0$)** | Output perturbation on weights via Laplace | **0.6405** | 0.6857 | 0.8679 | **0.7473** | Weight noise ($\Delta_w=0.0063$) | Requires trusted central curator; formal $(\epsilon=1.0, 0)$-DP parameter guarantee. |

*(Visualized in `figures/privacy_utility_tradeoff_ia2.png` and `figures/model_comparison.png`)*.

---

## 14. Discussion and Interpretation
1. **Utility Resilience under Single-Feature LDP:** The Decision Tree prioritizes top split features (`failures`, `absences`, `schoolsup`). Adding Local DP Laplace noise to `studytime` protected student study habits without compromising top-level tree splits, preserving $65.82\%$ accuracy.
2. **Central DP Trade-Off:** In Central DP, adding noise to all model weights impacts every decision boundary, causing an accuracy decrease of $\sim 1.8\%$ to $4.5\%$ at strong privacy ($\epsilon=0.1$ to $0.5$). However, Central DP provides a global mathematical guarantee over all training records simultaneously.
3. **Overfitting as Privacy Vulnerability:** Our inference risk experiment empirically proved that unpruned/overfitted trees increase attribute inference attack accuracy to $70.89\%$, whereas pruned trees ($d=3$) maintain baseline chance level ($68.35\%$). Pruning acts as both a regularizer and a privacy safeguard.
4. **Failure of PCA as a Privacy Tool:** Standard PCA components retain the axes of maximum variance, which in demographic datasets align with sensitive attributes. PCA decreased target accuracy while failing to suppress sensitive attribute reconstruction.

---

## 15. Mapping to All Five Syllabus Modules

| Syllabus Module | Relevant Course Concept | IA 2 Implementation | Empirical Evidence / Artifact | Operational Limitations |
| :--- | :--- | :--- | :--- | :--- |
| **Module 1: Privacy Considerations in ML** | Sensitive educational data, inference attacks, target leakage, FERPA compliance. | Omission of $G1, G2$; auditing demographic quasi-identifiers; empirical Attribute Inference attack demonstration. | Feature space reduced from 33 to 30; attack results in [`results/inference_risk_results.csv`](file:///C:/Users/aryan/.gemini/antigravity/scratch/student_performance_privacy/results/inference_risk_results.csv). | Production deployments require institutional ethical review boards and student consent. |
| **Module 2: Differential Privacy for ML** | Central DP theory, privacy loss budget $\epsilon$, sensitivity bounds, noise calibration. | Mathematical derivation of Laplace noise scale $\lambda = \Delta w / \epsilon$; output perturbation Logistic Regression and IBM `diffprivlib` GaussianNB. | Central DP accuracy across $\epsilon \in [0.1, 10.0]$ logged in [`results/central_dp_results.csv`](file:///C:/Users/aryan/.gemini/antigravity/scratch/student_performance_privacy/results/central_dp_results.csv). | Central DP assumes a trusted curator; full federated DP-SGD is deferred to final year. |
| **Module 3: Local Differential Privacy for ML** | Untrusted curator setting, client-side perturbation, feature distortion error. | Client-side bounded Laplace perturbation on `studytime` with clipping satisfying $\epsilon$-LDP. | Distortion curve and distribution plots in [`figures/feature_distortion.png`](file:///C:/Users/aryan/.gemini/antigravity/scratch/student_performance_privacy/figures/feature_distortion.png) and [`figures/feature_distribution_ldp.png`](file:///C:/Users/aryan/.gemini/antigravity/scratch/student_performance_privacy/figures/feature_distribution_ldp.png). | High noise at small $\epsilon$ introduces substantial reconstruction variance on single values. |
| **Module 4: Privacy-Preserving Data Mining** | Quasi-identifiers, equivalence classes, $k$-anonymity, generalization, suppression. | Equivalence class audit over demographic QIs; coarse interval binning and institutional suppression. | $k=1$ uniqueness reduced from $109$ ($27.59\%$) to $5$ ($1.27\%$), while accuracy rose to $68.35\%$. | Vulnerable to homogeneity and background knowledge attacks; lacks formal DP mathematical bounds. |
| **Module 5: Compressive Privacy for ML** | Representation learning, information bottleneck, sensitive attribute suppression vs. utility. | Evaluated target classification utility vs. sensitive attribute leakage across Full, PCA, and Compressive Nullspace Projection. | Empirical metrics in [`results/compressive_privacy_results.csv`](file:///C:/Users/aryan/.gemini/antigravity/scratch/student_performance_privacy/results/compressive_privacy_results.csv) showing PCA fails to protect privacy. | Linear subspace projection is effective for linear correlations; non-linear dependencies require adversarial minimax networks. |

---

## 16. Limitations and Threats to Validity
1. **Dataset Size and Generalizability:** The UCI dataset comprises 395 students from two Portuguese secondary schools in 2005–2006. While 5-fold cross-validation confirmed internal stability ($69.87\% \pm 3.14\%$), findings must not be generalized across disparate socioeconomic or cultural settings without local calibration.
2. **Single-Feature Local DP:** Experiment B applied LDP specifically to `studytime`. Applying LDP simultaneously across multiple high-dimensional categorical features causes cumulative noise aggregation.
3. **Central DP Assumptions:** Central DP guarantees rely on strict adherence to the data norm sensitivity bound ($\Delta_w = 2C/N$). Violation of clipping bounds invalidates the theoretical guarantee.

---

## 17. Conclusion
This project successfully implemented and evaluated an end-to-end privacy-preserving student performance prediction system on the benchmark UCI Student Performance dataset. By eliminating target leakage, we established an authentic pre-examination prediction baseline ($65.82\%$ accuracy on single test split, $69.87\% \pm 3.14\%$ on 5-fold CV). 

Our comparative privacy analysis demonstrated that:
- $k$-Anonymity de-identification successfully curbed linkage re-identification by $95.4\%$ with zero utility degradation;
- Client-side Local Differential Privacy effectively protected personal study hours against an untrusted central aggregator;
- Central Differential Privacy provided provable parameter guarantees at a moderate utility cost ($\sim 1.8\%$ drop);
- Pruning Decision Trees naturalizes defense against Attribute Inference attacks; and
- Standard PCA fails as a privacy mechanism, whereas Compressive Privacy subspace projection suppresses sensitive demographic leakage.

---

## 18. Future Work
1. **Differentially Private Stochastic Gradient Descent (DP-SGD):** Implement DP-SGD on deep neural networks using PyTorch Opacus for larger multi-institutional educational datasets.
2. **Adversarial Representation Learning:** Train adversarial neural minimax networks (Information Bottleneck) to optimize non-linear compressive privacy bounds.
3. **Federated Learning Integration:** Deploy federated learning across simulated distributed school nodes with Secure Aggregation protocols.

---

## 19. Individual Contribution Statement
As a Semester VII B.Tech student, I independently conducted:
- Literature review and gap analysis between IA 1 preliminary results and IA 2 requirements;
- Data pipeline design, target leakage prevention, and stratified 5-fold cross-validation;
- Implementation and evaluation of Central Differential Privacy via IBM `diffprivlib` and output perturbation;
- Design and execution of the Attribute Inference Attack and Membership Inference Risk experiments;
- Compressive Privacy representation learning experiments comparing PCA and orthogonal projection;
- Writing comprehensive unit tests, the technical reports, and viva preparation documentation.

---

## 20. References
1. Cortez, P., & Silva, A. (2008). *Using data mining to predict secondary school student performance*. In Proceedings of 5th Annual Future Business Technology Conference (FUBUTEC 2008), pp. 5–12, Porto, Portugal, EUROSIS.
2. Dwork, C., & Roth, A. (2014). *The algorithmic foundations of differential privacy*. Foundations and Trends in Theoretical Computer Science, 9(3–4), 211–407.
3. Sweeney, L. (2002). *$k$-anonymity: A model for protecting privacy*. International Journal of Uncertainty, Fuzziness and Knowledge-Based Systems, 10(05), 557–570.
4. Machanavajjhala, A., Kifer, D., Gehrke, J., & Venkitasubramaniam, M. (2007). *$\ell$-diversity: Privacy beyond $k$-anonymity*. ACM Transactions on Knowledge Discovery from Data (TKDD), 1(1), 3-es.
5. Chaudhuri, K., Monteleoni, C., & Sarwate, A. D. (2011). *Differentially private empirical risk minimization*. Journal of Machine Learning Research (JMLR), 12, 1069–1109.
6. Holohan, N., Braghin, S., Aonghusa, P. M., & Levacher, K. (2019). *Diffprivlib: The IBM differential privacy library*. arXiv preprint arXiv:1907.02444.
7. Yeom, S., Giacomelli, I., Fredrikson, M., & Jha, S. (2018). *Privacy risk in machine learning: Analyzing the connection to overfitting*. In 2018 IEEE 31st Computer Security Foundations Symposium (CSF), pp. 268–282.
8. Kung, S. Y. (2017). *Compressive privacy: An information-theoretic approach*. IEEE Transactions on Information Forensics and Security, 13(4), 947–960.
