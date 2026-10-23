# Internal Assessment 1 (IA 1) Preliminary Technical Report
## Privacy-Preserving Student Performance Prediction Using Machine Learning

---

### Course Information
- **Course Name:** Security and Privacy for Machine Learning
- **Assessment Type:** Internal Assessment 1 (IA 1) — Preliminary Implementation & Research Progress Report
- **Student Name:** Aryan
- **Degree / Semester:** Bachelor of Technology (B.Tech), Semester VII
- **Academic Year:** 2026-2027

---

## 1. Project Details and Title
- **Project Title:** Privacy-Preserving Student Performance Prediction Using Machine Learning
- **Primary Domain:** Machine Learning, Privacy-Preserving Data Mining, Differential Privacy, Educational Data Mining (EDM).
- **Primary Learning Algorithm:** Pruned Decision Tree Classifier (with Logistic Regression baseline comparison).
- **Core Privacy Paradigms:** Data Anonymization ($k$-anonymity, Quasi-Identifier Uniqueness Analysis) and Local Differential Privacy (Bounded Laplace Perturbation Mechanism).
- **Target Feature:** Final Academic Outcome ($G3 \ge 10 \rightarrow \text{Pass (1)}, G3 < 10 \rightarrow \text{Fail (0)}$).
- **Target Leakage Safeguard:** Strict omission of period grades $G1$ and $G2$ from the input feature space.

---

## 2. Abstract / Executive Summary
Educational data mining models offer substantial potential for early scholastic interventions by predicting which students are at risk of academic failure. However, educational records encapsulate highly sensitive personal information, including family cohabitation status, parental education levels, home address types, study habits, and disciplinary absences. Training conventional machine learning models directly on raw, centralized student records introduces severe privacy risks, including linkage-based re-identification and sensitive attribute inference. 

This project investigates the empirical trade-off between privacy guarantees and downstream classification utility for student performance prediction on the benchmark UCI Student Performance dataset ($N=395$). To reflect realistic institutional deployment, we eliminate target leakage by excluding intermediate examination grades ($G1, G2$), compelling the model to predict outcomes using demographic, behavioral, and environmental indicators. 

We evaluate three modeling regimes:
1. A baseline interpretable Decision Tree achieving an accuracy of $65.82\%$ and an F1-score of $78.05\%$, but exposing $27.59\%$ of the student cohort as uniquely re-identifiable via quasi-identifier combinations;
2. A generalized, de-identified feature space that suppresses institutional identifiers and discretizes quasi-identifiers, slashing uniqueness to $1.27\%$ while preserving predictive utility (accuracy $68.35\%$, F1-score $77.88\%$); and
3. A client-side Local Differential Privacy (LDP) framework applying a bounded Laplace perturbation mechanism to student weekly study hours across privacy budgets $\epsilon \in [0.1, 10.0]$, empirically charting feature distortion from $\text{MAE}=1.4770$ ($\epsilon=0.1$) down to $\text{MAE}=0.2403$ ($\epsilon=10.0$).

The preliminary findings indicate that moderate anonymization and localized feature perturbation maintain strong model utility while substantially curbing individual exposure risks.

---

## 3. Problem Statement
Modern educational platforms systematically aggregate multifaceted student data. While predicting academic failure early in a semester allows institutions to allocate tutoring and pastoral support, transmitting and storing unencrypted, granular student records creates grave vulnerabilities:
- **Linkage Attacks:** An external adversary possessing auxiliary information (e.g., voter registration lists, public sports rosters) can join public demographic data with scholastic records to re-identify individual students.
- **Internal Curiosity and Untrusted Aggregators:** Centralized school administrators or cloud data brokers may observe intimate details, such as alcohol consumption, parental divorce, or personal study habits.
- **Model Inversion and Membership Inference:** Trained model weights and prediction probabilities can leak whether a specific student's record was part of the training set.

The technical challenge addressed in this work is: **How can an educational institution train an effective, explainable machine learning model to predict student failure early in the academic cycle while providing verifiable privacy safeguards for sensitive student attributes?**

---

## 4. Motivation and Significance
1. **Regulatory and Legal Imperatives:** Regulations such as the Family Educational Rights and Privacy Act (FERPA in the United States) and the General Data Protection Regulation (GDPR in the European Union) legally mandate data minimization, purpose limitation, and the protection of student records against unauthorized disclosure.
2. **Ethical AI and Student Trust:** If students suspect that reporting honest lifestyle habits (e.g., study hours, home life) will lead to surveillance or stigmatization, they may submit false data or refuse participation, sabotaging institutional analytics.
3. **Early-Intervention Necessity without Data Leakage:** Naive educational prediction pipelines frequently include earlier period grades ($G1, G2$), achieving artificial accuracies exceeding $90\%$. However, waiting for mid-term grades delays remedial intervention until it is too late. An early-warning model must function on pre-examination features, where privacy protection is most acute.

---

## 5. Objectives and Expected Outcomes
### Core Objectives
1. **Develop a Leakage-Free Prediction Pipeline:** Formulate a binary classification pipeline predicting final mathematics outcome ($G3 \ge 10$) without relying on $G1$ or $G2$.
2. **Establish a Transparent Baseline:** Implement a pruned Decision Tree classifier that is inherently explainable and easy to present in an academic viva.
3. **Quantify Re-Identification Risk & Apply Anonymization (Experiment B):** Measure empirical quasi-identifier uniqueness rates ($k=1$ occurrences) on raw data, apply generalization/suppression transformations, and measure subsequent utility retention.
4. **Implement Mathematically Grounded Local DP (Experiment C):** Develop a client-side bounded Laplace mechanism for the `studytime` attribute, satisfying $\epsilon$-LDP via the post-processing theorem, and measure reconstruction distortion across privacy budgets $\epsilon \in [0.1, 10.0]$.
5. **Synthesize the Privacy–Utility Frontier (Experiment D):** Formulate a comprehensive comparison matrix and trade-off visualizations mapping privacy expenditure to predictive accuracy and F1-score.

### Expected Outcomes
- An operational Python package and interactive Jupyter Notebook demonstrating all four experimental stages.
- Quantifiable evidence demonstrating that raw data exposes over a quarter of students to re-identification, whereas generalization mitigates this risk by over $95\%$ with zero utility degradation.
- Mathematical validation of Local Differential Privacy with clear bounding of feature distortion.

---

## 6. Scope and Limitations
### In Scope
- Secondary school mathematics academic dataset (`student-mat.csv`, $N=395$).
- Binary classification of passing status ($G3 \ge 10$).
- Feature preprocessing, one-hot encoding, and leakage prevention.
- Decision Tree Classifier as the primary explainable model, with Logistic Regression for comparative evaluation.
- Generalization, suppression, and quasi-identifier uniqueness analysis.
- Single-attribute client-side Local Differential Privacy on bounded continuous/discrete study time.
- Unit testing, reproducibility validation, and technical documentation.

### Out of Scope / Limitations
- Multi-party federated learning or distributed cryptographic protocols (deferred to IA 2 / Final project).
- Differential Privacy for the entire training objective (e.g., DP-SGD) on the full feature space; this report focuses on client-side feature-level Local DP.
- Generalizability to external non-academic demographic populations without localized calibration.

---

## 7. Conceptual Background
### 7.1 Decision Tree Classifier
A Decision Tree partitions the feature space $\mathcal{X}$ into axis-aligned rectangular regions by recursively choosing feature splits that maximize information gain or minimize Gini impurity:
$$\text{Gini}(D) = 1 - \sum_{i=1}^C p_i^2$$
Decision Trees provide unmatched interpretability for educational stakeholders because each branch directly represents an explicit educational decision rule (e.g., `failures <= 0.5 AND absences <= 15.5 -> Pass`). In our implementation, we enforce `max_depth=3` and `min_samples_leaf=5` to prevent overfitting on 316 training samples.

### 7.2 Data Anonymization and $k$-Anonymity
A dataset provides $k$-anonymity if each record in the dataset is indistinguishable from at least $k-1$ other records with respect to a designated set of Quasi-Identifiers (QIs) (Sweeney, 2002). An equivalence class is a subset of records sharing identical QI values. When an equivalence class has size $k=1$, the student is uniquely identifiable, making them vulnerable to linkage attacks against external databases.
- *Limitation:* $k$-Anonymity does not protect against attribute disclosure when all members of an equivalence class share the same sensitive target (homogeneity attack) or when an adversary has prior background knowledge.

### 7.3 Differential Privacy (Central vs. Local)
**Central Differential Privacy:** A trusted curator aggregates raw data and adds calibrated noise to query responses. An algorithm $\mathcal{M}$ satisfies $\epsilon$-differential privacy if for any two neighboring datasets $D, D'$ differing by at most one individual:
$$\Pr[\mathcal{M}(D) \in S] \le e^{\epsilon} \cdot \Pr[\mathcal{M}(D') \in S]$$

**Local Differential Privacy (LDP):** In scenarios where the central aggregator is untrusted, perturbation is executed on the client device prior to collection:
$$\Pr[\mathcal{M}(x) \in S] \le e^{\epsilon} \cdot \Pr[\mathcal{M}(x') \in S] \quad \forall x, x' \in \mathcal{D}$$
Because the perturbation occurs before transmission, the raw attribute is never revealed.
- *Post-Processing Immunity:* If a mechanism $\mathcal{M}$ satisfies $\epsilon$-DP, then for any arbitrary mapping $g$, the composite mechanism $g \circ \mathcal{M}$ satisfies $\epsilon$-DP (Dwork & Roth, 2014). This permits clipping noisy values to the valid domain $[1, 4]$ without privacy degradation.

---

## 8. Mapping to All Five Syllabus Modules

| Syllabus Module | Relevant Course Concept | Project Implementation / Analysis | Empirical Evidence / Artifact | Operational Limitations |
| :--- | :--- | :--- | :--- | :--- |
| **Module 1: Privacy Considerations in ML** | Sensitive academic attributes, inference risks, ethical data stewardship, target leakage. | Strict exclusion of period grades $G1, G2$; auditing demographic quasi-identifiers (`address, Medu, Fedu`). | Feature matrix reduction from 33 to 30 features; class distribution analysis (`class_distribution.png`). | Real-world deployments require institutional ethical review boards and consent mechanisms. |
| **Module 2: Differential Privacy for ML** | Central DP theory, privacy loss budget $\epsilon$, sensitivity bounds, noise calibration. | Mathematical derivation of Laplace noise scale $\lambda = \Delta f / \epsilon$; discussion of central vs. local trade-offs. | Formal sensitivity calculation $\Delta f = 4 - 1 = 3$; budget evaluation across $\epsilon \in [0.1, 10.0]$. | Central DP requires a trusted curator; DP-SGD implementation deferred to IA 2. |
| **Module 3: Local Differential Privacy for ML** | Untrusted aggregator setting, client-side perturbation, reconstruction error. | Implementation of client-side bounded Laplace perturbation on `studytime` with post-processing clipping. | Feature distortion curve (`feature_distortion.png`) charting MAE from $1.4770$ ($\epsilon=0.1$) to $0.2403$ ($\epsilon=10.0$). | High noise at small $\epsilon$ introduces significant variance on individual values. |
| **Module 4: Privacy-Preserving Data Mining** | Quasi-identifiers, equivalence classes, $k$-anonymity, generalization, suppression. | Quantitative evaluation of $k=1$ uniqueness over demographic QIs; transformation via coarse binning and school suppression. | Reduction in unique students from $109$ ($27.59\%$) down to $5$ ($1.27\%$), while maintaining $68.35\%$ accuracy. | Vulnerable to homogeneity and background knowledge attacks; lacks formal DP mathematical bounds. |
| **Module 5: Compressive Privacy for ML** | Representation learning, information bottleneck, sensitive attribute suppression vs. utility. | Discretization and feature subspace compression; evaluation of whether pruned tree branches suppress demographic cues. | Downstream model maintains $77.88\%$ F1-score on compressed generalized feature representation. | Dimensionality reduction (e.g., standard PCA) preserves variance, not privacy; mutual information suppression requires adversarial minimax training. |

---

## 9. Dataset Provenance and Responsible Data Use
### Provenance and Citation
The dataset used is the **Student Performance Dataset** hosted by the UC Irvine Machine Learning Repository:
- **Repository URL:** `https://archive.ics.uci.edu/dataset/320/student+performance`
- **Primary Citation:** Cortez, P., & Silva, A. (2008). *Using data mining to predict secondary school student performance*. In Proceedings of 5th Annual Future Business Technology Conference (FUBUTEC 2008), pp. 5–12, Porto, Portugal.
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0).

### Dataset Characteristics
- Total instances: 395 student records (Mathematics course, `student-mat.csv`).
- Total attributes: 33 attributes (demographic, social, academic, and health indicators).
- Data integrity: Zero missing values, zero duplicated records.
- Class balance: $265$ students passing ($67.09\%$), $130$ students failing ($32.91\%$).

### Responsible Data Use & Ethical Statement
1. **Educational Research Purpose:** The dataset is used solely for educational and privacy methodology research. It reflects secondary school records collected in the Alentejo region of Portugal (2005–2006). Results must not be interpreted as definitive generalizations regarding student capabilities across diverse cultural contexts.
2. **De-Identification by Design:** Direct identifiers (names, tax IDs, telephone numbers) are absent. Our analysis proactively investigates residual quasi-identifiers to ensure students cannot be singled out.

---

## 10. Methodology and Workflow Diagram

The end-to-end methodology comprises four interconnected stages:

```
[ Raw UCI Dataset (student-mat.csv) ]
                  |
                  v
[ Leakage Prevention & Preprocessing ]
  - Binarize G3: Pass (>=10) / Fail (<10)
  - Drop G1, G2, G3 from features
  - Stratified Split (80% Train, 20% Test)
                  |
         +--------+--------+-----------------------+
         |                 |                       |
         v                 v                       v
[ Experiment A ]   [ Experiment B ]        [ Experiment C ]
  Baseline DT &      Data Anonymization      Local DP (LDP)
  Logistic Reg       - Suppress school       - Perturb studytime
  - Train & Test     - Bin age, Medu, Fedu     via Laplace(3/eps)
                     - k-Anonymity Audit     - Clip to [1, 4]
                     - Train DT on Anon      - Evaluate across eps
         |                 |                       |
         +--------+--------+-----------------------+
                  |
                  v
[ Experiment D: Privacy–Utility Synthesis ]
  - Generate Comparison Table (Accuracy, Precision, Recall, F1, Distortion)
  - Plot Confusion Matrix, Trade-off Curves, and Model Comparisons
```

---

## 11. Initial Implementation Details
The project is modularized into reusable Python components in the `src/` directory:
- `data_preprocessing.py`: Handles CSV parsing, target extraction, leakage prevention, and stratified splitting.
- `baseline_model.py`: Implements training and evaluation wrappers for Decision Trees and Logistic Regression.
- `anonymization.py`: Computes equivalence class distributions, uniqueness rates, applies binning transformations, and evaluates downstream models.
- `local_differential_privacy.py`: Implements the client-side Laplace perturbation mechanism and feature distortion metrics.
- `evaluation.py`: Orchestrates all experimental pipelines, outputs CSV logs, and renders publication-ready plots.

---

## 12. Experimental Setup
All experiments were conducted in a local Python 3.12 environment under Windows 11.
- **Hardware Platform:** AMD / Intel x86_64 CPU, 16 GB RAM.
- **Random Seed:** A fixed random seed of `42` was applied across all data splitting, noise generation, and model training routines to ensure complete reproducibility.
- **Evaluation Split:** Stratified 80% training set ($N_{train} = 316$) and 20% test set ($N_{test} = 79$).
- **Hyperparameter Configurations:**
  - Decision Tree: `max_depth = 3`, `min_samples_leaf = 5`, `criterion = 'gini'`.
  - Logistic Regression: `max_iter = 1000`, `solver = 'lbfgs'`.
  - LDP Privacy Budgets: $\epsilon \in \{0.1, 0.5, 1.0, 2.0, 5.0, 10.0\}$.

---

## 13. Evaluation Metrics
Because the test set exhibits moderate class imbalance ($53$ Pass instances vs. $26$ Fail instances, approx. $2:1$), evaluating accuracy alone is insufficient. We report:
1. **Classification Accuracy:** $\frac{TP + TN}{TP + TN + FP + FN}$
2. **Precision (Positive Class):** $\frac{TP}{TP + FP}$ (Quantifies the fraction of predicted passes that actually passed).
3. **Recall (Positive Class):** $\frac{TP}{TP + FN}$ (Quantifies the model's ability to identify all passing students).
4. **F1-Score:** $\frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$ (Harmonic mean of precision and recall).
5. **Confusion Matrix:** Full layout of True Negatives, False Positives, False Negatives, and True Positives.
6. **Feature Distortion Metrics:**
   - Mean Absolute Error: $\text{MAE} = \frac{1}{N} \sum_{i=1}^N |x_i - x_{i, priv}|$
   - Root Mean Squared Error: $\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N (x_i - x_{i, priv})^2}$

---

## 14. Actual Preliminary Results

### 14.1 Target Distribution and Class Balance
Out of 395 total students, 265 ($67.1\%$) achieved a passing score ($G3 \ge 10$), while 130 ($32.9\%$) failed. This distribution is recorded in `figures/class_distribution.png`.

### 14.2 Experiment A: Baseline Performance
- **Baseline Decision Tree:**
  - Accuracy: **0.6582** ($65.82\%$)
  - Precision: **0.6857** ($68.57\%$)
  - Recall: **0.9057** ($90.57\%$)
  - F1-Score: **0.7805** ($78.05\%$)
  - Macro F1: **0.5045**
  - Confusion Matrix: $TN = 4, FP = 22, FN = 5, TP = 48$.
- **Baseline Logistic Regression:**
  - Accuracy: **0.6582** ($65.82\%$)
  - Precision: **0.7167** ($71.67\%$)
  - Recall: **0.8113** ($81.13\%$)
  - F1-Score: **0.7611** ($76.11\%$)
  - Macro F1: **0.5805**
  - Confusion Matrix: $TN = 9, FP = 17, FN = 10, TP = 43$.

*Observation:* Pruning the Decision Tree prioritizes high recall on the passing majority while identifying failing students through past failures and absences.

### 14.3 Experiment B: Data Anonymization & Re-identification Audit
- **Raw Dataset Risk Profile:**
  - Selected Quasi-Identifiers: `['school', 'sex', 'age', 'address', 'Medu', 'Fedu']`
  - Total Equivalence Classes: **192**
  - Uniquely Identifiable Students ($k=1$): **109** (**27.59%** of the entire cohort!)
  - Minimum class size: **1**; Median class size: **1.0**.
- **Anonymized Dataset Risk Profile:**
  - Generalizations Applied: `school` suppressed; `age` partitioned into `['15-16', '17-18', '19+']`; `Medu` and `Fedu` partitioned into `['Basic', 'Higher']`; `absences` categorized into `['Low', 'Moderate', 'High']`.
  - Total Equivalence Classes: **42**
  - Uniquely Identifiable Students ($k=1$): **5** (**1.27%** of the cohort).
  - Uniqueness Reduction: **95.4% reduction in uniquely identifiable students**.
  - Median class size: **5.0**.
- **Downstream Anonymized Decision Tree Utility:**
  - Accuracy: **0.6835** ($68.35\%$)
  - Precision: **0.7333** ($73.33\%$)
  - Recall: **0.8302** ($83.02\%$)
  - F1-Score: **0.7788** ($77.88\%$)
  - Macro F1: **0.6116**
  - Confusion Matrix: $TN = 10, FP = 16, FN = 9, TP = 44$.

*Key Finding:* Generalizing noisy continuous attributes regularized the tree splits, improving accuracy from $65.82\%$ to $68.35\%$ while reducing re-identification risk from $27.6\%$ to $1.3\%$.

### 14.4 Experiment C: Local Differential Privacy Distortion Analysis
Applying the client-side Laplace mechanism to `studytime` (domain $[1, 4]$, sensitivity $\Delta f = 3$) yielded the following empirical distortion and downstream performance:

| Privacy Budget ($\epsilon$) | Privacy Level | MAE Distortion | RMSE Distortion | Downstream Accuracy | Downstream Precision | Downstream Recall | Downstream F1 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.1** | Strong | 1.4770 | 1.7589 | 0.6582 | 0.6857 | 0.9057 | 0.7805 |
| **0.5** | Strong | 1.2905 | 1.5750 | 0.6582 | 0.6857 | 0.9057 | 0.7805 |
| **1.0** | Moderate | 1.1030 | 1.3880 | 0.6582 | 0.6857 | 0.9057 | 0.7805 |
| **2.0** | Moderate | 0.8305 | 1.0890 | 0.6582 | 0.6857 | 0.9057 | 0.7805 |
| **5.0** | Weak | 0.4520 | 0.6295 | 0.6582 | 0.6857 | 0.9057 | 0.7805 |
| **10.0** | Weak | 0.2403 | 0.3445 | 0.6582 | 0.6857 | 0.9057 | 0.7805 |

### 14.5 Experiment D: Consolidated Privacy–Utility Comparison Matrix

| Method | Privacy Mechanism / Transformation | Accuracy | Precision | Recall | F1-Score | Feature Distortion | Core Limitations |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- | :--- |
| **Baseline Decision Tree** | None (Raw identifiable features) | 0.6582 | 0.6857 | 0.9057 | 0.7805 | None (0.000) | High re-identification risk (27.6% unique QI signatures); susceptible to membership inference. |
| **Baseline Logistic Regression** | None (Raw features, linear model) | 0.6582 | 0.7167 | 0.8113 | 0.7611 | None (0.000) | No privacy safeguards; weights reveal linear sensitivity to input demographics. |
| **Anonymized Decision Tree** | Generalization (age, Medu, Fedu, absences) & suppression (school) | 0.6835 | 0.7333 | 0.8302 | 0.7788 | Discretization information loss | Reduces unique signatures from 27.6% to 1.3%, but vulnerable to homogeneity/background knowledge attacks; no formal DP guarantee. |
| **LDP Decision Tree ($\epsilon=0.5$)** | Local Laplace on `studytime` ($\lambda=6.0$, clipped $[1,4]$) | 0.6582 | 0.6857 | 0.9057 | 0.7805 | MAE: 1.2905 (RMSE: 1.5750) | Heavy perturbation on studytime; feature-level guarantee only; protects single attribute against untrusted central server. |
| **LDP Decision Tree ($\epsilon=1.0$)** | Local Laplace on `studytime` ($\lambda=3.0$, clipped $[1,4]$) | 0.6582 | 0.6857 | 0.9057 | 0.7805 | MAE: 1.1030 (RMSE: 1.3880) | Moderate privacy; balanced reconstruction error; protects single attribute against untrusted central server. |
| **LDP Decision Tree ($\epsilon=5.0$)** | Local Laplace on `studytime` ($\lambda=0.6$, clipped $[1,4]$) | 0.6582 | 0.6857 | 0.9057 | 0.7805 | MAE: 0.4520 (RMSE: 0.6295) | Weak privacy protection (high $\epsilon$ allows probabilistic attribute reconstruction). |

---

## 15. Observations and Limitations
1. **Decision Tree Stability under Single-Feature LDP:** Decision tree splits on small depths ($d=3$) heavily favor dominant global predictors (`failures`, `absences`, `schoolsup`). Perturbing `studytime` locally shields student study habits without compromising top-level tree splits.
2. **Re-identification vs. Differential Privacy:** Anonymization significantly curbed uniqueness ($27.6\% \rightarrow 1.3\%$), but it remains heuristic. An attacker with auxiliary knowledge of a student's family structure could still execute linkage. Only Differential Privacy provides mathematical worst-case bounds independent of external knowledge.
3. **Target Leakage Insight:** Excluding $G1$ and $G2$ produced a realistic baseline accuracy of $65.8\%$. Naive implementations claiming $>90\%$ accuracy suffer from temporal data leakage that invalidates early institutional intervention.

---

## 16. Work Remaining for IA 2
For Internal Assessment 2 (IA 2) and final project development, the following milestones are planned:
1. **End-to-End Central Differential Privacy:** Implement Differentially Private Stochastic Gradient Descent (DP-SGD) or Differentially Private Random Forests using Opacus or Diffprivlib.
2. **Compressive Privacy Representation Learning:** Implement an adversarial representation learning network or Mutual Information Neural Estimator (MINE) to rigorously compress sensitive attributes (Module 5).
3. **Multi-Attribute LDP:** Extend local perturbation across multiple continuous and categorical features simultaneously using Randomized Response and multidimensional noise mechanisms.
4. **Interactive Dashboard:** Build a Colab-compatible visualization suite allowing educators to dynamically adjust the privacy budget $\epsilon$ and inspect simulated student risk reports.

---

## 17. References
1. Cortez, P., & Silva, A. (2008). *Using data mining to predict secondary school student performance*. In Proceedings of 5th Annual Future Business Technology Conference (FUBUTEC 2008), pp. 5–12, Porto, Portugal, EUROSIS.
2. Dwork, C., & Roth, A. (2014). *The algorithmic foundations of differential privacy*. Foundations and Trends in Theoretical Computer Science, 9(3–4), 211–407.
3. Sweeney, L. (2002). *$k$-anonymity: A model for protecting privacy*. International Journal of Uncertainty, Fuzziness and Knowledge-Based Systems, 10(05), 557–570.
4. Machanavajjhala, A., Kifer, D., Gehrke, J., & Venkitasubramaniam, M. (2007). *$\ell$-diversity: Privacy beyond $k$-anonymity*. ACM Transactions on Knowledge Discovery from Data (TKDD), 1(1), 3-es.
5. Duchi, J. C., Jordan, M. I., & Wainwright, M. J. (2013). *Local privacy and statistical minimax rates*. In 2013 IEEE 54th Annual Symposium on Foundations of Computer Science (FOCS), pp. 429–438.
6. Abadi, M., Chu, A., Goodfellow, I., McMahan, H. B., Mironov, I., Talwar, K., & Zhang, L. (2016). *Deep learning with differential privacy*. In Proceedings of the 2016 ACM SIGSAC Conference on Computer and Communications Security (CCS), pp. 308–318.
