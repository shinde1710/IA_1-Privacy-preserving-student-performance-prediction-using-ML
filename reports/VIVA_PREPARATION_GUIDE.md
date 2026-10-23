# Viva Preparation Guide: Privacy-Preserving Student Performance Prediction
### Subject: Security and Privacy for Machine Learning | Semester VII B.Tech

This guide contains simple, clear, and technically precise answers to the 12 most important questions examiners ask during an IA 1 / project viva.

---

### Q1: What is the main objective of your project?
**Answer:**  
The objective is to build an explainable machine learning model that predicts whether a student will pass or fail mathematics early in the semester, while applying and comparing data privacy techniques (Data Anonymization and Local Differential Privacy) to prevent individual students from being re-identified or having their sensitive habits leaked.

---

### Q2: Why did you choose a Decision Tree Classifier as your primary model?
**Answer:**  
1. **Explainability and Transparency:** A Decision Tree is a "white-box" model. Each prediction can be traced through human-readable IF-THEN rules (e.g., `IF failures <= 0 AND absences <= 15 THEN Pass`). In educational settings, teachers and administrators must understand *why* a student is flagged for intervention.
2. **Tabular Data Suitability:** On 395 records, deep neural networks easily overfit and act as uninterpretable black boxes.
3. **Controlled Pruning:** We set `max_depth = 3` and `min_samples_leaf = 5`, which prevents overfitting on our 316 training records while making the entire tree easy to draw and explain on a whiteboard during viva.

---

### Q3: Why did you drop the earlier grades G1 and G2 from your training features?
**Answer:**  
In the UCI dataset, $G1$ (period 1 grade) and $G2$ (period 2 grade) have a correlation of over $0.85$ with the final grade $G3$.  
If we include $G1$ and $G2$, the model achieves an artificially high accuracy ($>90\%$), but this is **target leakage** (temporal data leakage). In a real school, if an educator waits for mid-term grades $G1$ and $G2$, half the semester has already passed and it is often too late for early remedial tutoring. Excluding $G1$ and $G2$ forces the model to predict performance from demographic, behavioral, and attendance data *before* major exams take place.

---

### Q4: What is the difference between Direct Identifiers and Quasi-Identifiers?
**Answer:**  
- **Direct Identifiers:** Attributes that uniquely identify an individual on their own without needing any other information (e.g., Student Name, Roll Number, Aadhaar/Social Security Number, Email).
- **Quasi-Identifiers (QIs):** Attributes that do not identify a person alone, but when combined together, form a unique fingerprint that can re-identify a person when linked to an external dataset (e.g., combining `[School, Sex, Age, Address, Mother's Education, Father's Education]`).

---

### Q5: What is $k$-anonymity and what was your empirical finding on the student dataset?
**Answer:**  
A dataset satisfies $k$-anonymity if each distinct combination of quasi-identifiers is shared by at least $k$ individuals in the released data.
- **Before Anonymization:** Across 6 quasi-identifiers (`school, sex, age, address, Medu, Fedu`), there were 192 equivalence classes, and **109 students (27.59%) had $k=1$**. This means more than a quarter of the entire student body was completely unique and vulnerable to linkage re-identification.
- **After Anonymization:** We suppressed `school`, generalized `age` into 3 bins (`15-16`, `17-18`, `19+`), grouped parental education into `Basic` vs `Higher`, and categorized `absences`. This reduced the number of uniquely identifiable students from 109 down to **5 students (1.27%)**—a **95.4% reduction in re-identification risk**—while model accuracy actually improved from $65.82\%$ to $68.35\%$.

---

### Q6: What are the two major weaknesses of $k$-anonymity?
**Answer:**  
1. **Homogeneity Attack:** If all $k$ individuals in an equivalence class have the same target value (for example, all 10 students in a group failed), an attacker immediately discovers that any student in that group failed, even though the attacker cannot tell which specific record belongs to that student.
2. **Background Knowledge Attack:** If an attacker has auxiliary facts (e.g., knowing that a specific student never misses school or has high study hours), they can eliminate candidates within an equivalence class and single out the victim.  
$k$-Anonymity does not provide a formal mathematical guarantee against arbitrary external knowledge.

---

### Q7: What is Differential Privacy (DP), and what does the privacy parameter $\epsilon$ (epsilon) represent?
**Answer:**  
Differential Privacy provides a mathematically provable privacy guarantee. It ensures that an algorithm's output distribution remains virtually unchanged whether any single individual's record is present or absent in the dataset.
- **Privacy Budget $\epsilon$ (Epsilon):** Measures the privacy loss.
  - **Small $\epsilon$ (e.g., $0.1, 0.5$):** Heavy noise, very strong privacy protection, higher feature distortion, lower data fidelity.
  - **Large $\epsilon$ (e.g., $5.0, 10.0$):** Light noise, weak privacy protection, low distortion, higher data fidelity.

---

### Q8: What is the key difference between Central Differential Privacy and Local Differential Privacy (LDP)?
**Answer:**  
- **Central DP:** Students send raw, unprotected data to a central school server. The central server is trusted to add noise before releasing query answers or training models.
- **Local DP (LDP):** The central server is **untrusted**. Each student's device (laptop or smartphone) adds calibrated noise locally *before* transmitting the data. The server only ever receives noisy values, protecting the student even if the server is breached or corrupted.

---

### Q9: How did you implement Local DP on the `studytime` attribute, and why is clipping valid?
**Answer:**  
1. **Range & Sensitivity:** In the UCI dataset, weekly study time is bounded in the domain $[1, 4]$. The global $L_1$ sensitivity is $\Delta f = 4 - 1 = 3$.
2. **Laplace Perturbation:** Each student's device samples noise $Z_i \sim \text{Laplace}(0, 3 / \epsilon)$ and computes:
   $$x_{priv} = \text{clip}(x_i + Z_i, 1.0, 4.0)$$
3. **Validity of Clipping:** Under the **Post-Processing Invariance Property of Differential Privacy** (Dwork & Roth, 2014), performing any post-processing operation (like clipping to the known domain $[1, 4]$) on an already differentially private result strictly maintains the $\epsilon$-LDP guarantee without leaking extra information.

---

### Q10: How did feature distortion change across different values of $\epsilon$?
**Answer:**  
Our empirical measurements showed:
- At $\epsilon = 0.1$ (Strong privacy): $\text{MAE} = 1.4770$ (Mean Absolute Error) and $\text{RMSE} = 1.7589$.
- At $\epsilon = 1.0$ (Moderate privacy): $\text{MAE} = 1.1030$ and $\text{RMSE} = 1.3880$.
- At $\epsilon = 10.0$ (Weak privacy): $\text{MAE} = 0.2403$ and $\text{RMSE} = 0.3445$.  
This clearly demonstrates that increasing privacy (smaller $\epsilon$) exponentially increases reconstruction error, reflecting the fundamental privacy-utility trade-off.

---

### Q11: What is Compressive Privacy (Module 5) and why is PCA alone NOT sufficient for privacy?
**Answer:**  
- **Compressive Privacy:** The objective of learning a compressed representation (feature encoding) that retains information predictive of the target (passing or failing) while actively discarding or compressing sensitive demographic information (such as parental cohabitation or gender).
- **Why PCA alone is insufficient:** Standard Principal Component Analysis (PCA) merely projects data onto directions of maximum variance. If a sensitive attribute (e.g., family size or age) has high variance, PCA will preserve it rather than suppress it. PCA offers no worst-case mathematical bounds on reconstruction and does not guarantee differential privacy.

---

### Q12: How do your results map to the five syllabus modules?
**Answer:**  
- **Module 1 (Privacy Considerations in ML):** Identifying sensitive educational attributes, preventing target leakage by removing $G1$ and $G2$.
- **Module 2 (Differential Privacy for ML):** Mathematical derivation of the Laplace mechanism scale $\lambda = \Delta f / \epsilon$ and privacy budget $\epsilon$.
- **Module 3 (Local Differential Privacy for ML):** Client-side perturbation of `studytime` under an untrusted aggregator model and empirical distortion charting.
- **Module 4 (Privacy-Preserving Data Mining):** Evaluating quasi-identifier equivalence classes, reducing $k=1$ uniqueness from $27.59\%$ to $1.27\%$ via generalization and suppression.
- **Module 5 (Compressive Privacy for ML):** Analyzing feature representation compression where discretized features maintained an F1-score of $77.88\%$ while suppressing individual granular details.
