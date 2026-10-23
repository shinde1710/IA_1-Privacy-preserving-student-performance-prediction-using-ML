# IA 2 Viva Preparation Guide: Privacy-Preserving Student Performance Prediction
### Subject: Security and Privacy for Machine Learning | Semester VII B.Tech

This guide contains simple, clear, and technically precise answers to the 10 most crucial viva questions examiners ask for IA 2.

---

### Q1: What was the main motivation for extending your project from IA 1 to IA 2?
**Answer:**  
In IA 1, we established an explainable baseline (Decision Tree) and preliminary Local Differential Privacy and $k$-anonymity experiments. In IA 2, we addressed five key gaps:
1. **Model Stability:** Added 5-Fold Stratified Cross-Validation to evaluate variance on small datasets.
2. **Central Differential Privacy:** Implemented formal $(\epsilon, 0)$-Central DP using output perturbation and IBM `diffprivlib`.
3. **Privacy Risk Analysis:** Designed and executed an empirical Attribute Inference Attack to test whether academic predictions leak unreleased lifestyle attributes (relationship status).
4. **Compressive Privacy:** Implemented a representation learning demonstration proving that standard PCA fails to preserve privacy, whereas orthogonal subspace projection suppresses sensitive demographic leakage.
5. **Distribution Auditing:** Analyzed probability density shifts before and after Local DP Laplace perturbation.

---

### Q2: What is the operational difference between Central Differential Privacy and Local Differential Privacy (LDP) in your project?
**Answer:**  
- **Threat Model Difference:**
  - **In Central DP (Experiment C):** The university/curator is **trusted** to hold raw student records. Calibrated noise is added to the model's parameters (weights) during training or release. It provides a formal global guarantee over the entire training set simultaneously.
  - **In Local DP (Experiment B):** The central aggregator is **untrusted**. Each student's personal laptop or phone adds calibrated Laplace noise to their own `studytime` *before* transmitting it over the network. The server never observes raw individual study hours.
- **Utility Impact:** In Local DP, noise is applied to individual values, causing feature distortion ($\text{MAE}=1.10$ at $\epsilon=1.0$), but the Decision Tree's top splits remained stable ($65.82\%$ accuracy). In Central DP, weight perturbation slightly shifted all decision boundaries, yielding an accuracy of $64.05\% \pm 1.29\%$ at $\epsilon=1.0$.

---

### Q3: How did you implement Central Differential Privacy and what is the underlying mathematical theorem?
**Answer:**  
We implemented **Output Perturbation on Regularized Empirical Risk Minimization** (Chaudhuri et al., JMLR 2011; Dwork & Roth, 2014):
1. **Sensitivity Derivation:** For $L_2$-regularized Logistic Regression with inverse regularization parameter $C=1.0$ and sample size $N=316$, the $L_1$ sensitivity of the trained weight vector $w^*$ is strictly bounded by:
   $$\Delta w = \frac{2 \cdot C}{N} = \frac{2.0 \times 1.0}{316} \approx 0.00633$$
2. **Noise Mechanism:** Using `diffprivlib.mechanisms.Laplace`, we sample independent noise from $\text{Laplace}\left(0, \frac{\Delta w}{\epsilon}\right)$ and add it directly to the model weights:
   $$w_{priv} = w^* + \text{Laplace}\left(0, \frac{\Delta w}{\epsilon}\right)$$
3. **Guarantee:** By the output perturbation theorem, the resulting classifier satisfies formal $(\epsilon, 0)$-Central Differential Privacy.

---

### Q4: How did you evaluate Attribute Inference Risk (Experiment E) and what did you discover?
**Answer:**  
- **Threat Model:** An adversary observes public student demographics (`school, sex, age, address, studytime`) and the model's predicted academic probability $\hat{p} = \Pr(G3 \ge 10)$. The adversary trains an attack model to predict an unreleased private attribute: student romantic relationship status (`romantic`).
- **Empirical Findings:**
  - The majority-class baseline chance was **68.35%**.
  - Using public features alone gave **68.35%** (F1 = 0.1379).
  - An **overfitted, unpruned Decision Tree** increased attack accuracy to **70.89%** and attack precision jumped to **66.67%** (F1 = 0.2581). This proves that overfitted models memorize and leak sensitive attribute correlations through prediction confidences!
  - Our **pruned Decision Tree ($d=3$)** neutralized the attack back to baseline chance (**68.35%**), demonstrating that tree pruning acts as an inherent privacy safeguard.

---

### Q5: What is Compressive Privacy (Module 5) and why does standard PCA fail as a privacy mechanism?
**Answer:**  
- **Compressive Privacy Goal:** Learn a feature representation that retains predictive information for the primary task ($G3 \ge 10$) while compressing and removing information correlating with a sensitive demographic attribute ($S = \text{sex}$).
- **Why Standard PCA Fails:** PCA merely finds orthogonal axes that maximize dataset variance. In demographic and lifestyle data, sensitive attributes often account for significant variance. Consequently, standard PCA ($k=10$) preserved sensitive attribute correlation (leakage accuracy was **64.56%** vs. 59.49% baseline chance) while reducing target accuracy to **62.03%**.
- **Our Compressive Privacy Solution:** We projected the feature space onto the nullspace orthogonal to the sensitive attribute's correlation vector:
  $$P = I - v v^T$$
  This preserved target accuracy at **69.62%** (F1 = 78.95%) while suppressing the sensitive attribute's direct predictive weight.

---

### Q6: What did 5-Fold Stratified Cross-Validation reveal about baseline model stability?
**Answer:**  
On small datasets ($N=395$), a single train/test split can be misleading due to sampling variability. Evaluating our pruned Decision Tree across 5 stratified folds demonstrated remarkable stability:
- **Accuracy:** $69.87\% \pm 3.14\%$
- **Precision:** $73.62\% \pm 2.41\%$
- **Recall:** $86.04\% \pm 4.40\%$
- **F1-Score:** $79.28\% \pm 2.34\%$
This proves that the model consistently captures authentic behavioral predictors of academic performance with very low fold-to-fold variance ($\sim 3\%$).

---

### Q7: Why did data anonymization (generalization) actually improve downstream accuracy from 65.82% to 68.35%?
**Answer:**  
In our dataset, fine-grained continuous attributes like exact age ($15, 16, 17, 18\dots$) and exact absence counts ($0, 1, 2, 3\dots 93$) contain sample noise. When we generalized `age` into coarse cohorts (`15-16`, `17-18`, `19+`) and `absences` into categories (`Low`, `Moderate`, `High`), the discretization acted as a form of **feature regularization**. It eliminated idiosyncratic noise in the training set, allowing the Decision Tree to make more robust splits on test data.

---

### Q8: What was your finding on Membership Inference Attack (MIA) risk?
**Answer:**  
We implemented a loss-threshold membership inference attack (Yeom et al., 2018), where an adversary guesses whether a student was in the training set based on whether their prediction confidence exceeds the median. On our pruned Decision Tree, attack accuracy was **45.18%**, which is below random guessing ($50\%$). Because the tree was pruned to depth 3, it avoided memorizing individual training instances, naturally suppressing membership leakage.

---

### Q9: Summarize the Privacy–Utility Frontier: which method caused the largest utility loss and which provided the best balance?
**Answer:**  
- **Largest Utility Loss:** Central DP at very strong privacy ($\epsilon=0.1$) caused the largest drop in accuracy ($61.27\%$, down by $4.55\%$) because global weight perturbation shifts decision boundaries for all features.
- **Best Balance:** 
  - **Data Anonymization:** Provided an outstanding practical balance, slashing $k=1$ uniqueness from $27.59\%$ to $1.27\%$ while improving accuracy to $68.35\%$.
  - **Local DP at $\epsilon=1.0$:** Maintained full baseline accuracy ($65.82\%$) while providing mathematically provable local privacy on weekly study hours with an acceptable distortion error ($\text{MAE}=1.10$).

---

### Q10: How does your IA 2 implementation map to all five syllabus modules?
**Answer:**  
1. **Module 1 (Privacy Considerations in ML):** Target leakage prevention (dropping $G1, G2$), FERPA context, and empirical Attribute Inference Attack demonstration.
2. **Module 2 (Differential Privacy for ML):** Output perturbation Central DP on Logistic Regression weights and IBM `diffprivlib` GaussianNB across $\epsilon \in [0.1, 10.0]$.
3. **Module 3 (Local Differential Privacy for ML):** Client-side bounded Laplace perturbation on `studytime` with distribution and distortion plots.
4. **Module 4 (Privacy-Preserving Data Mining):** Quasi-identifier audit reducing uniqueness from $27.59\%$ to $1.27\%$ via generalization and suppression.
5. **Module 5 (Compressive Privacy for ML):** Empirical demonstration of orthogonal nullspace projection retaining $69.62\%$ accuracy while proving why standard PCA fails to preserve privacy.
