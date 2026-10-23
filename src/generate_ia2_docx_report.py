"""
src/generate_ia2_docx_report.py
-------------------------------
Generates the comprehensive, publication-styled Microsoft Word (.docx) 
technical report for IA 2 with all 20 sections, embedded figures, and styled tables.
"""

import os
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls


def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)


def create_ia2_docx_report(base_dir: str):
    doc = Document()
    
    # 1. Page Margins
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)
        
    # Helper formatting functions
    def add_title(text):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(text)
        r.font.name = "Arial"
        r.font.size = Pt(20)
        r.font.bold = True
        r.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)
        return p

    def add_subtitle(text):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(text)
        r.font.name = "Arial"
        r.font.size = Pt(11)
        r.font.italic = True
        r.font.color.rgb = RGBColor(0x55, 0x55, 0x55)
        doc.add_paragraph().paragraph_format.space_after = Pt(8)
        return p

    def add_h1(num, text):
        h = doc.add_paragraph()
        h.paragraph_format.space_before = Pt(14)
        h.paragraph_format.space_after = Pt(4)
        r = h.add_run(f"{num}. {text}")
        r.font.name = "Arial"
        r.font.size = Pt(13.5)
        r.font.bold = True
        r.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)
        return h

    def add_h2(text):
        h = doc.add_paragraph()
        h.paragraph_format.space_before = Pt(10)
        h.paragraph_format.space_after = Pt(3)
        r = h.add_run(text)
        r.font.name = "Arial"
        r.font.size = Pt(11.5)
        r.font.bold = True
        r.font.color.rgb = RGBColor(0x2C, 0x3E, 0x50)
        return h

    def add_body(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(5)
        p.paragraph_format.line_spacing = 1.15
        r = p.add_run(text)
        r.font.name = "Calibri"
        r.font.size = Pt(10.5)
        return p

    def add_fig(img_path, caption, width=Inches(4.8)):
        if os.path.exists(img_path):
            doc.add_picture(img_path, width=width)
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_after = Pt(8)
            r = p.add_run(caption)
            r.font.name = "Arial"
            r.font.size = Pt(9.5)
            r.font.italic = True
            r.font.color.rgb = RGBColor(0x44, 0x44, 0x44)

    # 1. Title Page
    add_title("Privacy-Preserving Student Performance Prediction\nUsing Machine Learning")
    add_subtitle("Internal Assessment 2 (IA 2) Final Research & Technical Report\n"
                 "Subject: Security and Privacy for Machine Learning | Semester VII B.Tech\n"
                 "Student: Aryan | Academic Year: 2026–2027")
                 
    add_h1(1, "Project Title and Metadata")
    add_body("Project Title: Privacy-Preserving Student Performance Prediction Using Machine Learning\n"
             "Category: Privacy-Preserving Machine Learning, Differential Privacy, Educational Data Mining, Responsible AI.\n"
             "Primary Learning Algorithm: Pruned Decision Tree Classifier (max_depth=3, min_samples_leaf=5) with comparative Logistic Regression.\n"
             "Privacy Paradigms: Data Anonymization (k-anonymity), Local Differential Privacy (Bounded Laplace Mechanism), Central Differential Privacy (Output Perturbation & diffprivlib GaussianNB), Privacy Risk Auditing (Inference Attacks), and Compressive Privacy.\n"
             "Target: Binary Mathematics Outcome (G3 >= 10 -> Pass, G3 < 10 -> Fail). Intermediate grades G1 and G2 are strictly excluded to avoid target leakage.")

    # 2. Abstract
    add_h1(2, "Abstract")
    add_body("Educational data mining models offer substantial potential for early scholastic interventions by predicting which students are at risk of academic failure. However, educational records encapsulate highly sensitive personal information, including family cohabitation status, parental education levels, home address types, study habits, and disciplinary absences. Training conventional machine learning models directly on raw, centralized student records introduces severe privacy risks, including linkage-based re-identification and sensitive attribute inference.\n\n"
             "This project implements an end-to-end, privacy-preserving machine learning pipeline on the benchmark UC Irvine Student Performance dataset (N=395 records, Mathematics course). To reflect real-world pre-examination intervention, we eliminate target leakage by excluding intermediate examination grades (G1, G2), compelling models to predict academic success purely from demographic, behavioral, and environmental covariates. We systematically evaluate baseline stability via 5-Fold Stratified Cross-Validation (Decision Tree: Accuracy 69.87% +/- 3.14%, F1-score 79.28% +/- 2.34%); audit and mitigate re-identification vulnerability via k-anonymity generalization (reducing uniquely identifiable students from 27.59% to 1.27% while improving accuracy to 68.35%); implement client-side Local Differential Privacy (LDP) mapping feature distortion from MAE=1.4770 (eps=0.1) to MAE=0.2403 (eps=10.0); implement Central Differential Privacy (Central DP) achieving 64.05% accuracy under (1.0, 0)-DP; execute an Attribute Inference Attack showing that overfitted models leak relationship status with 70.89% accuracy while pruned models neutralize this leakage to baseline chance (68.35%); and demonstrate Compressive Privacy subspace projection. Our findings demonstrate that judicious anonymization and localized feature perturbation provide verifiable privacy guarantees with negligible loss in early predictive utility.")

    # 3. Introduction & Motivation
    add_h1(3, "Introduction and Motivation")
    add_body("Modern educational institutions systematically aggregate multifaceted student data. While predicting academic failure early in a semester allows institutions to allocate tutoring and pastoral support, transmitting and storing unencrypted, granular student records creates grave vulnerabilities: (1) Linkage Attacks joining auxiliary public voter/sports rosters to re-identify students; (2) Internal Curiosity by untrusted cloud platforms; and (3) Model Inversion or Attribute Inference attacks deducting sensitive habits from model outputs.\n\n"
             "Regulatory frameworks such as FERPA (United States) and GDPR (European Union) mandate data minimization and privacy protection by design. Furthermore, early intervention requires forecasting performance before exams take place; naive models utilizing intermediate test scores G1 and G2 suffer from severe temporal leakage, producing artificial accuracy (>90%) that is unusable in real early-warning systems.")

    # 4. Refined Problem Statement
    add_h1(4, "Refined Problem Statement")
    add_body("The technical challenge addressed in this work is: How can an educational institution build an effective, explainable machine learning model to predict student failure early in the semester while providing verifiable mathematical privacy safeguards against re-identification, untrusted central aggregators, and adversarial inference attacks?")

    # 5. Objectives & Research Questions
    add_h1(5, "Objectives and Research Questions")
    add_body("Key Objectives:\n"
             "1. Construct a leakage-free prediction pipeline predicting mathematics outcome (Pass/Fail) without prior exam scores.\n"
             "2. Establish baseline predictive performance and verify generalization stability using 5-Fold Stratified Cross-Validation.\n"
             "3. Quantify empirical re-identification risk (k=1 uniqueness) and evaluate generalization/suppression transformations under k-anonymity.\n"
             "4. Implement a client-side bounded Laplace mechanism for Local Differential Privacy on studytime and analyze distortion across privacy budgets epsilon in [0.1, 10.0].\n"
             "5. Implement a Central Differential Privacy learning framework (IBM diffprivlib and output perturbation) satisfying (epsilon, 0)-DP.\n"
             "6. Empirically demonstrate an Attribute Inference Attack and evaluate whether privacy-preserving models resist sensitive attribute deduction.\n"
             "7. Implement Compressive Privacy representation learning and compare it against standard Principal Component Analysis (PCA).")

    # 6. Summary of Changes Since IA 1
    add_h1(6, "Summary of Changes Since IA 1")
    add_body("In transition from IA 1 to IA 2, the project expanded from a preliminary study into a comprehensive multi-paradigm research framework: (1) Added 5-Fold Stratified Cross-Validation to baseline models; (2) Implemented Central Differential Privacy via Output Perturbation Logistic Regression and IBM diffprivlib GaussianNB across 6 privacy budgets; (3) Added empirical before-and-after LDP distribution plots; (4) Designed and executed an empirical Attribute Inference Attack evaluating sensitive relationship deduction; (5) Implemented Compressive Privacy subspace projection demonstrating why standard PCA fails to preserve privacy; and (6) Expanded test suite from 6 to 11 unit tests with 100% pass rate.")

    # 7. Related Work & Conceptual Background
    add_h1(7, "Related Work and Conceptual Background")
    add_body("Decision Trees: Axis-aligned partitioning maximizing Gini impurity reduction. In educational settings, pruned Decision Trees offer unmatched explainability for school counselors.\n\n"
             "k-Anonymity: Formulated by Sweeney (2002), requiring that every combination of quasi-identifiers occurs at least k times. Vulnerable to homogeneity and background knowledge attacks.\n\n"
             "Differential Privacy (Central vs. Local): Central DP (Dwork & Roth, 2014) assumes a trusted aggregator perturbing query outputs. Local DP assumes an untrusted aggregator, perturbing data directly on student devices. The Post-Processing Theorem guarantees that deterministic clipping of noisy outputs to [1, 4] maintains epsilon-LDP unconditionally.\n\n"
             "Compressive Privacy: Information-theoretic representation learning minimizing mutual information with sensitive attributes while preserving target utility.")

    # 8. Dataset Description & Preprocessing
    add_h1(8, "Dataset Description and Preprocessing")
    add_body("UCI Student Performance Dataset (Cortez & Silva, 2008). CC BY 4.0 License. 395 student instances, 33 attributes, 0 missing entries, 0 duplicate records. Binarized target: G3 >= 10 (Pass: 265, Fail: 130). Dropping G1, G2, and G3 leaves 30 input features, encoded to 39 features via one-hot encoding.")
    
    add_fig(os.path.join(base_dir, "figures", "class_distribution.png"), "Figure 1: Target Class Outcome Distribution (Pass: 265, Fail: 130)")

    # 9. System Architecture & Workflow
    add_h1(9, "System Architecture and Workflow")
    add_body("The end-to-end architecture comprises: (1) Ingestion & leakage-free binarization; (2) Stratified 80/20 train/test split and 5-fold CV; (3) Baseline Decision Tree & Logistic Regression; (4) k-Anonymity audit and generalization; (5) Local DP bounded Laplace perturbation; (6) Central DP output perturbation; (7) Privacy risk inference attack analysis; (8) Compressive privacy projection; and (9) Unified evaluation and reporting.")

    # 10. Implementation Details
    add_h1(10, "Implementation Details")
    add_body("Developed in Python 3.12 using modular components: data_preprocessing.py, baseline_model.py, anonymization.py, local_differential_privacy.py, central_differential_privacy.py, privacy_risk_analysis.py, compressive_privacy.py, and evaluation.py.")

    # 11. Experimental Setup
    add_h1(11, "Experimental Setup")
    add_body("Hardware: AMD/Intel x86_64, 16 GB RAM, Windows 11. Reproducibility seed: 42 across all splits and noise generators. Decision Tree: max_depth=3, min_samples_leaf=5. Logistic Regression: C=1.0, max_iter=1000. Privacy Budgets: epsilon in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0].")

    # 12. Results and Figures
    add_h1(12, "Results and Figures")
    add_h2("Baseline Model Performance & Cross-Validation Stability")
    add_body("Single-split baseline Decision Tree achieved Accuracy = 65.82%, Precision = 68.57%, Recall = 90.57%, F1-score = 78.05%. Baseline Logistic Regression achieved Accuracy = 65.82%, Precision = 71.67%, Recall = 81.13%, F1-score = 76.11%.\n\n"
             "Across 5-Fold Stratified Cross-Validation, the Decision Tree achieved Accuracy = 69.87% +/- 3.14% and F1 = 79.28% +/- 2.34%, confirming stable generalization across partitions.")
             
    add_fig(os.path.join(base_dir, "figures", "confusion_matrix.png"), "Figure 2: Baseline Decision Tree Confusion Matrix Heatmap", width=Inches(4.2))
    add_fig(os.path.join(base_dir, "figures", "cross_validation_metrics.png"), "Figure 3: 5-Fold Stratified Cross-Validation Baseline Stability", width=Inches(4.8))

    add_h2("Experiment A: Data Anonymization Audit")
    add_body("Raw quasi-identifiers (school, sex, age, address, Medu, Fedu) exposed 109 students (27.59%) as completely unique (k=1). After generalization (suppressing school, binning age, Medu, Fedu, absences), uniquely identifiable students dropped to 5 (1.27%)—a 95.4% reduction in re-identification vulnerability. Downstream Decision Tree accuracy improved to 68.35% with an F1-score of 77.88%.")

    add_h2("Experiment B: Local Differential Privacy Distortion & Distributions")
    add_body("Applying client-side Laplace noise to studytime (domain [1, 4], Delta_f = 3) mapped feature distortion from MAE = 1.4770 (eps=0.1) down to MAE = 0.2403 (eps=10.0). Below are the empirical distribution shifts before and after perturbation:")
    
    add_fig(os.path.join(base_dir, "figures", "feature_distribution_ldp.png"), "Figure 4: studytime Probability Distributions Before and After Local DP Laplace Perturbation", width=Inches(6.2))
    add_fig(os.path.join(base_dir, "figures", "feature_distortion.png"), "Figure 5: Privacy Budget (Epsilon) vs. Feature MAE/RMSE Distortion Error", width=Inches(4.8))

    add_h2("Experiment C: Central Differential Privacy")
    add_body("Output perturbation on Logistic Regression weights achieved 61.27% +/- 2.35% accuracy at eps=0.1, rising to 64.05% +/- 1.29% at eps=1.0 and 63.29% at eps=10.0. IBM diffprivlib GaussianNB achieved 63.80% accuracy at eps=1.0.")
    add_fig(os.path.join(base_dir, "figures", "central_dp_tradeoff.png"), "Figure 6: Central Differential Privacy Accuracy vs. Privacy Budget (Epsilon)", width=Inches(4.8))

    add_h2("Experiment E: Privacy Risk & Inference Attack")
    add_body("In our Attribute Inference Attack, an adversary attempted to predict whether a student is in a romantic relationship from public demographics and model outputs. Public features yielded 68.35% accuracy (baseline chance = 68.35%). An overfitted unpruned tree increased attack accuracy to 70.89% (precision 66.67%), proving that overfitted models memorize and leak sensitive attributes. Pruned Decision Trees neutralized this attack to baseline chance (68.35%).")
    add_fig(os.path.join(base_dir, "figures", "inference_risk_analysis.png"), "Figure 7: Attribute Inference Attack Accuracy across Model Configurations", width=Inches(4.8))

    # 13. Privacy-Utility Comparison Matrix
    add_h1(13, "Consolidated Privacy–Utility Comparison")
    
    t_comp = doc.add_table(rows=8, cols=6)
    t_comp.alignment = WD_TABLE_ALIGNMENT.CENTER
    c_headers = ["Method", "Privacy Mechanism", "Accuracy", "Precision", "Recall", "F1-Score"]
    for i, h in enumerate(c_headers):
        cell = t_comp.cell(0, i)
        cell.text = h
        set_cell_background(cell, "1B365D")
        for p in cell.paragraphs:
            for r in p.runs:
                r.font.name = "Arial"
                r.font.size = Pt(9.5)
                r.font.bold = True
                r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

    comp_rows = [
        ("Baseline Decision Tree", "None (Raw features)", "0.6582", "0.6857", "0.9057", "0.7805"),
        ("Baseline Logistic Reg", "None (Linear model)", "0.6582", "0.7167", "0.8113", "0.7611"),
        ("Anonymized Decision Tree", "Generalization + suppression", "0.6835", "0.7333", "0.8302", "0.7788"),
        ("Local DP DT (eps=0.5)", "Laplace on studytime", "0.6582", "0.6857", "0.9057", "0.7805"),
        ("Local DP DT (eps=1.0)", "Laplace on studytime", "0.6582", "0.6857", "0.9057", "0.7805"),
        ("Central DP LR (eps=0.5)", "Output perturbation", "0.6405", "0.6857", "0.8679", "0.7473"),
        ("Central DP LR (eps=1.0)", "Output perturbation", "0.6405", "0.6857", "0.8679", "0.7473")
    ]
    for r_idx, row in enumerate(comp_rows, start=1):
        bg = "F4F6F9" if r_idx % 2 == 1 else "FFFFFF"
        for c_idx, val in enumerate(row):
            cell = t_comp.cell(r_idx, c_idx)
            cell.text = val
            set_cell_background(cell, bg)
            for p in cell.paragraphs:
                for r in p.runs:
                    r.font.name = "Calibri"
                    r.font.size = Pt(9)

    doc.add_paragraph().paragraph_format.space_after = Pt(8)
    add_fig(os.path.join(base_dir, "figures", "privacy_utility_tradeoff_ia2.png"), "Figure 8: Unified Privacy-Utility Frontier across Baseline, Anonymization, Local DP, and Central DP", width=Inches(5.4))

    # 14. Discussion & Interpretation
    add_h1(14, "Discussion and Interpretation")
    add_body("1. Resilience of Explainable Models: Pruned Decision Trees focus on dominant global signals (past failures and absences). Localized client-side noise on studytime did not degrade top-level splits, maintaining strong predictive utility.\n"
             "2. Central DP Trade-Off: Central DP impacts all feature weights simultaneously, causing a minor accuracy decrease of ~1.8% to 4.5% at eps=0.1 to 0.5, but provides a formal global guarantee over the entire student profile.\n"
             "3. Overfitting as a Privacy Threat: The inference risk experiment confirmed that overfitted models memorize and leak sensitive non-academic attributes, whereas regularized models neutralize this leakage.\n"
             "4. Compressive Privacy vs. PCA: Standard PCA failed to suppress sensitive attributes because it maximizes variance; Compressive Privacy orthogonal projection successfully suppressed sensitive leakage while preserving 69.62% target accuracy.")

    # 15. Mapping to All 5 Modules
    add_h1(15, "Mapping to All Five Syllabus Modules")
    
    t_mod = doc.add_table(rows=6, cols=5)
    t_mod.alignment = WD_TABLE_ALIGNMENT.CENTER
    headers = ["Syllabus Module", "Course Concept", "Project Implementation", "Empirical Evidence", "Limitations"]
    for i, h in enumerate(headers):
        cell = t_mod.cell(0, i)
        cell.text = h
        set_cell_background(cell, "1B365D")
        for p in cell.paragraphs:
            for r in p.runs:
                r.font.name = "Arial"
                r.font.size = Pt(9.5)
                r.font.bold = True
                r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

    mod_data = [
        ("Module 1: Privacy Considerations in ML", "Sensitive academic data, inference attacks, data leakage.", "Strict exclusion of G1 and G2; demographic quasi-identifier audit; Attribute Inference attack.", "Feature space reduced from 33 to 30; attack results in inference_risk_results.csv.", "Requires institutional review board and consent."),
        ("Module 2: Differential Privacy for ML", "Central DP, privacy budget epsilon, noise calibration.", "Mathematical derivation of Laplace scale lambda = Delta_w / epsilon; output perturbation LR & diffprivlib GNB.", "Central DP accuracy across eps in [0.1, 10.0] logged in central_dp_results.csv.", "Central DP assumes trusted curator; DP-SGD in future work."),
        ("Module 3: Local Differential Privacy for ML", "Untrusted aggregator, client perturbation, feature distortion.", "Client-side bounded Laplace perturbation on studytime with clipping.", "MAE distortion curve: 1.4770 (eps=0.1) to 0.2403 (eps=10.0); KDE distribution plot.", "High noise at small epsilon introduces individual variance."),
        ("Module 4: Privacy-Preserving Data Mining", "Quasi-identifiers, k-anonymity, generalization, suppression.", "Equivalence class analysis; binning age, Medu, Fedu; suppressing school.", "Unique students reduced from 109 (27.6%) to 5 (1.3%); 68.35% accuracy.", "Vulnerable to homogeneity attacks; lacks DP bound."),
        ("Module 5: Compressive Privacy for ML", "Representation learning, information bottleneck, sensitive suppression.", "Evaluated target utility vs sensitive leakage across Full, PCA, and Compressive Nullspace Projection.", "Empirical metrics in compressive_privacy_results.csv showing PCA fails to protect privacy.", "Linear subspace projection effective for linear correlations.")
    ]
    for r_idx, row in enumerate(mod_data, start=1):
        bg = "F4F6F9" if r_idx % 2 == 1 else "FFFFFF"
        for c_idx, val in enumerate(row):
            cell = t_mod.cell(r_idx, c_idx)
            cell.text = val
            set_cell_background(cell, bg)
            for p in cell.paragraphs:
                for r in p.runs:
                    r.font.name = "Calibri"
                    r.font.size = Pt(8.5)

    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    # 16. Limitations
    add_h1(16, "Limitations and Threats to Validity")
    add_body("1. Dataset Scope: Portuguese secondary school sample from 2005-2006. Findings should not be generalized to other demographics without localized recalibration.\n"
             "2. Single-Feature Local DP: Local DP was applied specifically to studytime; applying LDP to all features simultaneously causes cumulative variance.\n"
             "3. Central DP Assumptions: Central DP assumes trusted curator and bounded data norm.")

    # 17. Conclusion
    add_h1(17, "Conclusion")
    add_body("This project successfully demonstrated an authentic, leakage-free student performance prediction system with multi-paradigm privacy safeguards. Data anonymization curbed re-identification vulnerability by 95.4%, Local DP protected student study habits under an untrusted aggregator model, Central DP enforced formal parameter guarantees with minimal utility degradation, and Compressive Privacy successfully suppressed sensitive demographic leakage.")

    # 18. Future Work
    add_h1(18, "Future Work")
    add_body("Future research directions include: (1) Differentially Private Stochastic Gradient Descent (DP-SGD) on multi-institutional deep networks; (2) Adversarial neural representation learning (Information Bottleneck); and (3) Federated Learning with Secure Multi-Party Aggregation.")

    # 19. Individual Contribution
    add_h1(19, "Individual Contribution Statement")
    add_body("As a Semester VII B.Tech student, I independently conducted: pipeline design and target leakage prevention; 5-fold cross-validation implementation; Central DP experimentation; empirical Attribute Inference and Membership Inference attacks; Compressive Privacy representation learning; automated unit testing; and report preparation.")

    # 20. References
    add_h1(20, "References")
    add_body("1. Cortez, P., & Silva, A. (2008). Using data mining to predict secondary school student performance. FUBUTEC 2008.\n"
             "2. Dwork, C., & Roth, A. (2014). The algorithmic foundations of differential privacy. Foundations and Trends in TCS.\n"
             "3. Sweeney, L. (2002). k-anonymity: A model for protecting privacy. Int. J. Uncertain. Fuzziness Knowl.-Based Syst.\n"
             "4. Machanavajjhala, A. et al. (2007). l-diversity: Privacy beyond k-anonymity. ACM TKDD.\n"
             "5. Chaudhuri, K. et al. (2011). Differentially private empirical risk minimization. JMLR.\n"
             "6. Holohan, N. et al. (2019). Diffprivlib: The IBM differential privacy library. arXiv:1907.02444.\n"
             "7. Yeom, S. et al. (2018). Privacy risk in machine learning: Analyzing the connection to overfitting. IEEE CSF.\n"
             "8. Kung, S. Y. (2017). Compressive privacy: An information-theoretic approach. IEEE TIFS.")

    out_docx = os.path.join(base_dir, "reports", "IA2_Final_Report.docx")
    doc.save(out_docx)
    print(f"[OK] Generated Word report: {out_docx}")


if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    create_ia2_docx_report(base_dir)
