# Dataset Documentation: Student Performance Dataset

## 1. Provenance and Citation
- **Dataset Title:** Student Performance
- **Repository:** UCI Machine Learning Repository
- **URL:** https://archive.ics.uci.edu/dataset/320/student+performance
- **Direct Download URL:** https://archive.ics.uci.edu/static/public/320/student+performance.zip
- **Original Source Paper:**
  Cortez, P., & Silva, A. (2008). *Using data mining to predict secondary school student performance*. In Proceedings of 5th Annual Future Business Technology Conference (FUBUTEC 2008), pp. 5-12, Porto, Portugal, EUROSIS, ISBN 978-9077381-39-7.
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0).

## 2. Dataset Overview
This dataset contains secondary school student performance from two Portuguese schools (*Gabriel Pereira* and *Mouzinho da Silveira*). The data was collected using school reports and questionnaires.

We focus on the **Mathematics** course dataset: `student-mat.csv` (395 records, 33 attributes).

### Note on Scope and Representativeness
This dataset represents a specific sample of secondary school students in Portugal in 2005-2006. It is utilized strictly for academic demonstration and privacy-preserving ML methodology research. Findings should not be generalized across disparate socioeconomic, cultural, or institutional contexts without localized calibration.

## 3. Attribute Dictionary

| Attribute | Type | Description / Domain |
| :--- | :--- | :--- |
| `school` | Binary | Student's school ('GP' - Gabriel Pereira or 'MS' - Mouzinho da Silveira) |
| `sex` | Binary | Student's sex ('F' - female or 'M' - male) |
| `age` | Numeric | Student's age (from 15 to 22) |
| `address` | Binary | Student's home address type ('U' - urban or 'R' - rural) |
| `famsize` | Binary | Family size ('LE3' - <= 3 or 'GT3' - > 3) |
| `Pstatus` | Binary | Parent's cohabitation status ('T' - living together or 'A' - apart) |
| `Medu` | Numeric | Mother's education (0 - none, 1 - 4th grade, 2 - 5th to 9th, 3 - secondary, 4 - higher) |
| `Fedu` | Numeric | Father's education (0 - none, 1 - 4th grade, 2 - 5th to 9th, 3 - secondary, 4 - higher) |
| `Mjob` | Nominal | Mother's job ('teacher', 'health' care, 'services', 'at_home', 'other') |
| `Fjob` | Nominal | Father's job ('teacher', 'health' care, 'services', 'at_home', 'other') |
| `reason` | Nominal | Reason to choose school ('home', 'reputation', 'course', 'other') |
| `guardian` | Nominal | Student's guardian ('mother', 'father', 'other') |
| `traveltime` | Numeric | Home to school travel time (1 - <15 min, 2 - 15-30 min, 3 - 30 min-1 hour, 4 - >1 hour) |
| `studytime` | Numeric | Weekly study time (1 - <2 hours, 2 - 2-5 hours, 3 - 5-10 hours, 4 - >10 hours) |
| `failures` | Numeric | Number of past class failures ($n$ if $1 \le n < 3$, else 4) |
| `schoolsup` | Binary | Extra educational support (yes or no) |
| `famsup` | Binary | Family educational support (yes or no) |
| `paid` | Binary | Extra paid classes within the course subject (yes or no) |
| `activities`| Binary | Extra-curricular activities (yes or no) |
| `nursery` | Binary | Attended nursery school (yes or no) |
| `higher` | Binary | Wants to take higher education (yes or no) |
| `internet` | Binary | Internet access at home (yes or no) |
| `romantic` | Binary | With a romantic relationship (yes or no) |
| `famrel` | Numeric | Quality of family relationships (from 1 - very bad to 5 - excellent) |
| `freetime` | Numeric | Free time after school (from 1 - very low to 5 - very high) |
| `goout` | Numeric | Going out with friends (from 1 - very low to 5 - very high) |
| `Dalc` | Numeric | Workday alcohol consumption (from 1 - very low to 5 - very high) |
| `Walc` | Numeric | Weekend alcohol consumption (from 1 - very low to 5 - very high) |
| `health` | Numeric | Current health status (from 1 - very bad to 5 - very good) |
| `absences` | Numeric | Number of school absences (from 0 to 93) |
| `G1` | Numeric | First period grade (0 to 20) — **Dropped to prevent leakage** |
| `G2` | Numeric | Second period grade (0 to 20) — **Dropped to prevent leakage** |
| `G3` | Numeric | Final grade (0 to 20) — **Converted to Binary Target (`Pass` >= 10, `Fail` < 10)** |

## 4. Privacy & Leakage Safeguards
1. **Target Leakage Prevention:** In real educational intervention systems, educators need early predictions before mid-semester examinations. Utilizing `G1` and `G2` creates an artificial shortcut yielding inflated performance ($>90\%$) that fails to evaluate true early risk indicators. Thus, `G1` and `G2` are strictly excluded from all training feature matrices.
2. **De-identification:** Direct identifiers (names, IDs, phone numbers) are not present in the dataset. Quasi-identifiers (`age`, `sex`, `address`, `school`, `Medu`, `Fedu`) are analyzed for potential re-identification and evaluated under anonymization transformations.
