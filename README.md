# Hepatitis Patient Outcome Classification

This project implements a **machine learning pipeline** to classify patient outcomes (Alive or Death) using the **Hepatitis dataset**.  
It covers **data preprocessing**, **exploratory data analysis (EDA)**, **feature engineering**, and **model training & evaluation** with multiple algorithms.

---

## Project Overview
- **Goal:** Predict patient outcome (`0` = Death, `1` = Alive) based on medical and demographic features.
- **Dataset:** Hepatitis dataset — 155 samples, ~19 features (clinical + laboratory test results).
- **Challenge:** Moderate class imbalance (~79% alive, ~21% death).
- **Approach:** Preprocessing, feature scaling, balancing via SMOTE, dimensionality reduction (PCA), and model evaluation with cross-validation.

---

## Steps & Methodology

### 1. Data Preprocessing
- Inspected dataset structure with `info()`, `head()`, and `describe()`.
- Converted categorical binary values (`1/2`) to (`0/1`) for `target` and `gender`.
- Handled missing values using **Iterative Imputer** (scikit-learn).
- Checked for duplicates — none found.
- Identified outliers using **IQR & Boxplots** (kept for modeling).

### 2. Exploratory Data Analysis (EDA)
- **Distribution Analysis:**  
  - Most numeric features are right-skewed.
  - Many binary clinical indicators (`fatigue`, `ascites`, `varices`, etc.) show higher prevalence among patients marked as alive.
- **Correlation Analysis:**  
  - Highest positive correlation with target: `ascites (0.48)`, `spiders`, `varices`.
  - Negative correlation: `age`, `histology`.
  - No severe multicollinearity detected.
- **Target Imbalance:** Alive ≈ 4× more common than Death.

### 3. Data Preparation for Modeling
- **Train-Test Split:** 70% train, 30% test (stratified by target).
- **Scaling:** Standardized numerical features with `StandardScaler`.
- **Class Balancing:** Applied **SMOTE** to oversample the minority class.
- **Dimensionality Reduction:** Used **PCA** to retain 95% variance → 16 components.

### 4. Models & Evaluation

We evaluated **Random Forest**, **SVM**, and **Multi-Layer Perceptron (MLP)** using 5-fold stratified cross-validation on the training set, followed by hyperparameter tuning and final testing.

#### Cross-Validation Results (Average ± Std)
| Model          | Accuracy       | Precision (Weighted) | Recall (Weighted) | F1 (Weighted) | ROC-AUC   |
|----------------|---------------|----------------------|-------------------|---------------|-----------|
| Random Forest  | 0.8953 ± 0.0146 | 0.8980 ± 0.0134      | 0.8953 ± 0.0146   | 0.8951 ± 0.0146 | **0.9744 ± 0.0056** |
| SVM            | **0.9126 ± 0.0267** | **0.9164 ± 0.0282** | **0.9126 ± 0.0267** | **0.9124 ± 0.0267** | 0.9716 ± 0.0159 |
| MLP            | 0.7911 ± 0.0961 | 0.8051 ± 0.0819      | 0.7911 ± 0.0961   | 0.7861 ± 0.1037 | 0.8951 ± 0.0541 |

#### Tuned Model (Final Test Performance)
| Model          | Train Acc | Test Acc | Train ROC-AUC | Test ROC-AUC | Test Precision | Test Recall | Test F1 |
|----------------|-----------|----------|---------------|--------------|----------------|-------------|---------|
| **MLP (Best Tuned)** | 0.9709    | **0.8511** | 0.9973        | **0.9351**   | **0.8719**    | **0.8511** | **0.8576** |

**Key Points:**
- **SVM** achieved the highest average accuracy in cross-validation.
- **Random Forest** recorded the highest ROC-AUC during cross-validation.
- **MLP**, after tuning, provided the best balanced performance on the test set, with strong recall and precision.

---

## Key Insights
- MLP achieved the best test set balance between classes.
- PCA reduced dimensionality without significant loss of information.
- SMOTE improved minority class recall.
- Clinical features such as `ascites`, `spiders`, `varices` are strong predictors.


