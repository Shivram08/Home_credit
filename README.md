# Home Credit Default Risk Prediction

Predicting the probability of a client defaulting on a loan using structured financial data, leveraging advanced feature engineering and ensemble machine learning models.

---

##  Project Overview

This project aims to predict **loan default risk** for clients based on their application data and credit history using the [Home Credit Default Risk Kaggle dataset](https://www.kaggle.com/competitions/home-credit-default-risk).

Financial institutions can use such predictive models to:
- Reduce non-performing loans.
- Enable inclusive lending by accurately assessing client risk.
- Optimize lending decisions.

---

##  Dataset

- **Main Dataset:** `application_train.csv` (~307K clients, ~122 features).
- **Supplementary Tables:**
  - `bureau.csv`, `bureau_balance.csv` (credit bureau history).
  - `previous_application.csv`, `POS_CASH_balance.csv`, `installments_payments.csv`, `credit_card_balance.csv`.
- **Target Variable:**
  - `TARGET`: 1 = Default, 0 = No Default (imbalanced ~8% default rate).

---

##  Preprocessing

- **Missing Values:**
  - Imputed numeric features with median.
  - Handled categorical features appropriately.
- **Outlier Detection:**
  - Identified using IQR.
  - Applied capping for extreme values where appropriate.
- **Feature Engineering:**
  - Aggregated supplementary tables (`bureau`, `bureau_balance`) at the customer level using mean, min, max, and count.
  - Created ratio features (`CREDIT_TO_INCOME_RATIO`, `ANNUITY_TO_INCOME_RATIO`).
  - Converted `DAYS_*` features into interpretable forms (e.g., age).
- **Encoding:**
  - One-hot encoding for categorical variables.
  - Dropped low-importance and highly collinear features to reduce noise.

---

##  Modeling Approach

Utilized **ensemble tree-based models** for their robustness on tabular financial data:

### Models Used:
- **LightGBM:** Fast and efficient gradient boosting.
- **XGBoost:** Stable and controllable boosting.
- **CatBoost:** Automatic categorical handling and strong regularization.

### Training Strategy:
- **Stratified 5-Fold Cross-Validation** for robust validation and to address class imbalance.
- Hyperparameters tuned for stability and performance.
- Used **early stopping** to prevent overfitting.

---

##  Ensembling

Combined predictions from LightGBM, XGBoost, and CatBoost using **simple averaging**:
$\[
\text{Final Prediction} = \frac{\text{LightGBM} + \text{XGBoost} + \text{CatBoost}}{3}
\]$
to leverage the strengths of each model and reduce variance.

---

## Results

- Achieved **Validation AUC: ~0.7810** on the blended ensemble.
- Key predictive features:
  - External risk scores (`EXT_SOURCE_1/2/3`).
  - `DAYS_BIRTH` (age).
  - Credit-to-income and annuity-to-income ratios.

---

## Key Takeaways

- Built a **robust pipeline for structured tabular risk prediction**.
- Gained experience in **multi-table feature engineering, memory management, and advanced ensemble modeling**.
- Balanced **model interpretability, stability, and predictive power** for a real-world financial prediction task.

---

