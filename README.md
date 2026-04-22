# Customer Churn Prediction - Telecom Retention Modeling

An end-to-end machine learning project to predict customer churn probability in a telecom company, with threshold optimization, SHAP explainability, and business ROI estimation for the retention team.

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7.6-orange)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-green)](https://shap.readthedocs.io/)
[![Kaggle](https://img.shields.io/badge/Dataset-Kaggle-20BEFF?logo=kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

---

## Business Problem

Acquiring a new customer costs 5-7x more than retaining an existing one. For a telecom company losing 26% of its customers annually, identifying who is about to leave and why is a direct revenue problem.

This project builds a churn scoring model that outputs a default probability per customer, segments customers into risk tiers, and generates a prioritized retention list for the customer success team. The model is evaluated on business-relevant metrics, not just accuracy, with threshold optimization aligned to the real cost of missing a churner versus a false alarm.

---

## Dataset

| Property | Details |
|---|---|
| Source | [Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) |
| Size | 7,043 customers |
| Raw Features | 21 |
| Target Variable | Binary — `1` = Churned, `0` = Retained |
| Churn Rate | 26.58% after cleaning |

---

## Project Structure

```
customer-churn-prediction/
│
├── customer-churn-prediction.ipynb
├── outputs/
│   ├── model_comparison.csv
│   └── retention_priority_list.csv
└── README.md
```

---

## Methodology

### 1. Exploratory Data Analysis

- Confirmed 26.58% churn rate moderate imbalance requiring careful metric selection
- `TotalCharges` was stored as a string dtype due to whitespace encoding; converted to numeric with 11 affected rows dropped
- Contract type showed the strongest categorical signal — month-to-month customers churn at 42.71% vs 2.85% for two-year contracts
- Cohort analysis revealed new customers (0–6 months tenure) churn at 53.33% vs 19.51% for established customers — the first 6 months is the highest risk window
- Customers with a high-risk profile (month-to-month + no online security + electronic check payment) churn at 58.63%

### 2. Feature Engineering

10+ domain-driven features were created across four categories:

| Category | Features |
|---|---|
| Service Features | `TotalServices`, `ChargePerService` |
| Charges Features | `ChargesTenureRatio`, `ChargesGap`, `TotalChargesExpected` |
| Tenure Risk Flags | `IsNewCustomer`, `IsLoyalCustomer`, `TenureScore` |
| Contract & Payment Flags | `IsMonthToMonth`, `IsElectronicCheck`, `NoOnlineSecurity`, `NoTechSupport`, `HighRiskProfile` |

### 3. Modeling

Three models were trained and compared with stratified 80/20 split:

| Model | AUC-ROC | Recall | Precision | F1-Score |
|---|---|---|---|---|
| **Logistic Regression** | **0.8383** | 0.7968 | 0.5017 | 0.6157 |
| Random Forest | 0.8132 | 0.4866 | 0.6212 | 0.5457 |
| XGBoost | 0.8254 | 0.7326 | 0.5170 | 0.6062 |

Logistic Regression achieved the highest AUC-ROC (0.8383) on this dataset, demonstrating that a well-tuned linear model can outperform complex ensembles when features are informative and the dataset is relatively small.

### 4. Threshold Optimization

The default classification threshold of 0.5 was replaced with an optimized threshold of 0.40 on XGBoost, selected by maximizing F1-Score across the Precision-Recall curve:

| Threshold | Recall | Precision | F1-Score |
|---|---|---|---|
| 0.50 (default) | 0.7326 | 0.5170 | 0.6062 |
| 0.40 (optimized) | 0.8128 | 0.4975 | 0.6173 |

Lowering the threshold increased Recall by 8 percentage points — meaning fewer churners are missed — at an acceptable cost to Precision. In a retention context, missing a churner is more costly than contacting a customer who would have stayed.

### 5. Explainability

SHAP was applied to the XGBoost model to generate interpretable explanations:

- **Global importance** — features driving churn predictions across all customers
- **Beeswarm plot** — direction and magnitude of each feature's impact per customer
- **Waterfall plot** — individual customer-level explanation for high-risk predictions
- **Dependence plot** — interaction between tenure and monthly charges on churn risk

### 6. Business ROI Estimation

| Metric | Value |
|---|---|
| Average monthly charge | $64.80 |
| Churners correctly flagged | 304 |
| False alarms | 307 |
| Total customers to contact | 611 |
| Estimated customers saved (30% retention rate) | 91 |
| Estimated annual revenue saved | $70,759.64 |
| Retention campaign cost ($20 per customer) | $12,220.00 |
| Net ROI | $58,539.64 |
| ROI % | 479% |

---

## Key Findings

**Tenure is the strongest predictor of churn.** Customers in their first 6 months churn at 53.33% — nearly three times the rate of established customers. Onboarding experience is the single most actionable retention lever.

**Contract type is the most actionable business feature.** Month-to-month customers churn at 42.71% compared to 2.85% for two-year contracts. Incentivizing customers to upgrade to longer contracts during the first 6 months directly addresses the highest-risk segment.

**The high-risk profile flag is highly predictive.** Customers who are simultaneously on a month-to-month contract, paying by electronic check, and have no online security churn at 58.63% — more than double the base rate.

**Logistic Regression outperformed XGBoost on AUC-ROC.** On a clean, relatively small dataset with well-engineered features, model complexity does not always win. This is an important finding for model selection in production environments where interpretability and simplicity matter.

**Threshold selection has direct business impact.** Shifting from 0.50 to 0.40 recovered additional true churners in the test set, translating to meaningful incremental revenue retention at low additional campaign cost.

---

## Business Recommendation

Prioritize outreach to customers flagged as Critical Risk (churn probability > 0.70) and High Risk (0.50–0.70), focusing specifically on those in their first 6 months with a month-to-month contract. Offering a discounted annual contract upgrade during this window directly addresses the two strongest churn signals simultaneously.

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.10 |
| ML Models | XGBoost, Scikit-learn |
| Explainability | SHAP |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Environment | Kaggle Notebooks |

---

## Author

**Om Kiran Kadam**
B.Tech - Artificial Intelligence & Data Science, Sanjivani University

[LinkedIn](https://linkedin.com/in/omkadam05) · [GitHub](https://github.com/Omkadam01) · [Kaggle](https://www.kaggle.com/omkadam05)

---

## License

This project is licensed under the MIT License.
