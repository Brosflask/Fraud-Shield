# FraudShield AI

A machine learning-powered web application for detecting **mobile money fraud**, built with Python, scikit-learn, XGBoost, and Streamlit. FraudShield scores a single transaction's fraud probability for a human fraud analyst to review. It is a decision-support tool, not an autonomous approve/decline system.

**Live Demo:** https://fraud-shield-cxoneh5rn7rjt4kmnlbjcs.streamlit.app/

---

## Project Overview

Mobile money fraud is rare, 0.13% of transactions in our data, but costly: for many users, a mobile money account is their only formal financial safety net, so a single successful fraud is a direct personal loss, not just a line item on a balance sheet. This project applies supervised machine learning to detect fraudulent mobile money transactions using **PaySim**, a synthetic mobile money simulator (6,362,604 transactions after cleaning), calibrated against real transaction logs from an African mobile money service.

This project was developed for the **CS 254 — Introduction to Artificial Intelligence** course and demonstrates a complete machine learning workflow: data cleaning, exploratory analysis, feature engineering, model comparison, rigorous evaluation under severe class imbalance, and deployment as a working web application.

> **Note on dataset history:** this project originally began with the ULB "Credit Card Fraud Detection" dataset (see the Week 9 proposal). The team switched to PaySim after proposal approval — PaySim's features (transaction type, amount, account balances) are directly interpretable, unlike ULB's anonymized PCA components, which gives the app a much stronger transparency story: it can explain *why* a transaction was flagged, not just that it was.

---

## Models

Three supervised classifiers are trained and available for comparison in the app. Results below are from the team's official held-out test set (15,000 transactions, 19 fraud cases), evaluated once, at each model's own configured threshold:

| Model | Threshold | PR-AUC | ROC-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| **XGBoost (recommended)** | 0.16 | 0.950 | 0.9999 | 0.889 | 0.842 | 0.865 |
| Random Forest* | 0.50 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Logistic Regression | 0.50 | 0.756 | 0.999 | 0.034 | 1.000 | 0.065 |

**XGBoost is the model this project selects and recommends.** It catches 16 of 19 fraud cases with only 2 false positives out of 14,981 legitimate transactions.

*\*Random Forest's perfect score is a known leakage artifact, not genuine superiority — see below. It is not the recommended model.*

Logistic Regression catches every fraud case but at a steep cost: 547 false positives out of 14,981 legitimate transactions (3.4% precision) — in practice, an analyst using this model would spend most of their review time on false alarms. This is why the project's model-selection metric is PR-AUC, not recall or accuracy: at a 0.13% fraud rate, a model that predicts "never fraud" scores 99.87% accuracy while catching nothing, and recall alone rewards a model like this Logistic Regression one.

## A leakage issue found — twice

An early XGBoost model trained with an engineered feature, `errorBalanceOrig`, scored a suspicious PR-AUC of **1.0** during validation — every fraud case caught, zero false positives. That's a warning sign, not an achievement. Investigation showed `errorBalanceOrig` has ~0 variance across all fraud transactions (std ≈ 4.9×10⁻¹¹), an artifact of how PaySim's simulator scripts its injected fraud records, not a real, generalizable pattern. XGBoost was retrained without this feature.

**The same pattern reappeared independently during final integration testing.** Random Forest — trained separately, on the original feature set that still includes `errorBalanceOrig` — scores a perfect 1.0 across every metric on the same genuinely held-out test set. Two independent models both hitting an exact 1.0 on a rare, hard classification problem isn't a coincidence; it's the same leakage artifact confirmed a second time. This is disclosed here rather than presented as Random Forest being the best model — see the Final Report's Methodology and Limitations sections for the full derivation.

## Key finding from the data

Fraud occurs **exclusively** in `TRANSFER` and `CASH_OUT` transactions — never `CASH_IN`, `PAYMENT`, or `DEBIT`. Fraudulent transactions almost always drain the sender's account to exactly zero (98.0% of cases, vs. 81.2% for legitimate transactions of the same types) while leaving the recipient's balance unchanged — the money doesn't visibly land anywhere.

---

## Project Structure

```
Fraud-Shield/
├── src/
│   ├── app.py                  # Streamlit entry point
│   ├── config.py                # model configs, transaction types
│   ├── loader.py                # loads model/feature/threshold artifacts
│   ├── feature_builder.py       # builds the model's feature row from form input
│   ├── predictor.py             # runs a prediction
│   ├── input_validation.py      # form input validation (errors + warnings)
│   └── visualization/
│       ├── components.py        # UI components (sidebar, result display)
│       └── charts.py            # fraud gauge, feature importance charts
├── model_container/
│   ├── xgboost_fraud_model.pkl
│   ├── random_f_fraud_model.pkl
│   ├── logistic_fraud_model.pkl
│   ├── logistic_scaler.pkl
│   ├── reduced_features.pkl     # XGBoost's (corrected) 13-feature set
│   ├── feature_list.pkl         # Random Forest / Logistic Regression's 16-feature set
│   └── *_feature_importance.pkl
├── new_trained_ds .ipynb         # exploratory training notebook — includes the
│                                  # full-feature model that surfaced the leakage
│                                  # issue described above; kept for transparency
├── paysim_cleaned_sample (1).csv # sample training data used by the notebook
├── requirements.txt
└── README.md
```

## Setup & Running

```bash
pip install -r requirements.txt
cd src
streamlit run app.py
```

## Usage Example

1. Select a model (XGBoost, Random Forest, or Logistic Regression) from the sidebar.
<img width="382" height="695" alt="image" src="https://github.com/user-attachments/assets/362455d0-61e9-45c1-a862-14fe177ad01d" />

2. Enter a transaction's details: type, amount, sender balance before/after, recipient balance before/after
3. Click **Check for Fraud**. The app returns a fraud probability, a flagged/legitimate status, and (where available) a feature importance breakdown for that model.

**Example — legitimate:** `CASH_OUT`, amount 20,000, sender 50,000 → 30,000, recipient 10,000 → 30,000 (properly credited). Expected: low fraud probability.
<img width="1532" height="878" alt="image" src="https://github.com/user-attachments/assets/1f643ebc-8b9b-44a7-bc15-aab463113584" />
<img width="1458" height="721" alt="image" src="https://github.com/user-attachments/assets/305c0632-8374-46a6-9524-f689813e2e7a" />

**Example — fraud pattern:** `TRANSFER`, amount 450,000, sender 450,000 → 0 (fully drained), recipient 0 → 0 (unchanged despite the transfer). Expected: high fraud probability.
<img width="1535" height="865" alt="image" src="https://github.com/user-attachments/assets/82f25c75-b078-411c-b247-d0b01d9e0930" />
<img width="1448" height="736" alt="image" src="https://github.com/user-attachments/assets/7564e351-e067-4c6a-b635-a506aec4c1c7" />

4. **View feature after fraud**- Scroll down and open the feature importance analysis section Random Forest/XGboost classifiers
<img width="1381" height="666" alt="image" src="https://github.com/user-attachments/assets/231132cd-798e-4c16-8871-e0d270657e83" />
<img width="1370" height="643" alt="image" src="https://github.com/user-attachments/assets/d5c1e600-4a19-44c6-8275-8b94335dc178" />
---

## Ethical Considerations
 
- **Bias:** PaySim is a simulation, not real transaction data, and has no demographic fields — the model can't discriminate on attributes it never sees, but can't be audited for demographic disparity either.
- **Fairness:** false-positive cost varies dramatically by which model is deployed — roughly 1 in 9 flags is a false alarm with XGBoost, versus the overwhelming majority with Logistic Regression. Which model an operator chooses is itself a fairness-relevant decision. Every flag, regardless of model, is reviewed by a human analyst before any consequence reaches a customer.
- **Privacy:** PaySim is fully synthetic and pre-anonymized. Account identifiers are excluded from the model's features entirely.
- **Transparency:** Every prediction can be inspected. The app shows the exact feature values and, where available, feature importance behind each result.

Full analysis in the project's Final Report.

## Team

| Role | Person |
|---|---|
| Data Engineer | James |
| AI/ML Engineer | Salma |
| Application Developer | Curtis |
| Documentation & Testing | Ronald |

Developed for **CS 254 — Introduction to Artificial Intelligence**.

