# FraudShield AI

A machine learning-powered web application for detecting **mobile money fraud**, built with Python, scikit-learn, XGBoost, and Streamlit. FraudShield scores a single transaction's fraud probability for a human fraud analyst to review — it is a decision-support tool, not an autonomous approve/decline system.

**Live Demo:** https://fraud-shield-cxoneh5rn7rjt4kmnlbjcs.streamlit.app/

---

## Project Overview

Mobile money fraud is rare — 0.13% of transactions in our data — but costly: for many users, a mobile money account is their only formal financial safety net, so a single successful fraud is a direct personal loss, not just a line item on a balance sheet. This project applies supervised machine learning to detect fraudulent mobile money transactions using **PaySim**, a synthetic mobile money simulator (6,362,604 transactions after cleaning), calibrated against real transaction logs from an African mobile money service.

This project was developed for the **CS 254 — Introduction to Artificial Intelligence** course and demonstrates a complete machine learning workflow: data cleaning, exploratory analysis, feature engineering, model comparison, rigorous evaluation under severe class imbalance, and deployment as a working web application.

> **Note on dataset history:** this project originally began with the ULB "Credit Card Fraud Detection" dataset (see the Week 9 proposal). The team switched to PaySim after proposal approval — PaySim's features (transaction type, amount, account balances) are directly interpretable, unlike ULB's anonymized PCA components, which gives the app a much stronger transparency story: it can explain *why* a transaction was flagged, not just that it was.

---

## Models

Three supervised classifiers were trained and compared on the same stratified train/validation/test split:

| Model | Validation PR-AUC |
|---|---|
| Logistic Regression | 0.563 |
| Random Forest | 0.893 |
| **XGBoost (selected)** | **0.950** |

**XGBoost** was selected as the final model, evaluated once on the held-out test set: **PR-AUC 0.954**, catching **18 of 19** fraud transactions (94.7% recall) with 10 false positives out of 14,981 legitimate transactions (64.3% precision), at a decision threshold of 0.90. All three trained models are available to compare directly in the app's model selector.

Accuracy is deliberately *not* the headline metric here: at a 0.13% fraud rate, a model that predicts "never fraud" scores 99.87% accuracy while catching nothing.

## A leakage issue we found and fixed

An early model trained with an engineered feature, `errorBalanceOrig`, scored a suspicious PR-AUC of **1.0** — every fraud case caught, zero false positives. That's a warning sign, not an achievement. Investigation showed `errorBalanceOrig` has ~0 variance across all fraud transactions (std ≈ 4.9×10⁻¹¹) — an artifact of how PaySim's simulator scripts its injected fraud records, not a real, generalizable pattern. The feature was removed and all models retrained on the corrected feature set.

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
|   |── full_features.pkl
│   ├── reduced_features.pkl
│   └── *_feature_importance.pkl
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
2. Enter a transaction's details: type, amount, sender balance before/after, recipient balance before/after.
3. Click **Check for Fraud** — the app returns a fraud probability, a flagged/legitimate status, and (where available) a feature importance breakdown for that model.

**Example — legitimate:** `CASH_OUT`, amount 20,000, sender 50,000 → 30,000, recipient 10,000 → 30,000 (properly credited). Expected: low fraud probability.

**Example — fraud pattern:** `TRANSFER`, amount 450,000, sender 450,000 → 0 (fully drained), recipient 0 → 0 (unchanged despite the transfer). Expected: high fraud probability.

---

## Ethical Considerations

- **Bias:** PaySim is a simulation, not real transaction data, and has no demographic fields — the model can't discriminate on attributes it never sees, but can't be audited for demographic disparity either.
- **Fairness:** at 64.3% precision, roughly 1 in 3 flags is a false positive. Every flag is reviewed by a human analyst before any consequence reaches a customer — the model never blocks a transaction on its own.
- **Privacy:** PaySim is fully synthetic and pre-anonymized. Account identifiers are excluded from the model's features entirely.
- **Transparency:** every prediction can be inspected — the app shows the exact feature values and, where available, feature importance behind each result.

Full analysis in the project's Final Report.

## Team

| Role | Person |
|---|---|
| Data Engineer | James |
| AI/ML Engineer | Salma |
| Application Developer | Curtis |
| Documentation & Testing | Ronald |

Developed for **CS 254 — Introduction to Artificial Intelligence**.

