MODEL_CONFIGS = {
    "XGBoost": {
        "model_file": "model_container/xgboost_fraud_model.pkl",
        "threshold_file": 0.16,
        "features_file":"model_container/reduced_features.pkl",
        "description": "XGBoost Classifier",
        "color": "#ff6b6b",
        "importance": "model_container/xgb_feature_importance.pkl"
    },

    "Random Forest": {
        "model_file": "model_container/random_f_fraud_model.pkl",
        "threshold_file": 0.5,
        "features_file": "model_container/feature_list.pkl",
        "description": "Random Forest Classifier",
        "color":"#4dabf7",
        "importance": "model_container/rf_feature_importance.pkl"
    },
    "Logistic Regression": {
        "model_file": "model_container/logistic_fraud_model.pkl",
        "threshold_file": 0.5,
        "features_file": "model_container/feature_list.pkl",
        "description": "Logistic Regression Classifier",
        "color": "#e599f7",
        "scaler": "model_container/logistic_scaler.pkl"
    }

}

TRANSACTION_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]

FRAUD_CAPABLE_TYPES = {"TRANSFER", "CASH_OUT"}
