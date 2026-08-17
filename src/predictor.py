from config import MODEL_CONFIGS
import streamlit as st
from feature_builder import build_row, set_feature_names
import joblib

"""Prediction logic for fraud detection."""

class FraudPredictor:
    """Handles fraud predictions."""
    
    def __init__(self, model, feature_names: list, model_name: str, threshold: float = 0.16, ):
        """
        Initialize predictor.
        
        Args:
            model: Trained ML model
            feature_names: List of feature names
            threshold: Classification threshold
        """
        self.model = model
        self.feature_builder = set_feature_names(feature_names)
        self.threshold = threshold
        self.model_name = model_name
    
    def predict(self, step: int, txn_type: str, amount: float,
                old_orig: float, new_orig: float,
                old_dest: float, new_dest: float):
        
        """
        Make a prediction on a single transaction.
        
        Returns:
            Tuple of (probability, is_fraud, feature_row)
        """
        X = build_row(
            step, txn_type,
            amount, old_orig,
            new_orig, old_dest, new_dest
        )
        for i in X.columns:
            print(f"Feature: {i}, Value: {X[i]}")
        print(len(X.columns))

        if self.model_name == "Logistic Regression":
            try:
              scaler = joblib.load(MODEL_CONFIGS[self.model_name]["scaler"])
              print("Scaler called")
            except Exception:
                st.error("Scaler for features not loaded")  
            X = scaler.fit(X)     # type: ignore

        probability = float(self.model.predict_proba(X)[0, 1])
        is_fraud = probability >= self.threshold
        
        return probability, is_fraud, X
