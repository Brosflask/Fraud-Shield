"""Main Streamlit application for fraud detection."""
import streamlit as st

# Import modules
from loader import get_available_models, load_artifacts
from config import MODEL_CONFIGS
from input_validation import validate
from predictor import FraudPredictor

from visualization.components import (
    display_prediction_result, display_validation_results,
    create_sidebar
)
from visualization.charts import (
    create_fraud_gauge,
    create_transaction_summary,
    create_feature_importance_chart,
    create_feature_contribution_chart  
)


# Page configuration
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


def main():
    """Main application entry point."""
    
    # Title
    st.title("Fraud Detection System")
    st.markdown("""
        Enter transaction details below to check for potential fraud. 
        The system will analyze the transaction and provide a risk assessment.
    """)
    
    # Initialize components
    #model_loader = get_model_loader()
    #validator = create_validator()
    
    # Get available models
    available_models = get_available_models()
    
    # Create sidebar with proper return values

    selected_model = create_sidebar(available_models)

    if not selected_model:
        st.error("Please select a model")
    
    # Main form
    with st.form("transaction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Transaction Details")
            txn_type = st.selectbox(
                "Transaction Type",
                ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
            )
            amount = st.number_input(
                "Amount ($)", 
                min_value=0.01, 
                value=1000.00,
                step=100.00, 
                format="%.2f"
            )
            step = st.number_input(
                "Hour of Transaction",
                min_value=0,
                max_value=743,
                value=0,
                help="0-743 hours in the simulation"
            )
        
        with col2:
            st.subheader("Sender Details")
            old_orig = st.number_input(
                "Sender Balance Before ($)",
                min_value=0.00,
                value= 5000.00,
                step=100.00,
                format="%.2f"
            )
            new_orig = st.number_input(
                "Sender Balance After ($)",
                min_value=0.00,
                value= 4000.00,
                step=100.00,
                format="%.2f"
            )
            
            st.subheader("Recipient Details")
            old_dest = st.number_input(
                "Recipient Balance Before ($)",
                min_value=0.00,
                value= 1000.00,
                step=100.00,
                format="%.2f"
            )
            new_dest = st.number_input(
                "Recipient Balance After ($)",
                min_value=0.00,
                value=2000.00,
                step=100.00,
                format="%.2f"
            )
        
        submitted = st.form_submit_button("Check for Fraud", type="primary", use_container_width=True)
    
    # Prediction logic
    if submitted:
        # Validate inputs
        errors, warnings = validate(
            txn_type, amount, old_orig, new_orig, old_dest, new_dest
        )
        
        # Display validation results
        display_validation_results(errors, warnings)
        
        # Stop if there are errors
        if errors:
            st.stop()
        
        # Get model config and adjusted threshold
        try:
           model, feature_names, com_para = load_artifacts(selected_model)
           threshold = com_para[0]
           feature_importance = com_para[2]

        except Exception as e:
           st.error(f"Error loading: {str(e)}")
           st.stop()
        # Prepare input data
        input_data = {
            "step": step,
            "type": txn_type,
            "amount": amount,
            "oldbalanceOrig": old_orig,
            "newbalanceOrig": new_orig,
            "oldbalanceDest": old_dest,
            "newbalanceDest": new_dest
        }
        

                # Make prediction
        with st.spinner("Analyzing transaction..."):
           try:
            predictor = FraudPredictor(model, feature_names,model_name=com_para[1] ,threshold=com_para[0])        

            probability, is_fraud, X = predictor.predict(
                        step, txn_type, amount,
                        old_orig, new_orig, old_dest, new_dest
                    )
           except Exception as e:
                           st.error(f" Error making prediction: {str(e)}")     
            # Display results
           st.markdown("---")
           st.subheader("Prediction Results")
                
            # Display metrics
           display_prediction_result(probability, threshold, is_fraud) # type: ignore
                
            # Create gauge chart
           gauge = create_fraud_gauge(probability, threshold) # type: ignore
           st.plotly_chart(gauge, use_container_width=True)
                
            # Display transaction summary
           st.subheader("Transaction Summary")
           create_transaction_summary(input_data)
        
            # Display feature details
           with st.expander("🔍 Detailed Feature Analysis"):
                cols = st.columns(3)
                
                with cols[0]:
                    st.metric("Transaction Type", txn_type)
                    st.metric("Hour", step)
                
                with cols[1]:
                    st.metric("Sender Balance Change", f"${new_orig - old_orig:,.2f}")
                    st.metric("Sender Balance Ratio", f"{(new_orig / (old_orig + 1)):.3f}")
              #Feature Importance Section
           st.subheader("Feature Importance Analysis")
   
           tab1, tab2 = st.tabs(["Feature Importance", "Feature Contributions"])
           with tab1:
        # Show overall feature importance
             if feature_importance is not None:
               fig = create_feature_importance_chart(feature_importance, top_n=10)
               st.plotly_chart(fig, use_container_width=True)
            
            # Show full table
               with st.expander("View All Features"):
                  st.dataframe(
                    feature_importance.style.format({'Importance': '{:.2%}'}),
                    use_container_width=True
                )
             else:
                st.info("Feature importance data not available for this model.")
           with tab2:
        # Show current feature contributions
               if feature_importance is not None:
            # Get current feature values
                   current_values = {
                'step': step,
                'amount': amount,
                'oldbalanceOrig': old_orig,
                'newbalanceOrig': new_orig,
                'oldbalanceDest': old_dest,
                'newbalanceDest': new_dest,
                # Add derived features
                'balanceChangeDest': new_dest - old_dest,
                'amountToDestBalance': amount / (old_dest + 1) if (old_dest + 1) != 0 else 0
            }
            
                   fig = create_feature_contribution_chart(current_values, feature_importance, top_n=10)
                   st.plotly_chart(fig, use_container_width=True)
            
            # Show interpretation
                   st.info("""
            **How to interpret this chart:**
            - **Blue bars** Importance of feature
            - **Red stars** Current value of each feature (normalized)
            """)
               else:
                  st.info("Feature contribution data not available for this model.")   

          


if __name__ == "__main__":
    main()
