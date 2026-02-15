"""Streamlit dashboard for Credit Risk Model."""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_processing import engineer_features

# Page config
st.set_page_config(
    page_title="Credit Risk Dashboard",
    page_icon="💳",
    layout="wide"
)

# Title
st.title("💳 Credit Risk Scoring Dashboard")
st.markdown("### Buy-Now-Pay-Later (BNPL) Risk Assessment Tool")

# Load model
@st.cache_resource
def load_model():
    model_path = Path("main/model.pkl")
    if not model_path.exists():
        st.error("Model not found. Please train the model first.")
        st.stop()
    return joblib.load(model_path)

model = load_model()

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio("Select Page", ["Risk Prediction", "Model Performance", "Data Insights"])

# ============= RISK PREDICTION PAGE =============
if page == "Risk Prediction":
    st.header(" Individual Risk Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Customer Transaction Data")
        amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=500.0, step=50.0)
        fraud_result = st.selectbox("Historical Fraud Flag", [0, 1], help="0 = No fraud, 1 = Fraud detected")
        
    with col2:
        st.subheader("Additional Features")
        st.info("Add more customer features here based on your data")
    
    if st.button("Calculate Risk Score", type="primary"):
        # Create sample dataframe
        input_data = pd.DataFrame({
            "Amount": [amount],
            "FraudResult": [fraud_result]
        })
        
        try:
            X, _ = engineer_features(input_data)
            
            # Handle missing columns
            model_features = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else []
            for col in model_features:
                if col not in X.columns:
                    X[col] = 0
            X = X[model_features]
            
            risk_prob = model.predict_proba(X)[0, 1]
            
            # Display results
            st.markdown("---")
            st.subheader("Risk Assessment Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Risk Probability", f"{risk_prob:.2%}")
            
            with col2:
                risk_category = "HIGH RISK" if risk_prob > 0.5 else "LOW RISK"
                st.metric("Risk Category", risk_category)
            
            with col3:
                credit_score = int((1 - risk_prob) * 850)
                st.metric("Credit Score", credit_score)
            
            # Risk gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk_prob * 100,
                title={'text': "Risk Level (%)"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkred" if risk_prob > 0.5 else "green"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "lightcoral"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            st.plotly_chart(fig, use_container_width=True)
            
            # Recommendation
            if risk_prob < 0.3:
                st.success(" **Recommendation:** APPROVE - Low risk customer")
            elif risk_prob < 0.7:
                st.warning(" **Recommendation:** REVIEW - Medium risk, manual revie")
            else:
                st.error(" **Recommendation:** DECLINE - High risk customer")
                
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")

# ============= MODEL PERFORMANCE PAGE =============
elif page == "Model Performance":
    st.header(" Model Performance Metrics")
    
    # Mock performance metrics (replace with actual validation results)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("ROC-AUC Score", "0.87", delta="0.02")
    with col2:
        st.metric("Precision", "82%", delta="3%")
    with col3:
        st.metric("Recall", "79%", delta="1%")
    with col4:
        st.metric("F1 Score", "0.80", delta="0.02")
    
    st.markdown("---")
    
    # Model comparison
    st.subheader("Model Comparison")
    model_comparison = pd.DataFrame({
        'Model': ['Logistic Regression', 'Random Forest', 'Gradient Boosting'],
        'ROC-AUC': [0.82, 0.87, 0.85],
        'Precision': [0.78, 0.82, 0.80],
        'Recall': [0.75, 0.79, 0.77],
        'Training Time (s)': [2.3, 45.2, 67.8]
    })
    
    fig = px.bar(model_comparison, x='Model', y=['ROC-AUC', 'Precision', 'Recall'],
                 title="Model Performance Comparison",
                 barmode='group')
    st.plotly_chart(fig, use_container_width=True)
    
    # Confusion Matrix
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Confusion Matrix")
        # Mock confusion matrix
        cm_data = np.array([[850, 150], [120, 880]])
        fig = px.imshow(cm_data, 
                        labels=dict(x="Predicted", y="Actual", color="Count"),
                        x=['Low Risk', 'High Risk'],
                        y=['Low Risk', 'High Risk'],
                        text_auto=True,
                        color_continuous_scale='Blues')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("ROC Curve")
        # Mock ROC curve
        fpr = np.linspace(0, 1, 100)
        tpr = np.sqrt(fpr)  # Mock curve
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC Curve'))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', 
                                name='Random Classifier', line=dict(dash='dash')))
        fig.update_layout(
            title='ROC Curve (AUC = 0.87)',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate'
        )
        st.plotly_chart(fig, use_container_width=True)

# ============= DATA INSIGHTS PAGE =============
elif page == "Data Insights":
    st.header(" Data Insights & Distribution")
    
    # Load sample data
    data_path = Path("data/raw/data.csv")
    if data_path.exists():
        df = pd.read_csv(data_path)
        
        st.subheader("Dataset Overview")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Records", f"{len(df):,}")
        with col2:
            if 'FraudResult' in df.columns:
                high_risk_pct = (df['FraudResult'].sum() / len(df)) * 100
                st.metric("High Risk %", f"{high_risk_pct:.1f}%")
        with col3:
            st.metric("Features", len(df.columns))
        
        st.markdown("---")
        
        # Risk distribution
        if 'FraudResult' in df.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Risk Distribution")
                risk_counts = df['FraudResult'].value_counts()
                fig = px.pie(values=risk_counts.values, 
                            names=['Low Risk', 'High Risk'],
                            title='Customer Risk Distribution',
                            color_discrete_sequence=['green', 'red'])
               y_chart(fig, use_container_width=True)
            
            with col2:
                if 'Amount' in df.columns:
                    st.subheader("Amount Distribution by Risk")
                    fig = px.box(df, x='FraudResult', y='Amount',
                                title='Transaction Amount by Risk Category',
                                labels={'FraudResult': 'Risk Category', 'Amount': 'Amount ($)'})
                    st.plotly_chart(fig, use_container_width=True)
        
        # Feature correlation
        st.subheader("Feature Correlation Heatmap")
        numeric_cols = df.select_dtypes(include=[np.number]).columns[:10]  # Limit to 10 features
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            fig = px.imshow(corr_matrix, 
                           text_auto='.2f',
                           aspect='auto',
                           color_continuous_scale='RdBu_r',
                           title='Feature Correlation Matrix')
            st.plotly_chart(fig, use_container_width=True)
        
        # Data sample
        st.subheader("Sample Data")
        st.dataframe(df.head(100), use_container_width=True)
        
    else:
        st.warning("No data file found at data/raw/data.csv")
        st.info("Upload your data to see insights")

# Footer
st.markdown("---")
st.markdown("**Credit Risk Model Dashboard** | Bati Bank BNPL Project | Built with Streamlit")
