import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef,
                             confusion_matrix, classification_report)
import matplotlib.pyplot as plt
import seaborn as sns

# Page config
st.set_page_config(
    page_title="ML Classification Models",
    page_icon="🤖",
    layout="wide"
)

st.title("Credit Card Fraud Detection - ML Classification Models")
st.markdown("---")

st.sidebar.header("Model Selection")
st.sidebar.markdown("Choose a model to evaluate:")

# Model options
model_options = {
    'Logistic Regression': 'model/logistic_regression_model.pkl',
    'Decision Tree': 'model/decision_tree_model.pkl',
    'KNN': 'model/knn_model.pkl',
    'Naive Bayes': 'model/naive_bayes_model.pkl',
    'Random Forest': 'model/random_forest_model.pkl',
    'XGBoost': 'model/xgboost_model.pkl'
}

selected_model = st.sidebar.selectbox(
    "Select Model:",
    list(model_options.keys())
)

# Load scaler
@st.cache_resource
def load_scaler():
    return joblib.load('model/scaler.pkl')

# Load model
@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)

scaler = load_scaler()

# DATA PREVIEW SECTION
st.header("Upload Test Data")
st.markdown("Upload a CSV file with the same features as the training data (without 'Class' column for prediction, or with 'Class' for evaluation)")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

# Option to use sample data
use_sample = st.checkbox("Use sample test data (provided)", value=True)

if use_sample:
    try:
        df = pd.read_csv('data/test_data.csv')
        st.success(f" Loaded sample data: {df.shape[0]} rows, {df.shape[1]} columns")
    except:
        st.error(" Sample data file not found! Please upload your own CSV.")
        st.stop()
elif uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success(f"Uploaded data: {df.shape[0]} rows, {df.shape[1]} columns")
else:
    st.info("Please upload a CSV file or check 'Use sample test data'")
    st.stop()

# Show data preview
with st.expander("View Data Preview"):
    st.dataframe(df.head(10))

# PREPARE DATA FOR PREDICTION

has_labels = 'Class' in df.columns

if has_labels:
    X_test = df.drop('Class', axis=1)
    y_test = df['Class']
else:
    X_test = df
    y_test = None

# Scale features
X_test_scaled = scaler.transform(X_test)

# LOAD AND PREDICT
st.markdown("---")
st.header(f"Model: {selected_model}")

# Add spinner for loading
with st.spinner(f'Loading {selected_model} and generating predictions...'):
    model = load_model(model_options[selected_model])
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Get prediction probabilities if available
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        y_pred_proba = y_pred

st.success('Predictions complete!')

# DISPLAY METRICS 
if has_labels:
    st.subheader("Evaluation Metrics")
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Accuracy", f"{accuracy:.4f}")
        st.metric("AUC Score", f"{auc:.4f}")
    
    with col2:
        st.metric("Precision", f"{precision:.4f}")
        st.metric("Recall", f"{recall:.4f}")
    
    with col3:
        st.metric("F1 Score", f"{f1:.4f}")
        st.metric("MCC Score", f"{mcc:.4f}")
    
    # CONFUSION MATRIX
    st.markdown("---")
    st.subheader("Confusion Matrix")

    with st.spinner('Generating confusion matrix...'):
        cm = confusion_matrix(y_test, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Not Fraud', 'Fraud'],
                    yticklabels=['Not Fraud', 'Fraud'])
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.title(f'Confusion Matrix - {selected_model}')
        st.pyplot(fig)
    
    # CLASSIFICATION REPORT
    st.markdown("---")
    st.subheader("Classification Report")
    
    report = classification_report(y_test, y_pred, 
                                target_names=['Not Fraud', 'Fraud'],
                                output_dict=True)
    report_df = pd.DataFrame(report).transpose()

    # Format
    st.dataframe(report_df.style.format({
        'precision': '{:.4f}',
        'recall': '{:.4f}',
        'f1-score': '{:.4f}',
        'support': '{:.0f}'
    }))

else:
    # only predictions
    st.subheader("Predictions")
    st.info("No 'Class' column found - showing predictions only (no evaluation metrics)")
    
    results_df = X_test.copy()
    results_df['Predicted_Class'] = y_pred
    results_df['Fraud_Probability'] = y_pred_proba
    
    st.dataframe(results_df.head(20))
    
    # Summary
    fraud_count = sum(y_pred == 1)
    st.metric("Predicted Frauds", f"{fraud_count} / {len(y_pred)}")

# FOOTER
st.markdown("---")
st.markdown("""
**Dataset:** Credit Card Fraud Detection (Kaggle)  
**Models:** Logistic Regression | Decision Tree | KNN | Naive Bayes | Random Forest | XGBoost  
**Assignment:** ML Classification with Streamlit Deployment
""")