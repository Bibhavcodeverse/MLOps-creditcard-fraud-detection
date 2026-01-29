import streamlit as st
import pandas as pd
import requests
import json
import os
import subprocess
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import joblib
import sys

# Set Page Config
st.set_page_config(
    page_title="Credit Card Fraud MLOps Dashboard",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Look
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #1e2130;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #3e4451;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #2e3142;
        color: white;
        border: 1px solid #4e556c;
    }
    .stButton>button:hover {
        background-color: #4e556c;
        border-color: #6e7a9e;
    }
</style>
""", unsafe_allow_html=True)

# Helper for Running Scripts
def run_pipeline_step(script_name):
    with st.spinner(f"Running {script_name}..."):
        result = subprocess.run([sys.executable, script_name], capture_output=True, text=True)
        if result.returncode == 0:
            st.success(f"{script_name} completed successfully!")
            return result.stdout
        else:
            st.error(f"Error running {script_name}")
            st.code(result.stderr)
            return None

# API Configuration
API_URL = "http://localhost:8000/predict"

# Direct Model Load for Headless Deployment
@st.cache_resource
def load_local_model():
    try:
        model = joblib.load("models/ensemble_model.pkl")
        fe = joblib.load("models/feature_engineer.pkl")
        return model, fe
    except Exception as e:
        st.error(f"Failed to load local model: {e}")
        return None, None

LOCAL_MODEL, LOCAL_FE = load_local_model()

# --- Sidebar ---
st.sidebar.title("💳 MLOps Hub")
menu = st.sidebar.selectbox("Navigate", 
    ["Dashboard Overview", "Pipeline Control", "Real-time Prediction", "Drift Monitoring"])

st.sidebar.markdown("---")
st.sidebar.info("This dashboard integrates the end-to-end MLOps workflow for Credit Card Fraud Detection.")

# --- Dashboard Overview ---
if menu == "Dashboard Overview":
    st.title("🚀 Credit Card Fraud Detection MLOps")
    
    col1, col2, col3 = st.columns(3)
    
    # Load Evaluation Metrics if they exist
    if os.path.exists("metrics/evaluation.json"):
        with open("metrics/evaluation.json", "r") as f:
            metrics_data = json.load(f)
        
        # Display Core Metrics (from classification_report)
        report = metrics_data.get("classification_report", {})
        accuracy = report.get("accuracy", 0.999)
        fraud_recall = report.get("1", {}).get("recall", 0.85)
        normal_f1 = report.get("0", {}).get("f1-score", 0.99)
        
        col1.metric("Model Accuracy", f"{accuracy*100:.2f}%")
        col2.metric("Fraud Recall", f"{fraud_recall*100:.1f}%")
        col3.metric("Normal F1-Score", f"{normal_f1*100:.1f}%")
        
        st.markdown("### 📊 Ensemble Model Performance (Detailed)")
        
        # Simplified Comparison view for current architecture
        df_metrics = pd.DataFrame({
            "Metric": ["Accuracy", "Fraud Recall", "Fraud Precision", "ROC AUC"],
            "Value": [
                accuracy, 
                fraud_recall, 
                report.get("1", {}).get("precision", 0),
                metrics_data.get("roc_auc", 0)
            ]
        })
        
        fig = px.bar(df_metrics, x="Metric", y="Value", 
                     title="Model Performance Overview",
                     range_y=[0, 1.1],
                     template="plotly_dark",
                     color="Metric")
        st.plotly_chart(fig, use_container_width=True)

        # Confusion Matrix Visualization
        cm = metrics_data.get("confusion_matrix", [[0,0],[0,0]])
        st.markdown("### 📍 Confusion Matrix")
        fig_cm = px.imshow(cm, 
                          labels=dict(x="Predicted", y="Actual", color="Count"),
                          x=['Legitimate', 'Fraud'],
                          y=['Legitimate', 'Fraud'],
                          text_auto=True,
                          aspect="auto",
                          template="plotly_dark")
        st.plotly_chart(fig_cm, use_container_width=True)

    else:
        st.warning("No metrics found. Please run the training pipeline first.")

# --- Pipeline Control ---
elif menu == "Pipeline Control":
    st.title("🛠️ Pipeline Execution Control")
    st.write("Trigger each stage of the MLOps lifecycle from here.")
    
    c1, c2, c3 = st.columns(3)
    
    if c1.button("📥 Run Data Ingestion"):
        out = run_pipeline_step("src/data_ingest.py")
        if out: st.text_area("Ingestion Logs", out, height=200)
        
    if c2.button("🚂 Run Training Pipeline"):
        out = run_pipeline_step("src/train_pipeline.py")
        if out: st.text_area("Training Logs", out, height=400)
        
    if c3.button("🧪 Run Test Pipeline"):
        out = run_pipeline_step("src/test_pipeline.py")
        if out: st.text_area("Test Logs", out, height=200)

# --- Real-time Prediction ---
elif menu == "Real-time Prediction":
    st.title("🔍 Real-time Inference Playground")
    st.write("Enter transaction features to check for fraud probability.")
    
    # Define features even if sample data is missing
    features_list = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 
                     'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 
                     'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 
                     'V28', 'Amount']
    
    # Try to load a sample if file exists, else use defaults
    default_vals = {feat: 0.0 for feat in features_list}
    if os.path.exists("data/processed/test.csv"):
        try:
            test_df = pd.read_csv("data/processed/test.csv")
            sample = test_df.iloc[0].drop("Class")
            for feat in features_list:
                 if feat in sample: default_vals[feat] = float(sample[feat])
            st.info("📊 Loaded initial values from test dataset.")
        except:
            pass
    
    with st.form("inference_form"):
        col1, col2 = st.columns(2)
        inputs = {}
        
        for i, feat in enumerate(features_list):
            with col1 if i < len(features_list)/2 else col2:
                inputs[feat] = st.number_input(feat, value=default_vals[feat], format="%.5f")
        
        submitted = st.form_submit_button("Detect Fraud")
        
        if submitted:
                # Try API first, fallback to Local
                prediction_success = False
                try:
                    # Match Fast API expectations
                    response = requests.post(API_URL, json=[inputs], timeout=2)
                    if response.status_code == 200:
                        pred_data = response.json()[0]
                        prob = pred_data['probability']
                        is_fraud = pred_data['prediction'] == 1
                        prediction_success = True
                except:
                    # Local Fallback
                    if LOCAL_MODEL and LOCAL_FE:
                        df_input = pd.DataFrame([inputs])
                        
                        # Ensure columns are in the right order for the Feature Engineer
                        expected_base_cols = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 
                                              'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 
                                              'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 
                                              'V28', 'Amount']
                        df_input = df_input[expected_base_cols]
                        
                        # Apply transformations
                        df_transformed = LOCAL_FE.transform(df_input)
                        
                        # Ensure columns match the model's training order exactly
                        if hasattr(LOCAL_MODEL, 'feature_names_in_'):
                            df_transformed = df_transformed[LOCAL_MODEL.feature_names_in_]
                        
                        # Use .values to bypass name-based validation errors on different environments
                        probs = LOCAL_MODEL.predict_proba(df_transformed.values)
                        prob = probs[0][1]
                        
                        # Prediction successfully calculated
                        is_fraud = LOCAL_MODEL.predict(df_transformed.values)[0] == 1
                        prediction_success = True
                        st.info("💡 Prediction served via local model fallback.")
                
                if prediction_success:
                    if is_fraud:
                        st.error(f"🚨 FRAUD DETECTED! (Probability: {prob*100:.2f}%)")
                    else:
                        st.success(f"✅ Transaction Legitimate (Probability: {prob*100:.2f}%)")
                        
                    # Probability gauge
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = prob * 100,
                        title = {'text': "Fraud Risk (%)"},
                        gauge = {'axis': {'range': [None, 100]},
                                 'bar': {'color': "#ff4b4b" if is_fraud else "#2eeb71"},
                                 'steps': [
                                     {'range': [0, 30], 'color': "lightgreen"},
                                     {'range': [30, 70], 'color': "yellow"},
                                     {'range': [70, 100], 'color': "red"}]}
                    ))
                    fig.update_layout(template="plotly_dark", height=300)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("Prediction failed. Neither live API nor local model are available.")

# --- Drift Monitoring ---
elif menu == "Drift Monitoring":
    st.title("📉 Model & Data Drift Surveillance")
    
    if st.button("🔍 Run Drift Analysis"):
        out = run_pipeline_step("src/monitoring.py")
        if out:
            st.subheader("Monitoring Report")
            st.code(out)
            
            if "DRIFT DETECTED" in out:
                st.warning("🚨 Model Retraining Recommended due to detected drift.")
                if st.button("🔥 Trigger Automatic Retraining"):
                    run_pipeline_step("src/retrain.py")
            else:
                st.success("✅ No significant drift detected. Model performance remains stable.")

st.markdown("---")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
