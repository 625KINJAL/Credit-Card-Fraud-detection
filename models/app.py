
"""
Streamlit Dashboard for Credit Card Fraud Detection.
Run with: streamlit run app.py
"""

import json
import os
import sys
import subprocess


import pandas as pd
import streamlit as st

from src.predict import FEATURE_COLUMNS, predict_transaction


# ==============================
# PROJECT PATHS
# ==============================

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs")
DATA_PATH = os.path.join(ROOT_DIR, "data", "creditcard.csv")

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="🛡️",
    layout="wide",
)


# ==============================
# AUTOMATIC TRAINING
# ==============================

def ensure_project_outputs():

    required_files = [
        "eda_summary.json",
        "class_distribution.png",
        "fraud_vs_legitimate_amount.png",
        "metrics.json",
        "model_comparison.png",
        "confusion_matrices.png",
    ]

    missing_files = [
        file for file in required_files
        if not os.path.exists(os.path.join(OUTPUT_DIR, file))
    ]

    if missing_files:

        st.info("Generating model and EDA results automatically...")

        if not os.path.exists(DATA_PATH):
            st.error(
                "Dataset not found! Please ensure "
                "data/creditcard.csv is uploaded to GitHub."
            )
            return False

        try:

            with st.spinner(
                "Training machine learning models. Please wait..."
            ):

                result = subprocess.run(
                    [sys.executable, "-m", "src.train"],
                    cwd=ROOT_DIR,
                    capture_output=True,
                    text=True,
                    timeout=600,
                )

            if result.returncode != 0:

                st.error("Model training failed.")

                if result.stderr:
                    st.code(result.stderr)

                if result.stdout:
                    st.code(result.stdout)

                return False

            st.success("Model training and EDA completed successfully!")

        except subprocess.TimeoutExpired:
            st.error(
                "Training took too long. "
                "Please check your training script."
            )
            return False

        except Exception as e:
            st.error(f"Training error: {e}")
            return False

    return True


# Run automatic training if required
training_success = ensure_project_outputs()


# ==============================
# DASHBOARD HEADER
# ==============================

st.title("🛡️ Credit Card Fraud Detection Dashboard")

st.markdown(
    "A beginner-friendly Machine Learning app using "
    "**Logistic Regression** and **Random Forest** with "
    "`class_weight='balanced'` on the Kaggle Credit Card Fraud dataset."
)


# ==============================
# SIDEBAR NAVIGATION
# ==============================

page = st.sidebar.radio(
    "Navigation",
    [
        "1. Dataset & EDA",
        "2. Model Evaluation",
        "3. Live Fraud Predictor",
    ],
)


# ==============================
# HELPER FUNCTIONS
# ==============================

def load_json(path):

    if os.path.exists(path):

        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    return None


def output_image(filename):

    return os.path.join(OUTPUT_DIR, filename)


# ==============================
# PAGE 1: DATASET & EDA
# ==============================

if page == "1. Dataset & EDA":

    st.header("Exploratory Data Analysis (EDA)")

    eda = load_json(
        output_image("eda_summary.json")
    )

    if eda is None:

        st.warning(
            "EDA results could not be generated. "
            "Please check your training script and dataset."
        )

    else:

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Total Transactions",
            f"{eda['total_transactions']:,}"
        )

        col2.metric(
            "Legitimate Cases",
            f"{eda['legitimate_count']:,}"
        )

        col3.metric(
            "Fraudulent Cases",
            f"{eda['fraud_count']:,}"
        )

        col4.metric(
            "Fraud Rate",
            f"{eda['fraud_percentage']}%"
        )

        st.subheader("Visualizations")

        c1, c2 = st.columns(2)

        with c1:

            image_path = output_image(
                "class_distribution.png"
            )

            if os.path.exists(image_path):

                st.image(
                    image_path,
                    caption="Class Distribution (Log Scale)",
                    use_container_width=True,
                )

        with c2:

            image_path = output_image(
                "fraud_vs_legitimate_amount.png"
            )

            if os.path.exists(image_path):

                st.image(
                    image_path,
                    caption="Transaction Amount: Legitimate vs. Fraudulent",
                    use_container_width=True,
                )


# ==============================
# PAGE 2: MODEL EVALUATION
# ==============================

elif page == "2. Model Evaluation":

    st.header("Model Performance & Comparison")

    metrics = load_json(
        output_image("metrics.json")
    )

    if metrics is None:

        st.warning(
            "Model metrics could not be generated. "
            "Please check your training script."
        )

    else:

        rows = []

        for model_name, vals in metrics.items():

            rows.append(
                {
                    "Model": model_name,
                    "Accuracy": vals["accuracy"],
                    "Precision": vals["precision"],
                    "Recall": vals["recall"],
                    "F1-Score": vals["f1_score"],
                    "ROC-AUC": vals["roc_auc"],
                }
            )

        st.subheader("Evaluation Metrics Table (Test Set)")

        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Comparison Charts")

        c1, c2 = st.columns(2)

        with c1:

            image_path = output_image(
                "model_comparison.png"
            )

            if os.path.exists(image_path):

                st.image(
                    image_path,
                    caption="Logistic Regression vs. Random Forest",
                    use_container_width=True,
                )

        with c2:

            image_path = output_image(
                "confusion_matrices.png"
            )

            if os.path.exists(image_path):

                st.image(
                    image_path,
                    caption="Confusion Matrices",
                    use_container_width=True,
                )


# ==============================
# PAGE 3: LIVE FRAUD PREDICTOR
# ==============================

elif page == "3. Live Fraud Predictor":

    st.header("Predict a Transaction")

    model_label = st.selectbox(
        "Choose Model",
        [
            "Random Forest",
            "Logistic Regression",
        ],
    )

    model_key = (
        "random_forest"
        if model_label == "Random Forest"
        else "logistic_regression"
    )

    # Initialize session state

    if "tx_features" not in st.session_state:

        st.session_state.tx_features = {
            col: 0.0 for col in FEATURE_COLUMNS
        }

    # Sample transaction buttons

    if os.path.exists(DATA_PATH):

        st.markdown(
            "**Quick Test:** Load a real sample row from the dataset."
        )

        b1, b2 = st.columns(2)

        if b1.button("Load Random Legitimate Transaction"):

            df_sample = pd.read_csv(DATA_PATH)

            row = (
                df_sample[df_sample["Class"] == 0]
                .sample(1)
                .iloc[0]
            )

            st.session_state.tx_features = {
                col: float(row[col])
                for col in FEATURE_COLUMNS
            }

        if b2.button("Load Random Fraudulent Transaction"):

            df_sample = pd.read_csv(DATA_PATH)

            row = (
                df_sample[df_sample["Class"] == 1]
                .sample(1)
                .iloc[0]
            )

            st.session_state.tx_features = {
                col: float(row[col])
                for col in FEATURE_COLUMNS
            }

    # Prediction form

    with st.form("prediction_form"):

        c1, c2 = st.columns(2)

        time_val = c1.number_input(
            "Time (seconds)",
            value=float(
                st.session_state.tx_features["Time"]
            ),
        )

        amount_val = c2.number_input(
            "Amount ($)",
            min_value=0.0,
            value=float(
                st.session_state.tx_features["Amount"]
            ),
        )

        with st.expander(
            "PCA Features (V1 - V28)",
            expanded=False,
        ):

            cols = st.columns(4)

            v_vals = {}

            for idx in range(1, 29):

                col_name = f"V{idx}"

                v_vals[col_name] = cols[
                    (idx - 1) % 4
                ].number_input(
                    col_name,
                    value=float(
                        st.session_state.tx_features[col_name]
                    ),
                    format="%.4f",
                )

        submitted = st.form_submit_button(
            "🔍 Predict Transaction"
        )

    # Prediction result

    if submitted:

        features = {
            "Time": time_val,
            "Amount": amount_val,
            **v_vals,
        }

        try:

            res = predict_transaction(
                features,
                model_choice=model_key,
            )

            prob_pct = (
                res["fraud_probability"] * 100
            )

            if res["prediction"] == 1:

                st.error(
                    "⚠️ **FRAUDULENT TRANSACTION DETECTED** "
                    f"— Fraud Probability: **{prob_pct:.2f}%**"
                )

            else:

                st.success(
                    "✅ **LEGITIMATE TRANSACTION** "
                    f"— Fraud Probability: **{prob_pct:.2f}%**"
                )

        except FileNotFoundError as err:

            st.error(
                f"Model file not found: {err}"
            )

        except Exception as err:

            st.error(
                f"Prediction error: {err}"
            )
