"""
Prediction helper and CLI demo for Credit Card Fraud Detection.
"""

import os
import joblib
import pandas as pd

MODELS_DIR = "models"
DATA_PATH = os.path.join("data", "creditcard.csv")
OUTPUT_DIR = "outputs"

FEATURE_COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]


def load_artifacts(model_choice: str = "random_forest", models_dir: str = MODELS_DIR):
    """Load the saved StandardScaler and chosen model."""
    scaler_path = os.path.join(models_dir, "scaler.joblib")
    model_file = (
        "random_forest.joblib"
        if model_choice == "random_forest"
        else "logistic_regression.joblib"
    )
    model_path = os.path.join(models_dir, model_file)

    if not os.path.exists(scaler_path) or not os.path.exists(model_path):
        raise FileNotFoundError(
            "Trained model files not found. Please run 'python -m src.train' first."
        )

    scaler = joblib.load(scaler_path)
    model = joblib.load(model_path)
    return scaler, model


def predict_transaction(features: dict, model_choice: str = "random_forest") -> dict:
    """Predict whether a single transaction dictionary is Fraudulent or Legitimate."""
    scaler, model = load_artifacts(model_choice=model_choice)

    row = {col: float(features.get(col, 0.0)) for col in FEATURE_COLUMNS}
    input_df = pd.DataFrame([row], columns=FEATURE_COLUMNS)
    input_df[["Time", "Amount"]] = scaler.transform(input_df[["Time", "Amount"]])

    pred = int(model.predict(input_df)[0])
    fraud_prob = float(model.predict_proba(input_df)[0][1])

    return {
        "prediction": pred,
        "label": "Fraudulent" if pred == 1 else "Legitimate",
        "fraud_probability": round(fraud_prob, 4),
    }


def run_sample_demo():
    """Run a sample prediction on one legitimate and one fraudulent row from creditcard.csv."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"'{DATA_PATH}' not found. Download creditcard.csv from Kaggle first."
        )

    df = pd.read_csv(DATA_PATH)
    legit_sample = df[df["Class"] == 0].iloc[0].drop("Class").to_dict()
    fraud_sample = df[df["Class"] == 1].iloc[0].drop("Class").to_dict()

    lines = ["=== Sample Prediction Output (Random Forest) ==="]
    for title, sample in [("Legitimate Sample", legit_sample), ("Fraudulent Sample", fraud_sample)]:
        result = predict_transaction(sample, model_choice="random_forest")
        line = (
            f"{title} -> Predicted: {result['label']} (Class {result['prediction']}) "
            f"| Fraud Probability: {result['fraud_probability'] * 100:.2f}% "
            f"| Amount: ${sample['Amount']:.2f}"
        )
        print(line)
        lines.append(line)

    out_path = os.path.join(OUTPUT_DIR, "sample_prediction.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n[Predict] Saved sample output to '{out_path}'.")


if __name__ == "__main__":
    run_sample_demo()
