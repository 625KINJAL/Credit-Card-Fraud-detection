"""
Train Logistic Regression and Random Forest models with class_weight='balanced'
to handle the severe class imbalance in Credit Card Fraud Detection.
"""

import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.evaluate import evaluate_all_models
from src.preprocess import load_dataset, preprocess_data, run_eda

MODELS_DIR = "models"


def train_models(X_train, y_train, models_dir: str = MODELS_DIR) -> dict:
    """Train Logistic Regression and Random Forest with balanced class weights."""
    os.makedirs(models_dir, exist_ok=True)

    print("\n[Train] Training Logistic Regression (class_weight='balanced')...")
    lr_model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
    )
    lr_model.fit(X_train, y_train)
    joblib.dump(lr_model, os.path.join(models_dir, "logistic_regression.joblib"))

    print("[Train] Training Random Forest (n_estimators=100, class_weight='balanced')...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf_model.fit(X_train, y_train)
    joblib.dump(rf_model, os.path.join(models_dir, "random_forest.joblib"))

    print(f"[Train] Models saved to '{models_dir}/'.")
    return {
        "Logistic Regression": lr_model,
        "Random Forest": rf_model,
    }


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    df = load_dataset()
    run_eda(df)

    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.joblib"))
    joblib.dump((X_test, y_test), os.path.join(MODELS_DIR, "test_split.joblib"))

    models = train_models(X_train, y_train, models_dir=MODELS_DIR)
    evaluate_all_models(models, X_test, y_test)
    print("\n[Done] Training and evaluation complete!")


if __name__ == "__main__":
    main()
