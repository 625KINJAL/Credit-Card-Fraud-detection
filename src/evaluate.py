"""
Model evaluation module: computes Accuracy, Precision, Recall, F1-score,
ROC-AUC, Confusion Matrices, and saves comparison visualizations.
"""

import json
import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

MODELS_DIR = "models"
OUTPUT_DIR = "outputs"


def evaluate_all_models(
    models: dict, X_test, y_test, output_dir: str = OUTPUT_DIR
) -> dict:
    """Evaluate trained models and generate confusion matrix & comparison plots."""
    os.makedirs(output_dir, exist_ok=True)
    results = {}
    confusion_matrices = {}

    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        cm = confusion_matrix(y_test, y_pred)

        metrics = {
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
            "f1_score": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4),
            "confusion_matrix": cm.tolist(),
        }
        results[name] = metrics
        confusion_matrices[name] = cm

        print(f"\n=== {name} ===")
        print(
            f"Accuracy : {metrics['accuracy']:.4f} | "
            f"Precision: {metrics['precision']:.4f} | "
            f"Recall   : {metrics['recall']:.4f} | "
            f"F1-Score : {metrics['f1_score']:.4f} | "
            f"ROC-AUC  : {metrics['roc_auc']:.4f}"
        )
        print(f"Confusion Matrix:\n{cm}")

    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # 1. Confusion Matrices Plot
    fig, axes = plt.subplots(1, len(confusion_matrices), figsize=(11, 4.5))
    if len(confusion_matrices) == 1:
        axes = [axes]

    for ax, (name, cm) in zip(axes, confusion_matrices.items()):
        sns.heatmap(
            cm,
            annot=True,
            fmt=",d",
            cmap="Blues",
            cbar=False,
            xticklabels=["Legitimate (0)", "Fraud (1)"],
            yticklabels=["Legitimate (0)", "Fraud (1)"],
            ax=ax,
        )
        ax.set_title(f"{name}\nConfusion Matrix", fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("Actual Label")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrices.png"), dpi=200)
    plt.close(fig)

    # 2. Model Comparison Bar Chart
    metric_keys = ["accuracy", "precision", "recall", "f1_score"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score"]
    model_names = list(results.keys())

    x = np.arange(len(metric_labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    colors = ["#1f77b4", "#2ca02c"]
    for idx, model_name in enumerate(model_names):
        vals = [results[model_name][k] for k in metric_keys]
        offset = (idx - (len(model_names) - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=model_name, color=colors[idx % 2])
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.015,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    ax.set_ylim(0, 1.12)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison (Test Set)", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_comparison.png"), dpi=200)
    plt.close(fig)

    print(f"\n[Evaluate] Saved metrics and charts to '{output_dir}/'.")
    return results


if __name__ == "__main__":
    lr = joblib.load(os.path.join(MODELS_DIR, "logistic_regression.joblib"))
    rf = joblib.load(os.path.join(MODELS_DIR, "random_forest.joblib"))
    X_test_saved, y_test_saved = joblib.load(os.path.join(MODELS_DIR, "test_split.joblib"))
    evaluate_all_models(
        {"Logistic Regression": lr, "Random Forest": rf},
        X_test_saved,
        y_test_saved,
    )
