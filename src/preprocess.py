"""
Data loading, Exploratory Data Analysis (EDA), and preprocessing
for the Kaggle Credit Card Fraud Detection dataset.
"""

import json
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_PATH = os.path.join("data", "creditcard.csv")
OUTPUT_DIR = "outputs"


def load_dataset(csv_path: str = DATA_PATH) -> pd.DataFrame:
    """Load the Kaggle creditcard.csv dataset."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Dataset not found at '{csv_path}'.\n"
            "Please download 'creditcard.csv' from Kaggle "
            "(https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) "
            "and place it inside the 'data/' folder."
        )
    df = pd.read_csv(csv_path)
    return df


def run_eda(df: pd.DataFrame, output_dir: str = OUTPUT_DIR) -> dict:
    """Compute summary statistics and save EDA visualizations."""
    os.makedirs(output_dir, exist_ok=True)

    total_rows = int(len(df))
    missing_values = int(df.isnull().sum().sum())
    class_counts = df["Class"].value_counts().to_dict()
    legit_count = int(class_counts.get(0, 0))
    fraud_count = int(class_counts.get(1, 0))
    fraud_pct = round((fraud_count / total_rows) * 100, 4) if total_rows else 0.0

    eda_summary = {
        "total_transactions": total_rows,
        "num_features": int(df.shape[1] - 1),
        "missing_values": missing_values,
        "legitimate_count": legit_count,
        "fraud_count": fraud_count,
        "fraud_percentage": fraud_pct,
        "legit_mean_amount": round(float(df[df["Class"] == 0]["Amount"].mean()), 2),
        "fraud_mean_amount": round(float(df[df["Class"] == 1]["Amount"].mean()), 2),
        "legit_median_amount": round(float(df[df["Class"] == 0]["Amount"].median()), 2),
        "fraud_median_amount": round(float(df[df["Class"] == 1]["Amount"].median()), 2),
    }

    with open(os.path.join(output_dir, "eda_summary.json"), "w", encoding="utf-8") as f:
        json.dump(eda_summary, f, indent=2)

    # 1. Class Distribution Plot
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    labels = ["Legitimate (0)", "Fraudulent (1)"]
    counts = [legit_count, fraud_count]
    colors = ["#1f77b4", "#d62728"]

    bars = ax.bar(labels, counts, color=colors, width=0.5)
    ax.set_yscale("log")
    ax.set_title("Class Distribution (Log Scale)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Number of Transactions (Log Scale)")

    for bar, count in zip(bars, counts):
        pct = (count / total_rows) * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.15,
            f"{count:,}\n({pct:.3f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_ylim(1, max(counts) * 3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "class_distribution.png"), dpi=200)
    plt.close(fig)

    # 2. Fraud vs Legitimate Transaction Amount Plot
    fig, ax = plt.subplots(figsize=(7, 4.5))
    plot_df = df[["Amount", "Class"]].copy()
    plot_df["Transaction Type"] = plot_df["Class"].map(
        {0: "Legitimate", 1: "Fraudulent"}
    )
    sns.boxplot(
        data=plot_df,
        x="Transaction Type",
        y="Amount",
        hue="Transaction Type",
        palette={"Legitimate": "#1f77b4", "Fraudulent": "#d62728"},
        legend=False,
        showfliers=False,
        ax=ax,
    )
    ax.set_title(
        "Transaction Amount: Legitimate vs. Fraudulent (Outliers Hidden)",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_ylabel("Transaction Amount")
    ax.set_xlabel("")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fraud_vs_legitimate_amount.png"), dpi=200)
    plt.close(fig)

    print(f"[EDA] Total transactions: {total_rows:,} | Fraud: {fraud_count:,} ({fraud_pct}%)")
    print(f"[EDA] Missing values: {missing_values}")
    print(f"[EDA] Saved plots and summary to '{output_dir}/'.")
    return eda_summary


def preprocess_data(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
):
    """
    Split dataset into stratified train/test sets and scale 'Time' and 'Amount'.
    Scaling is fit ONLY on the training set to prevent data leakage.
    """
    X = df.drop(columns=["Class"]).copy()
    y = df["Class"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    scaler = StandardScaler()
    cols_to_scale = ["Time", "Amount"]
    X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
    X_test[cols_to_scale] = scaler.transform(X_test[cols_to_scale])

    print(
        f"[Preprocess] Train shape: {X_train.shape} | Test shape: {X_test.shape} "
        f"| Test fraud cases: {int(y_test.sum())}"
    )
    return X_train, X_test, y_train, y_test, scaler


if __name__ == "__main__":
    dataframe = load_dataset()
    run_eda(dataframe)
    preprocess_data(dataframe)
