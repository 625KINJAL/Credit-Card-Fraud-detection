# Credit Card Fraud Detection Using Machine Learning

A beginner-friendly, end-to-end Machine Learning mini-project that detects fraudulent credit card transactions using **Logistic Regression** and **Random Forest** with `class_weight="balanced"`, plus an interactive **Streamlit** dashboard.

## Project Structure

```text
credit_card_fraud_detection/
├── data/
│   └── creditcard.csv          # Download from Kaggle (see below)
├── models/                     # Saved .joblib models and StandardScaler
├── outputs/                    # Generated EDA & evaluation charts + metrics.json
├── src/
│   ├── __init__.py
│   ├── preprocess.py           # Data loading, EDA plots, StandardScaler, 80/20 split
│   ├── train.py                # Train Logistic Regression & Random Forest
│   ├── evaluate.py             # Accuracy, Precision, Recall, F1, Confusion Matrix
│   └── predict.py              # CLI prediction demo & helper
├── app.py                      # Streamlit web dashboard
├── requirements.txt            # Python dependencies
└── README.md
```

## Step 1: Download the Dataset from Kaggle

1. Go to the [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
2. Sign in to Kaggle and click **Download** to get `archive.zip`.
3. Extract `creditcard.csv` from the ZIP file.
4. Move `creditcard.csv` into the `data/` folder inside this project (`credit_card_fraud_detection/data/creditcard.csv`).

## Step 2: Setup in VS Code on Windows

Open the `credit_card_fraud_detection` folder in **VS Code**, open the **Terminal** (`Ctrl + ~`), and run:

```powershell
# 1. Create a virtual environment
python -m venv venv

# 2. Activate the virtual environment (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# (If PowerShell blocks scripts, run this once and retry:)
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# 3. Install required packages
pip install -r requirements.txt
```

## Step 3: Train Models & Generate Graphs

Run the training pipeline (this performs EDA, preprocessing, model training, and evaluation):

```powershell
python -m src.train
```

This creates:
- Trained models in `models/`
- Metrics (`outputs/metrics.json`) and EDA/evaluation charts in `outputs/`:
  - `class_distribution.png`
  - `fraud_vs_legitimate_amount.png`
  - `confusion_matrices.png`
  - `model_comparison.png`

## Step 4: Run a Sample CLI Prediction

```powershell
python -m src.predict
```

## Step 5: Launch the Streamlit Dashboard

```powershell
streamlit run app.py
```

Your browser will open automatically at `http://localhost:8501`.
