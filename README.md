# CardShield – Credit Card Fraud Detection System

An AI/ML project that predicts potentially fraudulent credit card transactions using a trained machine learning model. It includes an interactive Streamlit dashboard and a FastAPI backend with prediction history stored in SQLite.

## Features

- Machine learning-based fraud prediction
- Interactive Streamlit dashboard
- Transaction CSV upload and screening
- Fraud analytics and model performance comparison
- FastAPI REST API for transaction predictions
- Single and batch prediction endpoints
- SQLite database for prediction history
- API health and model feature endpoints
- Demo transaction dataset for testing

## Tech Stack

- Python
- Pandas and NumPy
- Scikit-learn
- Streamlit
- FastAPI and Uvicorn
- SQLite
- Matplotlib and Seaborn

## Project Files

- `app.py` — Streamlit dashboard
- `api.py` — FastAPI backend and prediction history
- `fraud_model.pkl` — trained machine learning model
- `train_model.py` — model training script
- `demo_transactions.csv` — sample transaction data
- `make_demo_data.py` — demo dataset generator
- `model_comparison.csv` — model comparison results
- `requirements.txt` — Python dependencies
- `tests/` — automated API tests (add after creating the tests)

## Run Locally

### 1. Clone or download the repository

Open the project folder in VS Code.

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment (Windows PowerShell)

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 5. Start the Streamlit dashboard

```bash
python -m streamlit run app.py
```

### 6. Start the FastAPI backend

Open a second terminal in the project folder and run:

```bash
python -m uvicorn api:app --reload --port 8000
```

API documentation: http://127.0.0.1:8000/docs

Health check: http://127.0.0.1:8000/health

## Dataset

The original `creditcard.csv` dataset is not included in this repository. Some historical analytics or model-training operations may require the original dataset.

Use only appropriate sample data when testing the application. Do not upload confidential cardholder information.

## Disclaimer

This project is intended for educational and demonstration purposes. A model prediction is a risk alert, not proof of fraud, and this project is not a production banking or payment security system.
