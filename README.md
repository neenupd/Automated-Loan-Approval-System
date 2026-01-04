# Automated Loan Approval System

A machine learning-based framework for automated, data-driven loan approval decisions using Lending Club loan data.

## Project Overview

This project develops a comprehensive loan approval system that leverages machine learning to predict loan eligibility and automate approval processes. The system integrates data preprocessing, feature engineering, and supervised learning techniques to provide instant, accurate, and fair loan decisions.

## Key Features

- **Multiple ML Models**: Logistic Regression, Random Forest, XGBoost, and LightGBM
- **Comprehensive Evaluation**: AUC-ROC, precision-recall, Brier score, and business metrics
- **Real-time Decision Engine**: Instant approval/rejection based on calibrated probability scores
- **Explainability**: SHAP values for transparent decision-making
- **Fairness Audits**: Bias assessment across demographic groups
- **Feature Engineering**: Debt-to-income ratios, credit utilization, and other meaningful financial indicators

## Project Structure

```
Automated-Loan-Approval-System/
├── data/
│   ├── raw/              # Raw Lending Club data
│   └── processed/        # Processed and feature-engineered data
├── src/
│   ├── data_preprocessing.py    # Data cleaning and preprocessing
│   ├── feature_engineering.py   # Feature creation and transformation
│   ├── model_training.py        # Model training and calibration
│   ├── evaluation.py            # Model evaluation metrics
│   ├── decision_engine.py       # Real-time decision engine
│   ├── explainability.py        # SHAP-based explainability
│   └── fairness_audit.py        # Fairness and bias assessment
├── models/               # Saved trained models
├── notebooks/            # Jupyter notebooks for exploration
├── tests/                # Unit tests
├── requirements.txt      # Python dependencies
└── main.py              # Main training script
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Automated-Loan-Approval-System
```

2. Install dependencies:

**Windows:**
```bash
python -m pip install -r requirements.txt
```

**Linux/Mac:**
```bash
pip install -r requirements.txt
```

3. Download the Lending Club dataset from Kaggle:
   - Visit: https://www.kaggle.com/datasets/wordsforthewise/lending-club
   - Place the data in `data/raw/` directory

## Usage

### Training Models

```bash
python main.py
```

### Making Predictions

**Simple Command-Line Interface (Recommended):**

The easiest way to make predictions is using `predict_simple.py`:

```bash
python predict_simple.py [annual_income] [loan_amount] [dti] [fico_low] [fico_high]
```

**Examples:**
```bash
# Test with default values
python predict_simple.py

# Test with custom values
python predict_simple.py 120000 8000 10 780 790
python predict_simple.py 75000 15000 15.5 720 724
```

**Parameters:**
- `annual_income`: Annual income in dollars (default: 75000)
- `loan_amount`: Loan amount requested in dollars (default: 15000)
- `dti`: Debt-to-income ratio as percentage (default: 15.5)
- `fico_low`: FICO score low range (default: 720)
- `fico_high`: FICO score high range (default: 724)

**Output:**
```
Decision: APPROVED/REJECTED
Approval Probability: X.XX%
Confidence Level: X.XX%
```

### Using the Decision Engine Programmatically

For advanced usage, you can use the decision engine directly:

```python
from src.decision_engine import LoanDecisionEngine
import joblib
import pandas as pd
import numpy as np

# Load trained model and components
model = joblib.load('models/best_model_logistic_regression.pkl')
scaler = joblib.load('models/scaler.pkl')
feature_names = joblib.load('models/feature_names.pkl')

# Load processed data (example)
df = pd.read_csv('data/processed/features_engineered.csv', nrows=1)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
matching_cols = [col for col in numeric_cols if col in feature_names]
selected_features = matching_cols[:scaler.n_features_in_]

# Initialize decision engine
engine = LoanDecisionEngine(
    model=model,
    scaler=scaler,
    threshold=0.5,
    feature_names=selected_features
)

# Make a prediction
applicant_data = df[selected_features].iloc[0].to_dict()
decision = engine.make_decision(applicant_data)

print(f"Decision: {decision['decision']}")
print(f"Probability: {decision['probability']:.4f}")
```

See `example_usage.py` for more comprehensive examples.

## Dataset

This project uses the **All Lending Club Loan Data** from Kaggle, which contains millions of historical loan records with diverse borrower and loan attributes.

**Important**: Only features available at the time of loan application are used to avoid target leakage.

## Models

- **Logistic Regression**: Baseline interpretable model
- **Random Forest**: Robust ensemble method with feature importance
- **XGBoost**: State-of-the-art gradient boosting for high accuracy
- **LightGBM**: Fast gradient boosting with excellent performance

## Evaluation Metrics

- **Discrimination**: AUC-ROC, Precision-Recall
- **Calibration**: Brier Score
- **Business Metrics**: Approval Rate, Expected Loss
- **Fairness**: Demographic parity, equalized odds

## Model Selection

The best model is selected based on a combination of:
- AUC-ROC score (primary)
- Precision-Recall AUC
- Brier Score
- Business impact metrics

## Explainability

The system provides SHAP-based explanations for each decision, generating:
- Feature importance rankings
- Individual prediction explanations
- Reason codes for approvals/rejections

## Fairness & Compliance

The system includes fairness audits to assess:
- Demographic parity across protected groups
- Equalized odds
- Disparate impact analysis

## Quick Start Examples

```bash
# Make a prediction (easiest way)
python predict_simple.py

# Test with custom values
python predict_simple.py 120000 8000 10 780 790

# See comprehensive usage examples
python example_usage.py

# Train models from scratch
python main.py
```

## License

This project is for educational and research purposes.

## Acknowledgments

- Lending Club for providing the public dataset
- Kaggle for hosting the data

