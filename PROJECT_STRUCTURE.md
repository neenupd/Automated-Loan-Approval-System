# Project Structure

## Overview

This document describes the structure and components of the Automated Loan Approval System.

## Directory Structure

```
Automated-Loan-Approval-System/
│
├── data/                          # Data directory
│   ├── raw/                       # Raw Lending Club dataset (download from Kaggle)
│   │   └── .gitkeep
│   └── processed/                 # Processed and feature-engineered data
│       └── .gitkeep
│
├── src/                           # Source code modules
│   ├── __init__.py                # Package initialization
│   ├── data_preprocessing.py      # Data cleaning and preprocessing
│   ├── feature_engineering.py     # Feature creation and transformation
│   ├── model_training.py          # Model training and calibration
│   ├── evaluation.py              # Model evaluation metrics
│   ├── decision_engine.py         # Real-time decision engine
│   ├── explainability.py          # SHAP-based model explanations
│   └── fairness_audit.py          # Fairness and bias assessment
│
├── models/                        # Trained models and artifacts
│   └── .gitkeep
│
├── notebooks/                     # Jupyter notebooks for exploration
│
├── tests/                         # Unit tests
│
├── main.py                        # Main training pipeline script
├── example_usage.py               # Example usage demonstrations
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
├── SETUP.md                       # Setup instructions
└── .gitignore                     # Git ignore rules
```

## Module Descriptions

### 1. data_preprocessing.py

**Purpose:** Handles data loading, cleaning, and initial preprocessing.

**Key Components:**
- `DataPreprocessor` class
- Methods for handling missing values
- Categorical variable encoding
- Removal of post-origination features (to avoid target leakage)
- Target variable creation (binary: eligible/ineligible)

**Key Methods:**
- `load_data()`: Load CSV data
- `remove_post_origination_features()`: Remove features not available at application time
- `create_target_variable()`: Convert loan_status to binary target
- `handle_missing_values()`: Impute or remove missing values
- `encode_categorical_variables()`: Label encoding for categorical features
- `preprocess()`: Complete preprocessing pipeline

### 2. feature_engineering.py

**Purpose:** Creates meaningful financial features and ratios.

**Key Components:**
- `FeatureEngineer` class
- Financial ratio calculations
- Credit history features

**Engineered Features:**
- `loan_to_income`: Loan amount to annual income ratio
- `credit_utilization`: Revolving credit utilization percentage
- `credit_history_age`: Years since earliest credit line
- `fico_avg`: Average FICO score
- `loan_term_months`: Loan term in months
- `emp_length_years`: Employment length in years
- `payment_to_income`: Monthly payment to monthly income ratio
- `dti`: Debt-to-income ratio

### 3. model_training.py

**Purpose:** Trains multiple ML models and handles calibration.

**Key Components:**
- `ModelTrainer` class
- Multiple model implementations
- SMOTE for handling class imbalance
- Model calibration

**Models Implemented:**
- Logistic Regression (baseline, interpretable)
- Random Forest (robust ensemble)
- XGBoost (gradient boosting)
- LightGBM (fast gradient boosting)

**Key Methods:**
- `prepare_data()`: Data preparation with optional SMOTE
- `train_logistic_regression()`: Train LR model
- `train_random_forest()`: Train RF model
- `train_xgboost()`: Train XGBoost model
- `train_lightgbm()`: Train LightGBM model
- `train_all_models()`: Train all models
- `save_model()`: Save model to disk

### 4. evaluation.py

**Purpose:** Comprehensive model evaluation with multiple metrics.

**Key Components:**
- `ModelEvaluator` class
- Multiple evaluation metrics
- Business metrics calculation
- Visualization tools

**Evaluation Metrics:**
- **Discrimination:** AUC-ROC, Precision-Recall AUC
- **Calibration:** Brier Score
- **Classification:** Accuracy, Precision, Recall, F1-Score
- **Business:** Approval Rate, Default Rate, Expected Loss

**Key Methods:**
- `calculate_metrics()`: Calculate standard ML metrics
- `calculate_business_metrics()`: Calculate business-oriented metrics
- `evaluate_model()`: Comprehensive model evaluation
- `compare_models()`: Compare multiple models
- `plot_roc_curves()`: Plot ROC curves for all models
- `plot_precision_recall_curves()`: Plot PR curves
- `select_best_model()`: Select best model based on metric

### 5. decision_engine.py

**Purpose:** Real-time loan approval decision engine.

**Key Components:**
- `LoanDecisionEngine` class
- Probability-to-decision conversion
- Batch processing support

**Key Methods:**
- `predict_proba()`: Predict eligibility probability
- `make_decision()`: Make decision for single applicant
- `make_batch_decisions()`: Process multiple applicants
- `set_threshold()`: Update decision threshold
- `get_approval_rate()`: Calculate expected approval rate
- `load_from_files()`: Load engine from saved files

### 6. explainability.py

**Purpose:** Provides SHAP-based explanations for model predictions.

**Key Components:**
- `ModelExplainer` class
- SHAP value calculations
- Reason code generation

**Key Methods:**
- `explain_prediction()`: Explain a single prediction
- `get_feature_importance_ranking()`: Get top contributing features
- `generate_reason_codes()`: Generate human-readable reason codes
- `plot_shap_summary()`: Visualize SHAP values

### 7. fairness_audit.py

**Purpose:** Assesses model fairness across demographic groups.

**Key Components:**
- `FairnessAuditor` class
- Multiple fairness metrics
- Compliance checking

**Fairness Metrics:**
- **Demographic Parity:** Equal approval rates across groups
- **Equalized Odds:** Equal TPR and FPR across groups
- **Disparate Impact:** 4/5ths rule compliance

**Key Methods:**
- `demographic_parity()`: Calculate demographic parity
- `equalized_odds()`: Calculate equalized odds
- `disparate_impact()`: Check 4/5ths rule
- `audit_model()`: Comprehensive fairness audit
- `generate_fairness_report()`: Generate human-readable report

## Main Scripts

### main.py

**Purpose:** Complete training pipeline orchestrating all components.

**Workflow:**
1. Data Preprocessing
2. Feature Engineering
3. Model Training
4. Model Evaluation
5. Explainability Analysis
6. Fairness Audit (optional)
7. Decision Engine Demo

**Outputs:**
- Trained models (`.pkl` files)
- Model comparison CSV
- Feature importance CSV
- Evaluation plots (ROC, PR curves)
- Model artifacts (scaler, feature names)

### example_usage.py

**Purpose:** Demonstrates how to use trained models for predictions.

**Examples:**
1. Single applicant decision
2. Batch processing
3. Decision with explanations

## Data Flow

```
Raw Data (CSV)
    ↓
[Data Preprocessing]
    ↓
Cleaned Features
    ↓
[Feature Engineering]
    ↓
Engineered Features
    ↓
[Model Training]
    ↓
Trained Models
    ↓
[Model Evaluation]
    ↓
Best Model Selected
    ↓
[Decision Engine]
    ↓
Approval/Rejection Decisions
```

## Configuration

Key configuration parameters in `main.py`:

- `DATA_PATH`: Path to raw data file
- `SAMPLE_SIZE`: Number of samples (None for all data)
- `TEST_SIZE`: Proportion of test set (default: 0.2)
- `USE_SMOTE`: Whether to use SMOTE for class imbalance
- `CALIBRATE_MODELS`: Whether to calibrate probability outputs
- `RANDOM_STATE`: Random seed for reproducibility

## Dependencies

See `requirements.txt` for complete list. Key dependencies:

- pandas: Data manipulation
- numpy: Numerical computations
- scikit-learn: ML algorithms and utilities
- xgboost: XGBoost gradient boosting
- lightgbm: LightGBM gradient boosting
- shap: Model explainability
- matplotlib/seaborn: Visualization
- imbalanced-learn: Handling class imbalance

## Usage Workflow

1. **Setup:** Install dependencies and download data
2. **Train:** Run `python main.py` to train models
3. **Evaluate:** Review model comparison and evaluation metrics
4. **Deploy:** Use `LoanDecisionEngine` for real-time predictions
5. **Monitor:** Track performance and retrain as needed

## Extension Points

The system is designed to be extensible:

- **Add new features:** Modify `feature_engineering.py`
- **Add new models:** Extend `model_training.py`
- **Custom metrics:** Add to `evaluation.py`
- **Deployment:** Wrap `LoanDecisionEngine` in API (Flask/FastAPI)
- **Monitoring:** Add drift detection and retraining pipelines





