# Setup Instructions

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

## Installation Steps

### 1. Clone or Download the Project

```bash
cd Automated-Loan-Approval-System
```

### 2. Create a Virtual Environment (Recommended)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

**Windows:**
```bash
python -m pip install -r requirements.txt
```

**Linux/Mac:**
```bash
pip install -r requirements.txt
```

**Note:** If `pip` is not recognized on Windows, use `python -m pip` instead.

### 4. Download the Dataset

1. Visit [Kaggle Lending Club Dataset](https://www.kaggle.com/datasets/wordsforthewise/lending-club)
2. Download the dataset (you may need a Kaggle account)
3. Extract the CSV file(s) to the `data/raw/` directory

**Expected file location:** `data/raw/accepted_2007_to_2018Q4.csv`

**Note:** The dataset is large (~1.6GB). You can use a sample for testing by setting `SAMPLE_SIZE` in `main.py`.

### 5. Verify Installation

```bash
python -c "import pandas, sklearn, xgboost, lightgbm, shap; print('All packages installed successfully!')"
```

## Quick Start

### Training Models

```bash
python main.py
```

This will:
1. Load and preprocess the data
2. Engineer features
3. Train multiple models (Logistic Regression, Random Forest, XGBoost, LightGBM)
4. Evaluate models and select the best one
5. Generate explainability reports
6. Save models and results

**Note:** Training on the full dataset may take several hours. Use `SAMPLE_SIZE = 50000` in `main.py` for faster testing.

### Using the Trained Model

```bash
python example_usage.py
```

## Project Structure

```
Automated-Loan-Approval-System/
├── data/
│   ├── raw/              # Place downloaded dataset here
│   └── processed/        # Processed data (generated)
├── src/                  # Source code modules
├── models/               # Saved models (generated)
├── notebooks/            # Jupyter notebooks for exploration
├── tests/                # Unit tests
├── main.py              # Main training script
├── example_usage.py     # Example usage script
└── requirements.txt     # Python dependencies
```

## Troubleshooting

### Issue: ModuleNotFoundError

**Solution:** Make sure you've installed all dependencies:
```bash
# Windows
python -m pip install -r requirements.txt

# Linux/Mac
pip install -r requirements.txt
```

### Issue: Dataset not found

**Solution:** 
1. Download the dataset from Kaggle
2. Place it in `data/raw/accepted_2007_to_2018Q4.csv`
3. Or update the `DATA_PATH` in `main.py` to point to your data location

### Issue: Memory errors during training

**Solution:**
1. Reduce `SAMPLE_SIZE` in `main.py` (e.g., `SAMPLE_SIZE = 10000`)
2. Increase system RAM or use a machine with more memory
3. Process data in chunks

### Issue: SHAP installation problems

**Solution:** SHAP can be tricky to install. Try:
```bash
# Windows
python -m pip install shap --upgrade

# Linux/Mac
pip install shap --upgrade
```

Or if you encounter build issues:
```bash
# Windows
python -m pip install shap --no-build-isolation

# Linux/Mac
pip install shap --no-build-isolation
```

## Next Steps

1. **Explore the Data**: Use Jupyter notebooks in the `notebooks/` directory
2. **Customize Models**: Adjust hyperparameters in `src/model_training.py`
3. **Add Features**: Modify feature engineering in `src/feature_engineering.py`
4. **Deploy**: Integrate the decision engine into your application

## Support

For issues or questions, please refer to the README.md or open an issue in the repository.

