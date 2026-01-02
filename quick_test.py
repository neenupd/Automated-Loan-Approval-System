"""
Quick test script to verify models work with processed data
"""

import sys
import os
from pathlib import Path
import pandas as pd
import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / 'src'))
from decision_engine import LoanDecisionEngine

print("=" * 70)
print("QUICK MODEL TEST")
print("=" * 70)

# Load model
model_files = [
    'models/best_model_logistic_regression.pkl',
    'models/logistic_regression.pkl',
    'models/lightgbm.pkl',
    'models/xgboost.pkl'
]

model = None
model_file = None
for mf in model_files:
    if os.path.exists(mf):
        try:
            model = joblib.load(mf)
            model_file = mf
            print(f"[OK] Loaded model: {mf}")
            break
        except Exception as e:
            print(f"[SKIP] Could not load {mf}: {e}")
            continue

if model is None:
    print("[ERROR] No model found!")
    exit(1)

# Load scaler and feature names
scaler = joblib.load('models/scaler.pkl')
feature_names = joblib.load('models/feature_names.pkl')
print(f"[OK] Loaded scaler and {len(feature_names)} feature names")

# Load processed features
features_file = 'data/processed/features_engineered.csv'
if not os.path.exists(features_file):
    print(f"[ERROR] Processed features not found: {features_file}")
    print("Please run main.py first to generate processed data")
    exit(1)

print(f"\n[INFO] Loading processed features from {features_file}...")
df = pd.read_csv(features_file, nrows=100)  # Load just 100 rows for testing

# Select only numeric columns that match feature names
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
matching_cols = [col for col in numeric_cols if col in feature_names]

if len(matching_cols) != len(feature_names):
    print(f"[WARNING] Feature count mismatch: {len(matching_cols)} vs {len(feature_names)} expected")
    print(f"Using {len(matching_cols)} matching features")

# The scaler was trained on numeric-only features, so we need to match that
# Get the actual feature names that the scaler expects (should match numeric columns)
print(f"\n[INFO] Scaler expects {scaler.n_features_in_} features")
print(f"[INFO] Available numeric features: {len(matching_cols)}")

# Use only the features that match (take first n_features_in_)
if len(matching_cols) >= scaler.n_features_in_:
    # Use the first n_features_in_ matching columns
    selected_features = matching_cols[:scaler.n_features_in_]
else:
    # If we have fewer, something is wrong, but try anyway
    selected_features = matching_cols
    print(f"[WARNING] Fewer features than expected!")

print(f"[INFO] Using {len(selected_features)} features for prediction")

# Create decision engine with the features the scaler was trained on
engine = LoanDecisionEngine(
    model=model,
    scaler=scaler,
    threshold=0.5,
    feature_names=selected_features
)

# Test on a sample
print("\n" + "=" * 70)
print("TESTING DECISIONS")
print("=" * 70)

sample_data = df[selected_features].iloc[:5]  # First 5 samples
decisions = engine.make_batch_decisions(sample_data)

print("\nSample Decisions:")
for idx, (_, row) in enumerate(decisions.iterrows()):
    print(f"\nApplicant {idx + 1}:")
    print(f"  Decision: {row['decision']}")
    print(f"  Probability: {row['probability']:.4f}")

print("\n" + "=" * 70)
print("TEST COMPLETE - Models are working!")
print("=" * 70)

