"""
Example Usage Script
Demonstrates how to use the trained model for making loan approval decisions.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import joblib

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from decision_engine import LoanDecisionEngine
from explainability import ModelExplainer


def example_single_decision():
    """Example: Make a decision for a single applicant using processed data."""
    print("=" * 60)
    print("EXAMPLE 1: Single Applicant Decision")
    print("=" * 60)
    
    # Load model and scaler
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
                break
            except:
                continue
    
    if model is None:
        print("Error: Model files not found. Please run main.py first to train models.")
        return
    
    try:
        scaler = joblib.load('models/scaler.pkl')
        feature_names = joblib.load('models/feature_names.pkl')
        print(f"[OK] Loaded model from: {model_file}")
    except FileNotFoundError as e:
        print(f"Error: Required files not found: {e}")
        return
    
    # Load processed features to get a real example
    features_file = 'data/processed/features_engineered.csv'
    if not os.path.exists(features_file):
        print(f"Error: Processed features not found: {features_file}")
        print("Please run main.py first to generate processed data")
        return
    
    # Load a sample from processed data
    import numpy as np
    df = pd.read_csv(features_file, nrows=10)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    matching_cols = [col for col in numeric_cols if col in feature_names]
    
    # Use only the features the scaler expects
    selected_features = matching_cols[:scaler.n_features_in_]
    
    # Initialize decision engine
    engine = LoanDecisionEngine(
        model=model,
        scaler=scaler,
        threshold=0.5,
        feature_names=selected_features
    )
    
    # Get a sample applicant from processed data
    sample_data = df[selected_features].iloc[0]
    applicant_data = sample_data.to_dict()
    
    # Example: Using processed numeric features
    print("\n[INFO] Using sample applicant from processed data")
    print(f"[INFO] Using {len(selected_features)} numeric features")
    # Note: applicant_data now contains processed numeric features
    
    # Make decision
    decision = engine.make_decision(applicant_data)
    
    print(f"\nDecision: {decision['decision']}")
    print(f"Probability: {decision['probability']:.4f}")
    print(f"Confidence: {decision['confidence']:.4f}")
    print(f"Threshold: {decision['threshold']:.4f}")


def example_batch_decisions():
    """Example: Make decisions for multiple applicants."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Batch Decisions")
    print("=" * 60)
    
    # Load model and scaler (try multiple model names)
    model_files = [
        'models/best_model_logistic_regression.pkl',
        'models/best_model_xgboost.pkl',
        'models/best_model_lightgbm.pkl',
        'models/logistic_regression.pkl',
        'models/xgboost.pkl',
        'models/lightgbm.pkl'
    ]
    
    model = None
    for mf in model_files:
        if os.path.exists(mf):
            try:
                model = joblib.load(mf)
                break
            except:
                continue
    
    if model is None:
        print("Error: Model files not found. Please run main.py first to train models.")
        return
    
    try:
        scaler = joblib.load('models/scaler.pkl')
        feature_names = joblib.load('models/feature_names.pkl')
    except FileNotFoundError as e:
        print(f"Error: Required files not found: {e}")
        return
    
    engine = LoanDecisionEngine(
        model=model,
        scaler=scaler,
        threshold=0.5,
        feature_names=feature_names
    )
    
    # Load test data (if available)
    try:
        features = pd.read_csv('data/processed/features_engineered.csv')
        sample_data = features.head(10)  # Sample 10 applicants
        
        decisions = engine.make_batch_decisions(sample_data)
        
        print(f"\nProcessed {len(decisions)} applicants")
        print(f"Approval rate: {decisions['decision'].value_counts()['APPROVED'] / len(decisions):.2%}")
        print("\nSample decisions:")
        print(decisions.head().to_string())
        
    except FileNotFoundError:
        print("Processed data not found. Please run main.py first.")


def example_with_explanations():
    """Example: Make decision with SHAP explanations."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Decision with Explanations")
    print("=" * 60)
    
    # Load model and scaler (try multiple model names)
    model_files = [
        'models/best_model_logistic_regression.pkl',
        'models/best_model_xgboost.pkl',
        'models/best_model_lightgbm.pkl',
        'models/logistic_regression.pkl',
        'models/xgboost.pkl',
        'models/lightgbm.pkl'
    ]
    
    model = None
    for mf in model_files:
        if os.path.exists(mf):
            try:
                model = joblib.load(mf)
                break
            except:
                continue
    
    if model is None:
        print("Error: Model files not found. Please run main.py first to train models.")
        return
    
    try:
        scaler = joblib.load('models/scaler.pkl')
        feature_names = joblib.load('models/feature_names.pkl')
    except FileNotFoundError as e:
        print(f"Error: Required files not found: {e}")
        return
    
    # For explanations, we need training data
    try:
        from sklearn.model_selection import train_test_split
        features = pd.read_csv('data/processed/features_engineered.csv')
        target = pd.read_csv('data/processed/target.csv')
        
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, random_state=42
        )
        
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        explainer = ModelExplainer(
            model=model,
            X_train=X_train_scaled,
            feature_names=feature_names
        )
        
        # Explain a prediction
        sample_idx = 0
        reason_codes = explainer.generate_reason_codes(
            X_test_scaled[sample_idx:sample_idx+1],
            top_n=5,
            format='text'
        )
        
        print("\nReason Codes for Decision:")
        print(reason_codes)
        
        # Feature importance
        importance = explainer.get_feature_importance_ranking(
            X_test_scaled[sample_idx:sample_idx+1],
            top_n=10
        )
        
        print("\nTop Contributing Features:")
        print(importance.to_string(index=False))
        
    except Exception as e:
        print(f"Could not generate explanations: {e}")


if __name__ == "__main__":
    print("\nAUTOMATED LOAN APPROVAL SYSTEM - EXAMPLE USAGE")
    print("=" * 60)
    
    # Run examples
    example_single_decision()
    example_batch_decisions()
    example_with_explanations()
    
    print("\n" + "=" * 60)
    print("Examples complete!")
    print("=" * 60)

