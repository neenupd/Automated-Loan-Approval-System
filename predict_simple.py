"""
Simple loan prediction script with command-line arguments or defaults
Usage: python predict_simple.py [annual_inc] [loan_amnt] [dti] [fico_low] [fico_high]
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

sys.path.insert(0, str(Path(__file__).parent / 'src'))
from decision_engine import LoanDecisionEngine

def predict_loan(annual_inc=75000, loan_amnt=15000, dti=15.5, fico_low=720, fico_high=724):
    """
    Make a loan approval prediction.
    
    Args:
        annual_inc: Annual income in dollars
        loan_amnt: Loan amount requested
        dti: Debt-to-income ratio (percentage)
        fico_low: FICO score low range
        fico_high: FICO score high range
    """
    
    print("=" * 70)
    print("LOAN APPROVAL PREDICTION")
    print("=" * 70)
    print(f"\nInput Parameters:")
    print(f"  Annual Income: ${annual_inc:,.2f}")
    print(f"  Loan Amount: ${loan_amnt:,.2f}")
    print(f"  Debt-to-Income Ratio: {dti}%")
    print(f"  FICO Score Range: {fico_low}-{fico_high}")
    
    # Load model and scaler
    try:
        model = joblib.load('models/best_model_logistic_regression.pkl')
        scaler = joblib.load('models/scaler.pkl')
        feature_names = joblib.load('models/feature_names.pkl')
        print("\n[OK] Model loaded successfully")
    except FileNotFoundError as e:
        print(f"\n[ERROR] Model files not found: {e}")
        print("Please run main.py first to train models.")
        return None
    
    # Load a sample from processed data
    features_file = 'data/processed/features_engineered.csv'
    try:
        df = pd.read_csv(features_file, nrows=100)
    except FileNotFoundError:
        print(f"\n[ERROR] Processed features not found: {features_file}")
        return None
    
    # Get numeric columns matching what the scaler expects
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    matching_cols = [col for col in numeric_cols if col in feature_names]
    selected_features = matching_cols[:scaler.n_features_in_]
    
    # Get a sample applicant profile
    sample_data = df[selected_features].iloc[0].copy()
    
    # Update with user inputs where available
    if 'annual_inc' in sample_data.index:
        sample_data['annual_inc'] = annual_inc
    if 'loan_amnt' in sample_data.index:
        sample_data['loan_amnt'] = loan_amnt
    if 'dti' in sample_data.index:
        sample_data['dti'] = dti
    if 'fico_range_low' in sample_data.index:
        sample_data['fico_range_low'] = fico_low
    if 'fico_range_high' in sample_data.index:
        sample_data['fico_range_high'] = fico_high
    
    # Calculate derived features if they exist
    if 'fico_avg' in sample_data.index:
        sample_data['fico_avg'] = (fico_low + fico_high) / 2
    if 'loan_to_income' in sample_data.index:
        sample_data['loan_to_income'] = loan_amnt / annual_inc if annual_inc > 0 else 0
    
    # Create decision engine
    engine = LoanDecisionEngine(
        model=model,
        scaler=scaler,
        threshold=0.5,
        feature_names=selected_features
    )
    
    # Make prediction
    applicant_dict = sample_data.to_dict()
    decision = engine.make_decision(applicant_dict)
    
    # Display results
    print("\n" + "=" * 70)
    print("PREDICTION RESULT")
    print("=" * 70)
    print(f"\nDecision: {decision['decision']}")
    print(f"Approval Probability: {decision['probability']:.2%}")
    print(f"Confidence Level: {decision['confidence']:.2%}")
    print(f"Decision Threshold: {decision['threshold']:.0%}")
    
    if decision['decision'] == 'APPROVED':
        print("\n[APPROVED] LOAN APPROVED")
        print(f"  Model predicts {decision['probability']:.2%} probability of successful repayment.")
    else:
        print("\n[REJECTED] LOAN REJECTED")
        print(f"  Model predicts {1-decision['probability']:.2%} probability of default.")
        print("  Consider: improving credit score, reducing debt, or adjusting loan amount.")
    
    print("\n" + "=" * 70)
    print("Note: This uses a sample profile adjusted with your inputs.")
    print("For production use, full preprocessing pipeline is recommended.")
    print("=" * 70)
    
    return decision


if __name__ == "__main__":
    # Parse command-line arguments (optional)
    if len(sys.argv) > 1:
        try:
            annual_inc = float(sys.argv[1]) if len(sys.argv) > 1 else 75000
            loan_amnt = float(sys.argv[2]) if len(sys.argv) > 2 else 15000
            dti = float(sys.argv[3]) if len(sys.argv) > 3 else 15.5
            fico_low = float(sys.argv[4]) if len(sys.argv) > 4 else 720
            fico_high = float(sys.argv[5]) if len(sys.argv) > 5 else 724
        except ValueError:
            print("Error: All arguments must be numbers")
            print("Usage: python predict_simple.py [annual_inc] [loan_amnt] [dti] [fico_low] [fico_high]")
            sys.exit(1)
    else:
        # Use default values
        annual_inc = 75000
        loan_amnt = 15000
        dti = 15.5
        fico_low = 720
        fico_high = 724
    
    predict_loan(annual_inc, loan_amnt, dti, fico_low, fico_high)

