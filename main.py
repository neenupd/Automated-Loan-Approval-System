"""
Main Training Script
Complete pipeline for training and evaluating loan approval models.
"""

import sys
import os
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data_preprocessing import DataPreprocessor
from feature_engineering import FeatureEngineer
from model_training import ModelTrainer
from evaluation import ModelEvaluator
from decision_engine import LoanDecisionEngine
from explainability import ModelExplainer
from fairness_audit import FairnessAuditor


def main():
    """Main training pipeline."""
    
    print("=" * 70)
    print("AUTOMATED LOAN APPROVAL SYSTEM - MODEL TRAINING")
    print("=" * 70)
    
    # Configuration
    # Try nested path first (if dataset was extracted to a folder), then direct path
    DATA_PATH = 'data/raw/accepted_2007_to_2018q4.csv/accepted_2007_to_2018Q4.csv'
    if not os.path.exists(DATA_PATH):
        DATA_PATH = 'data/raw/accepted_2007_to_2018Q4.csv'
    SAMPLE_SIZE = 50000  # Set to None to use all data, or specify a number (e.g., 50000)
    TEST_SIZE = 0.2
    USE_SMOTE = True
    CALIBRATE_MODELS = True
    RANDOM_STATE = 42
    
    # Create output directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    # Step 1: Data Preprocessing
    print("\n" + "=" * 70)
    print("STEP 1: DATA PREPROCESSING")
    print("=" * 70)
    
    preprocessor = DataPreprocessor(target_column='loan_status')
    
    try:
        features, target = preprocessor.preprocess(
            file_path=DATA_PATH,
            sample_size=SAMPLE_SIZE,
            missing_threshold=0.5
        )
        
        # Save processed data
        features.to_csv('data/processed/features.csv', index=False)
        target.to_csv('data/processed/target.csv', index=False)
        print("\n[OK] Preprocessed data saved")
        
    except FileNotFoundError:
        print(f"\n[ERROR] Data file not found at {DATA_PATH}")
        print("Please download the Lending Club dataset from Kaggle and place it in data/raw/")
        print("Dataset: https://www.kaggle.com/datasets/wordsforthewise/lending-club")
        return
    except Exception as e:
        print(f"\n[ERROR] Error during preprocessing: {e}")
        return
    
    # Step 2: Feature Engineering
    print("\n" + "=" * 70)
    print("STEP 2: FEATURE ENGINEERING")
    print("=" * 70)
    
    engineer = FeatureEngineer()
    features_eng = engineer.engineer_features(features, exclude_original=False)
    
    # Save engineered features
    features_eng.to_csv('data/processed/features_engineered.csv', index=False)
    print("[OK] Engineered features saved")
    
    # Save feature names
    joblib.dump(features_eng.columns.tolist(), 'models/feature_names.pkl')
    
    # Step 3: Model Training
    print("\n" + "=" * 70)
    print("STEP 3: MODEL TRAINING")
    print("=" * 70)
    
    trainer = ModelTrainer(random_state=RANDOM_STATE)
    
    model_results = trainer.train_all_models(
        X=features_eng,
        y=target,
        test_size=TEST_SIZE,
        use_smote=USE_SMOTE,
        calibrate=CALIBRATE_MODELS
    )
    
    # Save scaler
    trainer.save_scaler('models/scaler.pkl')
    
    # Step 4: Model Evaluation
    print("\n" + "=" * 70)
    print("STEP 4: MODEL EVALUATION")
    print("=" * 70)
    
    evaluator = ModelEvaluator()
    
    # Compare models
    comparison_df = evaluator.compare_models(model_results)
    print("\nModel Comparison:")
    print(comparison_df.to_string(index=False))
    
    # Save comparison
    comparison_df.to_csv('models/model_comparison.csv', index=False)
    
    # Select best model
    best_model_name, best_model_result = evaluator.select_best_model(
        model_results, metric='auc_roc'
    )
    
    print(f"\n[OK] Best Model: {best_model_name}")
    print(f"  AUC-ROC: {evaluator.evaluate_model(best_model_result)['metrics']['auc_roc']:.4f}")
    
    # Save best model
    trainer.save_model(
        trainer.models[best_model_name],
        f'models/best_model_{best_model_name}.pkl'
    )
    
    # Save all models
    for model_name, model in trainer.models.items():
        trainer.save_model(model, f'models/{model_name}.pkl')
    
    # Plot evaluation curves
    print("\nGenerating evaluation plots...")
    evaluator.plot_roc_curves(model_results, save_path='models/roc_curves.png')
    evaluator.plot_precision_recall_curves(model_results, save_path='models/pr_curves.png')
    
    # Step 5: Explainability (for best model) - SKIPPED BY DEFAULT (very slow)
    # Uncomment the code below if you want SHAP explanations (can take hours)
    print("\n" + "=" * 70)
    print("STEP 5: MODEL EXPLAINABILITY (SKIPPED)")
    print("=" * 70)
    print("[INFO] SHAP explanations skipped for speed (can take hours)")
    print("       To enable, uncomment the SHAP section in main.py")
    
    # UNCOMMENT BELOW FOR SHAP EXPLANATIONS (SLOW!)
    """
    try:
        # Get training data for explainer
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            features_eng, target, test_size=TEST_SIZE, 
            random_state=RANDOM_STATE, stratify=target
        )
        
        # Scale data
        X_train_scaled = trainer.scaler.transform(X_train)
        
        explainer = ModelExplainer(
            model=trainer.models[best_model_name],
            X_train=X_train_scaled,
            feature_names=features_eng.columns.tolist()
        )
        
        # Explain a sample prediction
        X_test_scaled = trainer.scaler.transform(X_test)
        sample_idx = 0
        explanation = explainer.explain_prediction(X_test_scaled, instance_idx=sample_idx)
        
        print(f"\n[OK] SHAP explainer initialized")
        
        # Get feature importance ranking
        importance_df = explainer.get_feature_importance_ranking(
            X_test_scaled[:10], top_n=10
        )
        print("\nTop 10 Most Important Features:")
        print(importance_df.to_string(index=False))
        importance_df.to_csv('models/feature_importance.csv', index=False)
        
        # Generate reason codes for sample
        reason_codes = explainer.generate_reason_codes(
            X_test_scaled[sample_idx:sample_idx+1], 
            top_n=5, 
            format='text'
        )
        print("\nSample Reason Codes:")
        print(reason_codes)
        
    except Exception as e:
        print(f"\n[WARNING] Could not generate explanations: {e}")
    """
    
    # Step 6: Fairness Audit (if protected attributes available)
    print("\n" + "=" * 70)
    print("STEP 6: FAIRNESS AUDIT")
    print("=" * 70)
    
    # Note: Protected attributes need to be present in the data
    # Common examples: 'home_ownership', 'purpose', 'addr_state', etc.
    # This is a placeholder - adjust based on actual data
    protected_attributes = []  # Add relevant protected attributes here
    
    if protected_attributes:
        try:
            # Get test predictions
            X_test_scaled = trainer.scaler.transform(X_test)
            y_pred_best = trainer.models[best_model_name].predict(X_test_scaled)
            
            # Get original data for protected attributes
            # You may need to load original data to get protected attributes
            # For now, this is a placeholder
            
            auditor = FairnessAuditor(protected_attributes=protected_attributes)
            # audit_results = auditor.audit_model(y_test, y_pred_best, test_data_with_protected_attrs)
            # report = auditor.generate_fairness_report(audit_results)
            # print(report)
            
            print("\n[WARNING] Fairness audit skipped - protected attributes not found in processed data")
            print("   To enable fairness audit, include protected attributes in data preprocessing")
            
        except Exception as e:
            print(f"\n[WARNING] Fairness audit failed: {e}")
    else:
        print("\n[WARNING] Fairness audit skipped - no protected attributes specified")
        print("   Add protected attributes to the protected_attributes list to enable")
    
    # Step 7: Decision Engine Example
    print("\n" + "=" * 70)
    print("STEP 7: DECISION ENGINE DEMO")
    print("=" * 70)
    
    decision_engine = LoanDecisionEngine(
        model=trainer.models[best_model_name],
        scaler=trainer.scaler,
        threshold=0.5,
        feature_names=features_eng.columns.tolist()
    )
    
    # Example: Make decision for a sample applicant
    if 'X_test' in locals() and len(X_test) > 0:
        # Get numeric columns only for the sample
        sample_data = X_test.iloc[[0]]
        sample_applicant = sample_data.to_dict('records')[0]
        decision = decision_engine.make_decision(sample_applicant)
        
        print("\nExample Decision:")
        print(f"  Decision: {decision['decision']}")
        print(f"  Probability: {decision['probability']:.4f}")
        print(f"  Threshold: {decision['threshold']}")
    
    # Save decision engine configuration
    joblib.dump({
        'model_name': best_model_name,
        'threshold': 0.5,
        'feature_names': features_eng.columns.tolist()
    }, 'models/decision_engine_config.pkl')
    
    print("\n" + "=" * 70)
    print("TRAINING PIPELINE COMPLETE!")
    print("=" * 70)
    print("\nSaved outputs:")
    print("  - Trained models: models/*.pkl")
    print("  - Model comparison: models/model_comparison.csv")
    print("  - Feature importance: models/feature_importance.csv")
    print("  - Evaluation plots: models/roc_curves.png, models/pr_curves.png")
    print("\nNext steps:")
    print("  1. Review model_comparison.csv to select best model")
    print("  2. Use decision_engine.py for real-time predictions")
    print("  3. Monitor model performance and retrain as needed")


if __name__ == "__main__":
    main()

