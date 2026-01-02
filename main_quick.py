"""
Quick Training Script - Skips expensive operations like SHAP
Use this for faster iteration during development
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


def main():
    """Quick training pipeline without expensive operations."""
    
    print("=" * 70)
    print("AUTOMATED LOAN APPROVAL SYSTEM - QUICK TRAINING")
    print("=" * 70)
    
    # Configuration
    DATA_PATH = 'data/raw/accepted_2007_to_2018q4.csv/accepted_2007_to_2018Q4.csv'
    if not os.path.exists(DATA_PATH):
        DATA_PATH = 'data/raw/accepted_2007_to_2018Q4.csv'
    SAMPLE_SIZE = 50000  # Use sample for faster training
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
        print("\n[OK] Preprocessing complete")
        
    except Exception as e:
        print(f"\n[ERROR] Error during preprocessing: {e}")
        return
    
    # Step 2: Feature Engineering
    print("\n" + "=" * 70)
    print("STEP 2: FEATURE ENGINEERING")
    print("=" * 70)
    
    engineer = FeatureEngineer()
    features_eng = engineer.engineer_features(features, exclude_original=False)
    print("[OK] Feature engineering complete")
    
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
    best_metrics = evaluator.evaluate_model(best_model_result)['metrics']
    print(f"  AUC-ROC: {best_metrics['auc_roc']:.4f}")
    print(f"  Accuracy: {best_metrics['accuracy']:.4f}")
    print(f"  Precision: {best_metrics['precision']:.4f}")
    print(f"  Recall: {best_metrics['recall']:.4f}")
    
    # Save best model
    trainer.save_model(
        trainer.models[best_model_name],
        f'models/best_model_{best_model_name}.pkl'
    )
    
    # Save all models
    for model_name, model in trainer.models.items():
        trainer.save_model(model, f'models/{model_name}.pkl')
    
    # Plot evaluation curves (quick, no blocking)
    print("\nGenerating evaluation plots...")
    try:
        evaluator.plot_roc_curves(model_results, save_path='models/roc_curves.png')
        evaluator.plot_precision_recall_curves(model_results, save_path='models/pr_curves.png')
        print("[OK] Plots saved")
    except Exception as e:
        print(f"[WARNING] Could not generate plots: {e}")
    
    # Step 5: Decision Engine Example (Quick)
    print("\n" + "=" * 70)
    print("STEP 5: DECISION ENGINE DEMO")
    print("=" * 70)
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        features_eng, target, test_size=TEST_SIZE, 
        random_state=RANDOM_STATE, stratify=target
    )
    
    decision_engine = LoanDecisionEngine(
        model=trainer.models[best_model_name],
        scaler=trainer.scaler,
        threshold=0.5,
        feature_names=trainer.feature_names  # Use numeric features only
    )
    
    # Example: Make decision for a sample applicant
    if len(X_test) > 0:
        # Get only numeric columns
        numeric_cols = X_test.select_dtypes(include=[np.number]).columns.tolist()
        sample_data = X_test[numeric_cols].iloc[[0]]
        sample_applicant = sample_data.to_dict('records')[0]
        
        decision = decision_engine.make_decision(sample_applicant)
        
        print("\nExample Decision:")
        print(f"  Decision: {decision['decision']}")
        print(f"  Probability: {decision['probability']:.4f}")
        print(f"  Threshold: {decision['threshold']:.4f}")
    
    # Save decision engine configuration
    joblib.dump({
        'model_name': best_model_name,
        'threshold': 0.5,
        'feature_names': trainer.feature_names
    }, 'models/decision_engine_config.pkl')
    
    print("\n" + "=" * 70)
    print("QUICK TRAINING COMPLETE!")
    print("=" * 70)
    print("\nSaved outputs:")
    print("  - Trained models: models/*.pkl")
    print("  - Model comparison: models/model_comparison.csv")
    print("  - Evaluation plots: models/roc_curves.png, models/pr_curves.png")
    print("\nNote: SHAP explanations skipped for speed")
    print("Run main.py for full pipeline with explainability")


if __name__ == "__main__":
    main()

