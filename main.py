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
    
    DATA_PATH = 'data/raw/accepted_2007_to_2018q4.csv/accepted_2007_to_2018Q4.csv'
    if not os.path.exists(DATA_PATH):
        DATA_PATH = 'data/raw/accepted_2007_to_2018Q4.csv'
    SAMPLE_SIZE = 50000
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
    features_eng.to_csv('data/processed/features_engineered.csv', index=False)
    print("[OK] Engineered features saved")
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
    trainer.save_scaler('models/scaler.pkl')
    
    # Step 4: Model Evaluation
    print("\n" + "=" * 70)
    print("STEP 4: MODEL EVALUATION")
    print("=" * 70)
    
    evaluator = ModelEvaluator()
    comparison_df = evaluator.compare_models(model_results)
    print("\nModel Comparison:")
    print(comparison_df.to_string(index=False))
    comparison_df.to_csv('models/model_comparison.csv', index=False)
    
    best_model_name, best_model_result = evaluator.select_best_model(
        model_results, metric='auc_roc'
    )
    
    print(f"\n[OK] Best Model: {best_model_name}")
    print(f"  AUC-ROC: {evaluator.evaluate_model(best_model_result)['metrics']['auc_roc']:.4f}")
    
    trainer.save_model(
        trainer.models[best_model_name],
        f'models/best_model_{best_model_name}.pkl'
    )
    
    for model_name, model in trainer.models.items():
        trainer.save_model(model, f'models/{model_name}.pkl')
    
    print("\nGenerating evaluation plots...")
    evaluator.plot_roc_curves(model_results, save_path='models/roc_curves.png')
    evaluator.plot_precision_recall_curves(model_results, save_path='models/pr_curves.png')
    
    print("\n" + "=" * 70)
    print("STEP 5: MODEL EXPLAINABILITY (SKIPPED)")
    print("=" * 70)
    print("[INFO] SHAP explanations skipped for speed")
    
    print("\n" + "=" * 70)
    print("STEP 6: FAIRNESS AUDIT")
    print("=" * 70)
    
    protected_attributes = []
    
    if protected_attributes:
        try:
            auditor = FairnessAuditor(protected_attributes=protected_attributes)
            print("\n[WARNING] Fairness audit skipped - protected attributes not found in processed data")
            print("   To enable fairness audit, include protected attributes in data preprocessing")
        except Exception as e:
            print(f"\n[WARNING] Fairness audit failed: {e}")
    else:
        print("\n[WARNING] Fairness audit skipped - no protected attributes specified")
        print("   Add protected attributes to the protected_attributes list to enable")
    
    print("\n" + "=" * 70)
    print("STEP 7: DECISION ENGINE DEMO")
    print("=" * 70)
    
    decision_engine = LoanDecisionEngine(
        model=trainer.models[best_model_name],
        scaler=trainer.scaler,
        threshold=0.5,
        feature_names=features_eng.columns.tolist()
    )
    
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
    print("  - Evaluation plots: models/roc_curves.png, models/pr_curves.png")
    print("\nNext steps:")
    print("  1. Review model_comparison.csv to select best model")
    print("  2. Use predict_simple.py for real-time predictions")
    print("  3. Monitor model performance and retrain as needed")


if __name__ == "__main__":
    main()

