"""
Quick script to check training status and view results
"""

import os
import pandas as pd
import joblib
from pathlib import Path

print("=" * 70)
print("TRAINING STATUS CHECK")
print("=" * 70)

# Check if models exist
models_dir = Path('models')
if models_dir.exists():
    model_files = list(models_dir.glob('*.pkl'))
    csv_files = list(models_dir.glob('*.csv'))
    png_files = list(models_dir.glob('*.png'))
    
    print(f"\nModels directory: {models_dir}")
    print(f"  Model files (*.pkl): {len(model_files)}")
    print(f"  CSV files: {len(csv_files)}")
    print(f"  Image files (*.png): {len(png_files)}")
    
    # Show model comparison if exists
    comparison_file = models_dir / 'model_comparison.csv'
    if comparison_file.exists():
        print("\n" + "=" * 70)
        print("MODEL COMPARISON RESULTS")
        print("=" * 70)
        df = pd.read_csv(comparison_file)
        print(df.to_string(index=False))
        
        # Show best model
        if 'auc_roc' in df.columns:
            best_idx = df['auc_roc'].idxmax()
            best_model = df.loc[best_idx]
            print(f"\nBest Model (by AUC-ROC): {best_model['model']}")
            print(f"  AUC-ROC: {best_model['auc_roc']:.4f}")
            print(f"  Accuracy: {best_model['accuracy']:.4f}")
            print(f"  Precision: {best_model['precision']:.4f}")
            print(f"  Recall: {best_model['recall']:.4f}")
    
    # List available models
    print("\n" + "=" * 70)
    print("AVAILABLE MODEL FILES")
    print("=" * 70)
    for f in sorted(model_files):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name}: {size_mb:.2f} MB")
    
else:
    print("\nModels directory does not exist. Training not yet run.")

# Check processed data
processed_dir = Path('data/processed')
if processed_dir.exists():
    processed_files = list(processed_dir.glob('*.csv'))
    print("\n" + "=" * 70)
    print("PROCESSED DATA FILES")
    print("=" * 70)
    for f in processed_files:
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name}: {size_mb:.2f} MB")

print("\n" + "=" * 70)
print("STATUS CHECK COMPLETE")
print("=" * 70)

