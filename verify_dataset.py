"""
Quick script to verify the dataset is accessible and show basic info.
"""

import os
import sys
from pathlib import Path
import pandas as pd

# Try to find the dataset
possible_paths = [
    'data/raw/accepted_2007_to_2018q4.csv/accepted_2007_to_2018Q4.csv',  # Nested extraction
    'data/raw/accepted_2007_to_2018Q4.csv',  # Direct path
]

data_path = None
for path in possible_paths:
    if os.path.exists(path):
        data_path = path
        break

if not data_path:
    print("ERROR: Dataset not found!")
    print("Please ensure the file is at one of these locations:")
    for path in possible_paths:
        print(f"  - {path}")
    sys.exit(1)

print("=" * 60)
print("DATASET VERIFICATION")
print("=" * 60)
print(f"\nFile found at: {data_path}")

# Get file size
size_gb = os.path.getsize(data_path) / (1024**3)
print(f"File size: {size_gb:.2f} GB")

# Read a sample
print("\nReading sample (first 100 rows)...")
try:
    df_sample = pd.read_csv(data_path, nrows=100, low_memory=False)
    print(f"[OK] Successfully read sample")
    print(f"  Total columns: {len(df_sample.columns)}")
    print(f"  Sample rows: {len(df_sample)}")
    
    # Check for target column
    if 'loan_status' in df_sample.columns:
        print(f"\n[OK] Target column 'loan_status' found")
        print(f"  Unique values: {df_sample['loan_status'].unique()[:10]}")
        print(f"  Value counts:\n{df_sample['loan_status'].value_counts()}")
    else:
        print("\n[WARNING] 'loan_status' column not found")
        print(f"  Available columns (first 20): {list(df_sample.columns[:20])}")
    
    # Check for key features
    key_features = ['annual_inc', 'loan_amnt', 'dti', 'fico_range_low', 'fico_range_high']
    found_features = [f for f in key_features if f in df_sample.columns]
    print(f"\n[OK] Key features found: {len(found_features)}/{len(key_features)}")
    if found_features:
        print(f"  Found: {found_features}")
    missing = [f for f in key_features if f not in df_sample.columns]
    if missing:
        print(f"  Missing: {missing}")
    
    print("\n" + "=" * 60)
    print("DATASET VERIFICATION COMPLETE")
    print("=" * 60)
    print("\nYou can now run: python main.py")
    
except Exception as e:
    print(f"\n[ERROR] Error reading dataset: {e}")
    sys.exit(1)

