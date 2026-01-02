"""
Test Script
Verifies that all dependencies and modules are properly installed and importable.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))


def test_imports():
    """Test that all required packages can be imported."""
    print("Testing package imports...")
    
    packages = [
        'pandas',
        'numpy',
        'sklearn',
        'xgboost',
        'lightgbm',
        'shap',
        'matplotlib',
        'seaborn',
        'joblib',
        'imblearn'
    ]
    
    failed = []
    for package in packages:
        try:
            __import__(package)
            print(f"  [OK] {package}")
        except ImportError:
            print(f"  [FAIL] {package} - NOT INSTALLED")
            failed.append(package)
    
    if failed:
        print(f"\n[ERROR] Failed to import: {', '.join(failed)}")
        print("Please install missing packages: python -m pip install -r requirements.txt")
        return False
    else:
        print("\n[SUCCESS] All packages imported successfully!")
        return True


def test_modules():
    """Test that all project modules can be imported."""
    print("\nTesting project modules...")
    
    modules = [
        'data_preprocessing',
        'feature_engineering',
        'model_training',
        'evaluation',
        'decision_engine',
        'explainability',
        'fairness_audit'
    ]
    
    failed = []
    for module in modules:
        try:
            __import__(module)
            print(f"  [OK] {module}")
        except ImportError as e:
            print(f"  [FAIL] {module} - {str(e)}")
            failed.append(module)
    
    if failed:
        print(f"\n[ERROR] Failed to import modules: {', '.join(failed)}")
        return False
    else:
        print("\n[SUCCESS] All modules imported successfully!")
        return True


def test_classes():
    """Test that main classes can be instantiated."""
    print("\nTesting class instantiation...")
    
    try:
        from data_preprocessing import DataPreprocessor
        from feature_engineering import FeatureEngineer
        from model_training import ModelTrainer
        from evaluation import ModelEvaluator
        from decision_engine import LoanDecisionEngine
        from explainability import ModelExplainer
        from fairness_audit import FairnessAuditor
        
        # Try to instantiate (may fail if dependencies missing, but structure should work)
        preprocessor = DataPreprocessor()
        engineer = FeatureEngineer()
        trainer = ModelTrainer()
        evaluator = ModelEvaluator()
        auditor = FairnessAuditor(protected_attributes=[])
        
        print("  [OK] DataPreprocessor")
        print("  [OK] FeatureEngineer")
        print("  [OK] ModelTrainer")
        print("  [OK] ModelEvaluator")
        print("  [OK] FairnessAuditor")
        print("\n[SUCCESS] All classes can be instantiated!")
        return True
        
    except Exception as e:
        print(f"  [FAIL] Error: {str(e)}")
        return False


def test_directories():
    """Test that required directories exist."""
    print("\nTesting directory structure...")
    
    directories = [
        'data/raw',
        'data/processed',
        'src',
        'models'
    ]
    
    all_exist = True
    for directory in directories:
        if Path(directory).exists():
            print(f"  [OK] {directory}")
        else:
            print(f"  [MISSING] {directory} - DOES NOT EXIST")
            all_exist = False
    
    if all_exist:
        print("\n[SUCCESS] All required directories exist!")
    else:
        print("\n[WARNING] Some directories are missing, but they will be created when needed")
    
    return True  # Not critical


if __name__ == "__main__":
    print("=" * 60)
    print("AUTOMATED LOAN APPROVAL SYSTEM - SETUP TEST")
    print("=" * 60)
    
    results = []
    
    results.append(("Package Imports", test_imports()))
    results.append(("Module Imports", test_modules()))
    results.append(("Class Instantiation", test_classes()))
    results.append(("Directory Structure", test_directories()))
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{test_name}: {status}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        print("\n[SUCCESS] All tests passed! System is ready to use.")
        print("\nNext steps:")
        print("  1. Download the Lending Club dataset from Kaggle")
        print("  2. Place it in data/raw/accepted_2007_to_2018Q4.csv")
        print("  3. Run: python main.py")
    else:
        print("\n[ERROR] Some tests failed. Please fix the issues above.")
        sys.exit(1)

