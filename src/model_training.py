"""
Model Training Module
Trains multiple ML models and performs model selection and calibration.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
from typing import Dict, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')


class ModelTrainer:
    """Trains and evaluates multiple ML models for loan approval prediction."""
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the model trainer.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def prepare_data(self, X: pd.DataFrame, y: pd.Series,
                    test_size: float = 0.2,
                    use_smote: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for training with optional SMOTE for class imbalance.
        
        Args:
            X: Features DataFrame
            y: Target Series
            test_size: Proportion of data for testing
            use_smote: Whether to use SMOTE for oversampling
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        # Drop non-numeric columns (dates, strings) that can't be scaled
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        non_numeric_cols = [col for col in X.columns if col not in numeric_cols]
        
        if non_numeric_cols:
            print(f"Dropping {len(non_numeric_cols)} non-numeric columns before scaling: {non_numeric_cols[:5]}...")
            X = X[numeric_cols]
        
        self.feature_names = X.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # Scale features (now all numeric)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Apply SMOTE if requested
        if use_smote:
            print("Applying SMOTE to handle class imbalance...")
            smote = SMOTE(random_state=self.random_state)
            X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)
            print(f"After SMOTE - Train set: {len(X_train_scaled)} samples")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_logistic_regression(self, X_train: np.ndarray, y_train: np.ndarray,
                                 X_test: np.ndarray, y_test: np.ndarray,
                                 calibrate: bool = True) -> Dict[str, Any]:
        """
        Train logistic regression model.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target
            calibrate: Whether to calibrate probabilities
            
        Returns:
            Dictionary with model and performance metrics
        """
        print("\nTraining Logistic Regression...")
        
        # Base model
        lr = LogisticRegression(
            random_state=self.random_state,
            max_iter=1000,
            class_weight='balanced'
        )
        
        # Calibrate if requested
        if calibrate:
            model = CalibratedClassifierCV(lr, method='isotonic', cv=3)
        else:
            model = lr
        
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
        
        self.models['logistic_regression'] = model
        
        return {
            'model': model,
            'name': 'logistic_regression',
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
    
    def train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray,
                           X_test: np.ndarray, y_test: np.ndarray,
                           n_estimators: int = 100,
                           max_depth: Optional[int] = None,
                           calibrate: bool = True) -> Dict[str, Any]:
        """
        Train random forest model.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target
            n_estimators: Number of trees
            max_depth: Maximum depth of trees
            calibrate: Whether to calibrate probabilities
            
        Returns:
            Dictionary with model and performance metrics
        """
        print("\nTraining Random Forest...")
        
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=self.random_state,
            class_weight='balanced',
            n_jobs=-1
        )
        
        if calibrate:
            model = CalibratedClassifierCV(rf, method='isotonic', cv=3)
        else:
            model = rf
        
        model.fit(X_train, y_train)
        
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
        
        self.models['random_forest'] = model
        
        return {
            'model': model,
            'name': 'random_forest',
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
    
    def train_xgboost(self, X_train: np.ndarray, y_train: np.ndarray,
                     X_test: np.ndarray, y_test: np.ndarray,
                     calibrate: bool = True) -> Dict[str, Any]:
        """
        Train XGBoost model.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target
            calibrate: Whether to calibrate probabilities
            
        Returns:
            Dictionary with model and performance metrics
        """
        print("\nTraining XGBoost...")
        
        xgb = XGBClassifier(
            random_state=self.random_state,
            eval_metric='logloss',
            use_label_encoder=False,
            scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum() if (y_train == 1).sum() > 0 else 1
        )
        
        if calibrate:
            model = CalibratedClassifierCV(xgb, method='isotonic', cv=3)
        else:
            model = xgb
        
        model.fit(X_train, y_train)
        
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
        
        self.models['xgboost'] = model
        
        return {
            'model': model,
            'name': 'xgboost',
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
    
    def train_lightgbm(self, X_train: np.ndarray, y_train: np.ndarray,
                      X_test: np.ndarray, y_test: np.ndarray,
                      calibrate: bool = True) -> Dict[str, Any]:
        """
        Train LightGBM model.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target
            calibrate: Whether to calibrate probabilities
            
        Returns:
            Dictionary with model and performance metrics
        """
        print("\nTraining LightGBM...")
        
        lgbm = LGBMClassifier(
            random_state=self.random_state,
            verbose=-1,
            class_weight='balanced'
        )
        
        if calibrate:
            model = CalibratedClassifierCV(lgbm, method='isotonic', cv=3)
        else:
            model = lgbm
        
        model.fit(X_train, y_train)
        
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
        
        self.models['lightgbm'] = model
        
        return {
            'model': model,
            'name': 'lightgbm',
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
    
    def train_all_models(self, X: pd.DataFrame, y: pd.Series,
                        test_size: float = 0.2,
                        use_smote: bool = True,
                        calibrate: bool = True) -> Dict[str, Dict[str, Any]]:
        """
        Train all models and return results.
        
        Args:
            X: Features DataFrame
            y: Target Series
            test_size: Proportion of data for testing
            use_smote: Whether to use SMOTE
            calibrate: Whether to calibrate probabilities
            
        Returns:
            Dictionary with results for each model
        """
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data(
            X, y, test_size=test_size, use_smote=use_smote
        )
        
        results = {}
        
        # Train all models
        results['logistic_regression'] = self.train_logistic_regression(
            X_train, y_train, X_test, y_test, calibrate=calibrate
        )
        
        results['random_forest'] = self.train_random_forest(
            X_train, y_train, X_test, y_test, calibrate=calibrate
        )
        
        results['xgboost'] = self.train_xgboost(
            X_train, y_train, X_test, y_test, calibrate=calibrate
        )
        
        results['lightgbm'] = self.train_lightgbm(
            X_train, y_train, X_test, y_test, calibrate=calibrate
        )
        
        # Store test sets for evaluation
        for result in results.values():
            result['y_test'] = y_test
            result['X_test'] = X_test
        
        return results
    
    def save_model(self, model: Any, filepath: str):
        """Save model to disk."""
        joblib.dump(model, filepath)
        print(f"Model saved to {filepath}")
    
    def save_scaler(self, filepath: str):
        """Save scaler to disk."""
        joblib.dump(self.scaler, filepath)
        print(f"Scaler saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model from disk."""
        return joblib.load(filepath)


def load_model(filepath: str):
    """Convenience function to load a model."""
    return joblib.load(filepath)


if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.append('.')
    from data_preprocessing import DataPreprocessor
    from feature_engineering import FeatureEngineer
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    preprocessor = DataPreprocessor()
    features, target = preprocessor.preprocess(
        'data/raw/accepted_2007_to_2018Q4.csv',
        sample_size=50000
    )
    
    # Engineer features
    print("\nEngineering features...")
    engineer = FeatureEngineer()
    features_eng = engineer.engineer_features(features)
    
    # Train models
    print("\nTraining models...")
    trainer = ModelTrainer()
    results = trainer.train_all_models(features_eng, target, use_smote=True)
    
    # Save best model (to be determined by evaluation)
    trainer.save_model(trainer.models['xgboost'], 'models/xgboost_model.pkl')
    trainer.save_scaler('models/scaler.pkl')
    print("\nTraining complete!")





