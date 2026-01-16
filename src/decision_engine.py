"""
Decision Engine Module
Real-time loan approval decision engine that converts model predictions into actionable decisions.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
import joblib
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class LoanDecisionEngine:
    """Real-time loan approval decision engine."""
    
    def __init__(self, model: Any, scaler: Optional[StandardScaler] = None,
                 threshold: float = 0.5, feature_names: Optional[list] = None):
        """
        Initialize the decision engine.
        
        Args:
            model: Trained ML model (must have predict_proba method)
            scaler: Optional StandardScaler for feature scaling
            threshold: Decision threshold for approval (default: 0.5)
            feature_names: List of feature names in expected order
        """
        self.model = model
        self.scaler = scaler
        self.threshold = threshold
        self.feature_names = feature_names
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probability of eligibility.
        
        Args:
            X: Feature array
            
        Returns:
            Array of probabilities
        """
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        proba = self.model.predict_proba(X)
        
        if proba.ndim > 1 and proba.shape[1] > 1:
            return proba[:, 1]
        else:
            return proba.flatten()
    
    def make_decision(self, applicant_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make approval/rejection decision for a single applicant.
        
        Args:
            applicant_data: Dictionary with applicant features
            
        Returns:
            Dictionary with decision, probability, and metadata
        """
        if isinstance(applicant_data, dict):
            df = pd.DataFrame([applicant_data])
        else:
            df = applicant_data.copy()
        
        if self.feature_names:
            for feature in self.feature_names:
                if feature not in df.columns:
                    df[feature] = np.nan
            
            df = df[self.feature_names]
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) < len(df.columns):
            numeric_cols = [col for col in numeric_cols if col in self.feature_names]
            df = df[numeric_cols]
        
        X = df.values
        probability = self.predict_proba(X)[0]
        decision = 'APPROVED' if probability >= self.threshold else 'REJECTED'
        
        result = {
            'decision': decision,
            'probability': float(probability),
            'threshold': self.threshold,
            'confidence': float(abs(probability - self.threshold))
        }
        
        return result
    
    def make_batch_decisions(self, applicants_data: pd.DataFrame) -> pd.DataFrame:
        """
        Make decisions for multiple applicants at once.
        
        Args:
            applicants_data: DataFrame with applicant features
            
        Returns:
            DataFrame with decisions and probabilities
        """
        if self.feature_names:
            for feature in self.feature_names:
                if feature not in applicants_data.columns:
                    applicants_data[feature] = np.nan
            applicants_data = applicants_data[self.feature_names]
        
        numeric_cols = applicants_data.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) < len(applicants_data.columns):
            numeric_cols = [col for col in numeric_cols if col in self.feature_names]
            applicants_data = applicants_data[numeric_cols]
        
        X = applicants_data.values
        probabilities = self.predict_proba(X)
        decisions = ['APPROVED' if prob >= self.threshold else 'REJECTED' 
                    for prob in probabilities]
        
        results = pd.DataFrame({
            'decision': decisions,
            'probability': probabilities,
            'threshold': self.threshold
        })
        
        return results
    
    def set_threshold(self, threshold: float):
        """
        Update decision threshold.
        
        Args:
            threshold: New threshold value (0-1)
        """
        if not 0 <= threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1")
        self.threshold = threshold
    
    def get_approval_rate(self, applicants_data: pd.DataFrame) -> float:
        """
        Calculate expected approval rate for a dataset.
        
        Args:
            applicants_data: DataFrame with applicant features
            
        Returns:
            Expected approval rate
        """
        results = self.make_batch_decisions(applicants_data)
        return (results['decision'] == 'APPROVED').mean()
    
    @classmethod
    def load_from_files(cls, model_path: str, scaler_path: Optional[str] = None,
                       threshold: float = 0.5, feature_names_path: Optional[str] = None):
        """
        Load decision engine from saved model files.
        
        Args:
            model_path: Path to saved model
            scaler_path: Optional path to saved scaler
            threshold: Decision threshold
            feature_names_path: Optional path to feature names
            
        Returns:
            LoanDecisionEngine instance
        """
        model = joblib.load(model_path)
        
        scaler = None
        if scaler_path:
            scaler = joblib.load(scaler_path)
        
        feature_names = None
        if feature_names_path:
            feature_names = joblib.load(feature_names_path)
        
        return cls(model, scaler, threshold, feature_names)

