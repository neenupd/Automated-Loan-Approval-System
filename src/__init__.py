"""
Automated Loan Approval System
Machine learning-based loan approval prediction system.
"""

__version__ = '1.0.0'

from .data_preprocessing import DataPreprocessor
from .feature_engineering import FeatureEngineer
from .model_training import ModelTrainer
from .evaluation import ModelEvaluator
from .decision_engine import LoanDecisionEngine
from .explainability import ModelExplainer
from .fairness_audit import FairnessAuditor

__all__ = [
    'DataPreprocessor',
    'FeatureEngineer',
    'ModelTrainer',
    'ModelEvaluator',
    'LoanDecisionEngine',
    'ModelExplainer',
    'FairnessAuditor'
]





