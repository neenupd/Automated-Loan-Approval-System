"""
Evaluation Module
Computes comprehensive evaluation metrics for model performance.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    precision_score, recall_score, f1_score, accuracy_score,
    classification_report, confusion_matrix, roc_curve, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class ModelEvaluator:
    """Evaluates model performance using multiple metrics."""
    
    def __init__(self):
        """Initialize the evaluator."""
        self.results = {}
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                         y_pred_proba: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'auc_roc': roc_auc_score(y_true, y_pred_proba),
            'auc_pr': average_precision_score(y_true, y_pred_proba),
            'brier_score': brier_score_loss(y_true, y_pred_proba),
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0)
        }
        
        return metrics
    
    def calculate_business_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                  y_pred_proba: np.ndarray,
                                  loan_amounts: Optional[np.ndarray] = None,
                                  threshold: float = 0.5) -> Dict[str, float]:
        """
        Calculate business-oriented metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities
            loan_amounts: Loan amounts (optional, for loss calculation)
            threshold: Decision threshold
            
        Returns:
            Dictionary of business metrics
        """
        # Approval rate
        approval_rate = (y_pred == 1).mean()
        
        # Default rate among approvals
        approved_mask = y_pred == 1
        default_rate_approved = (y_true[approved_mask] == 0).mean() if approved_mask.sum() > 0 else 0
        
        metrics = {
            'approval_rate': approval_rate,
            'default_rate_approved': default_rate_approved,
            'threshold': threshold
        }
        
        # Expected loss (if loan amounts provided)
        if loan_amounts is not None:
            # Assume 100% loss on defaults
            expected_loss = (y_pred == 1) & (y_true == 0)
            total_expected_loss = loan_amounts[expected_loss].sum()
            metrics['total_expected_loss'] = total_expected_loss
            metrics['avg_expected_loss_per_loan'] = total_expected_loss / len(y_pred) if len(y_pred) > 0 else 0
        
        return metrics
    
    def evaluate_model(self, model_result: Dict[str, Any],
                      loan_amounts: Optional[np.ndarray] = None,
                      threshold: float = 0.5) -> Dict[str, Any]:
        """
        Comprehensive model evaluation.
        
        Args:
            model_result: Dictionary with model predictions and test data
            loan_amounts: Optional loan amounts for business metrics
            threshold: Decision threshold
            
        Returns:
            Dictionary with all evaluation metrics
        """
        y_test = model_result['y_test']
        y_pred = model_result['y_pred']
        y_pred_proba = model_result['y_pred_proba']
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_test, y_pred, y_pred_proba)
        business_metrics = self.calculate_business_metrics(
            y_test, y_pred, y_pred_proba, loan_amounts, threshold
        )
        
        # Combine results
        evaluation = {
            'metrics': metrics,
            'business_metrics': business_metrics,
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        return evaluation
    
    def compare_models(self, model_results: Dict[str, Dict[str, Any]],
                      loan_amounts: Optional[np.ndarray] = None) -> pd.DataFrame:
        """
        Compare multiple models and return summary DataFrame.
        
        Args:
            model_results: Dictionary of model results
            loan_amounts: Optional loan amounts for business metrics
            
        Returns:
            DataFrame with comparison metrics
        """
        comparisons = []
        
        for model_name, result in model_results.items():
            evaluation = self.evaluate_model(result, loan_amounts)
            
            comparison = {
                'model': model_name,
                **evaluation['metrics'],
                **evaluation['business_metrics']
            }
            comparisons.append(comparison)
        
        df_comparison = pd.DataFrame(comparisons)
        df_comparison = df_comparison.sort_values('auc_roc', ascending=False)
        
        return df_comparison
    
    def plot_roc_curves(self, model_results: Dict[str, Dict[str, Any]],
                       save_path: Optional[str] = None):
        """
        Plot ROC curves for all models.
        
        Args:
            model_results: Dictionary of model results
            save_path: Optional path to save figure
        """
        plt.figure(figsize=(10, 8))
        
        for model_name, result in model_results.items():
            y_test = result['y_test']
            y_pred_proba = result['y_pred_proba']
            
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            auc = roc_auc_score(y_test, y_pred_proba)
            
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.4f})', linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves Comparison', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC curves saved to {save_path}")
        
        plt.show()
    
    def plot_precision_recall_curves(self, model_results: Dict[str, Dict[str, Any]],
                                    save_path: Optional[str] = None):
        """
        Plot Precision-Recall curves for all models.
        
        Args:
            model_results: Dictionary of model results
            save_path: Optional path to save figure
        """
        plt.figure(figsize=(10, 8))
        
        for model_name, result in model_results.items():
            y_test = result['y_test']
            y_pred_proba = result['y_pred_proba']
            
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
            auc_pr = average_precision_score(y_test, y_pred_proba)
            
            plt.plot(recall, precision, label=f'{model_name} (AUC = {auc_pr:.4f})', linewidth=2)
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curves Comparison', fontsize=14, fontweight='bold')
        plt.legend(loc='lower left', fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Precision-Recall curves saved to {save_path}")
        
        plt.show()
    
    def select_best_model(self, model_results: Dict[str, Dict[str, Any]],
                         metric: str = 'auc_roc') -> Tuple[str, Dict[str, Any]]:
        """
        Select best model based on specified metric.
        
        Args:
            model_results: Dictionary of model results
            metric: Metric to use for selection (default: 'auc_roc')
            
        Returns:
            Tuple of (best_model_name, best_model_result)
        """
        best_model_name = None
        best_score = -np.inf if metric != 'brier_score' else np.inf
        
        for model_name, result in model_results.items():
            evaluation = self.evaluate_model(result)
            score = evaluation['metrics'][metric]
            
            # For brier_score, lower is better
            if metric == 'brier_score':
                if score < best_score:
                    best_score = score
                    best_model_name = model_name
            else:
                if score > best_score:
                    best_score = score
                    best_model_name = model_name
        
        return best_model_name, model_results[best_model_name]


if __name__ == "__main__":
    # Example usage
    print("ModelEvaluator module loaded successfully")
    print("Use this module with ModelTrainer to evaluate trained models")





