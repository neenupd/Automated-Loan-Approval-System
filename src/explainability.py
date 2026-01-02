"""
Explainability Module
Provides SHAP-based explanations for model predictions.
"""

import numpy as np
import pandas as pd
import shap
import joblib
from typing import Dict, Any, List, Optional
import warnings
warnings.filterwarnings('ignore')


class ModelExplainer:
    """Provides model explainability using SHAP values."""
    
    def __init__(self, model: Any, X_train: Optional[np.ndarray] = None,
                 feature_names: Optional[List[str]] = None):
        """
        Initialize the explainer.
        
        Args:
            model: Trained ML model
            X_train: Training data for SHAP (or sample for background)
            feature_names: List of feature names
        """
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names
        self.explainer = None
        self._initialize_explainer()
    
    def _initialize_explainer(self):
        """Initialize SHAP explainer based on model type."""
        if self.X_train is None:
            print("Warning: No training data provided. SHAP explainer may not work optimally.")
            return
        
        # Use sample of training data as background
        background_size = min(100, len(self.X_train))
        background = shap.sample(self.X_train, background_size)
        
        try:
            # Try TreeExplainer for tree-based models
            if hasattr(self.model, 'get_booster') or hasattr(self.model, 'tree_'):
                self.explainer = shap.TreeExplainer(self.model)
            elif hasattr(self.model, 'estimators_'):
                # Random Forest
                self.explainer = shap.TreeExplainer(self.model)
            else:
                # Use KernelExplainer for other models (slower but more general)
                self.explainer = shap.KernelExplainer(
                    self.model.predict_proba, background
                )
        except Exception as e:
            print(f"Warning: Could not initialize TreeExplainer, using KernelExplainer: {e}")
            try:
                self.explainer = shap.KernelExplainer(
                    self.model.predict_proba, background
                )
            except Exception as e2:
                print(f"Error initializing SHAP explainer: {e2}")
                self.explainer = None
    
    def explain_prediction(self, X: np.ndarray, 
                          instance_idx: int = 0) -> Dict[str, Any]:
        """
        Explain a single prediction.
        
        Args:
            X: Feature array (can be single instance or batch)
            instance_idx: Index of instance to explain (if batch provided)
            
        Returns:
            Dictionary with SHAP values and explanation
        """
        if self.explainer is None:
            raise ValueError("SHAP explainer not initialized. Provide training data.")
        
        # Handle single instance
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Get SHAP values
        shap_values = self.explainer.shap_values(X)
        
        # Handle multi-output (take positive class for binary classification)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Positive class
        
        # Extract instance
        instance_shap = shap_values[instance_idx]
        instance_features = X[instance_idx]
        
        # Get feature importance (absolute SHAP values)
        feature_importance = np.abs(instance_shap)
        
        # Create explanation dictionary
        explanation = {
            'shap_values': instance_shap.tolist(),
            'feature_values': instance_features.tolist(),
            'feature_importance': feature_importance.tolist(),
            'feature_names': self.feature_names if self.feature_names else None
        }
        
        return explanation
    
    def get_feature_importance_ranking(self, X: np.ndarray,
                                      top_n: int = 10) -> pd.DataFrame:
        """
        Get top features contributing to prediction.
        
        Args:
            X: Feature array
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance ranking
        """
        explanation = self.explain_prediction(X)
        
        shap_values = np.array(explanation['shap_values'])
        feature_names = explanation['feature_names'] or [f'Feature_{i}' for i in range(len(shap_values))]
        
        # Create DataFrame
        df = pd.DataFrame({
            'feature': feature_names,
            'shap_value': shap_values,
            'abs_shap_value': np.abs(shap_values)
        })
        
        # Sort by absolute SHAP value
        df = df.sort_values('abs_shap_value', ascending=False)
        
        return df.head(top_n)
    
    def generate_reason_codes(self, X: np.ndarray, 
                             top_n: int = 5,
                             format: str = 'dict') -> Dict[str, Any]:
        """
        Generate human-readable reason codes for a prediction.
        
        Args:
            X: Feature array
            top_n: Number of reason codes to generate
            format: Output format ('dict' or 'text')
            
        Returns:
            Dictionary or text with reason codes
        """
        explanation = self.explain_prediction(X)
        
        shap_values = np.array(explanation['shap_values'])
        feature_values = np.array(explanation['feature_values'])
        feature_names = explanation['feature_names'] or [f'Feature_{i}' for i in range(len(shap_values))]
        
        # Get top contributing features
        top_indices = np.argsort(np.abs(shap_values))[-top_n:][::-1]
        
        reason_codes = []
        for idx in top_indices:
            feature_name = feature_names[idx]
            shap_val = shap_values[idx]
            feature_val = feature_values[idx]
            
            direction = "increased" if shap_val > 0 else "decreased"
            reason_codes.append({
                'feature': feature_name,
                'value': float(feature_val),
                'contribution': float(shap_val),
                'direction': direction
            })
        
        if format == 'text':
            text_reasons = []
            for reason in reason_codes:
                text_reasons.append(
                    f"{reason['feature']}: {reason['value']:.2f} "
                    f"({reason['direction']} probability by {abs(reason['contribution']):.4f})"
                )
            return '\n'.join(text_reasons)
        
        return {'reason_codes': reason_codes}
    
    def plot_shap_summary(self, X: np.ndarray, 
                         save_path: Optional[str] = None,
                         max_display: int = 20):
        """
        Plot SHAP summary plot.
        
        Args:
            X: Feature array
            save_path: Optional path to save figure
            max_display: Maximum number of features to display
        """
        if self.explainer is None:
            raise ValueError("SHAP explainer not initialized.")
        
        shap_values = self.explainer.shap_values(X)
        
        # Handle multi-output
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        # Plot
        shap.summary_plot(shap_values, X, 
                         feature_names=self.feature_names,
                         max_display=max_display,
                         show=False)
        
        if save_path:
            import matplotlib.pyplot as plt
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"SHAP summary plot saved to {save_path}")
            plt.close()
        else:
            shap.plots.waterfall(shap_values[0])


if __name__ == "__main__":
    print("ModelExplainer module loaded successfully")
    print("Use this module to explain model predictions using SHAP values")





