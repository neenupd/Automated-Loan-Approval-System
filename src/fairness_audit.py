"""
Fairness Audit Module
Assesses model fairness across demographic groups and protected attributes.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from typing import Dict, Any, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class FairnessAuditor:
    """Audits model fairness across demographic groups."""
    
    def __init__(self, protected_attributes: List[str]):
        """
        Initialize the fairness auditor.
        
        Args:
            protected_attributes: List of column names representing protected attributes
                                 (e.g., ['gender', 'race', 'age_group'])
        """
        self.protected_attributes = protected_attributes
    
    def demographic_parity(self, y_pred: np.ndarray, 
                          protected_groups: pd.Series) -> Dict[str, float]:
        """
        Calculate demographic parity (equal approval rates across groups).
        
        Args:
            y_pred: Predicted labels
            protected_groups: Series indicating group membership
            
        Returns:
            Dictionary with approval rates per group and disparity metric
        """
        approval_rates = {}
        
        for group in protected_groups.unique():
            group_mask = protected_groups == group
            approval_rate = y_pred[group_mask].mean()
            approval_rates[group] = approval_rate
        
        # Calculate disparity (max - min approval rates)
        rates = list(approval_rates.values())
        disparity = max(rates) - min(rates) if len(rates) > 0 else 0
        
        return {
            'approval_rates': approval_rates,
            'disparity': disparity,
            'disparity_ratio': max(rates) / min(rates) if min(rates) > 0 else np.inf
        }
    
    def equalized_odds(self, y_true: np.ndarray, y_pred: np.ndarray,
                      protected_groups: pd.Series) -> Dict[str, Any]:
        """
        Calculate equalized odds (equal TPR and FPR across groups).
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            protected_groups: Series indicating group membership
            
        Returns:
            Dictionary with TPR and FPR per group and disparity metrics
        """
        results = {}
        
        for group in protected_groups.unique():
            group_mask = protected_groups == group
            y_true_group = y_true[group_mask]
            y_pred_group = y_pred[group_mask]
            
            # Calculate TPR and FPR
            tn, fp, fn, tp = confusion_matrix(y_true_group, y_pred_group).ravel()
            
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            
            results[group] = {
                'tpr': tpr,
                'fpr': fpr,
                'tnr': tn / (tn + fp) if (tn + fp) > 0 else 0,
                'fnr': fn / (fn + tp) if (fn + tp) > 0 else 0
            }
        
        # Calculate disparities
        tprs = [r['tpr'] for r in results.values()]
        fprs = [r['fpr'] for r in results.values()]
        
        return {
            'group_metrics': results,
            'tpr_disparity': max(tprs) - min(tprs) if len(tprs) > 0 else 0,
            'fpr_disparity': max(fprs) - min(fprs) if len(fprs) > 0 else 0
        }
    
    def disparate_impact(self, y_pred: np.ndarray,
                        protected_groups: pd.Series,
                        reference_group: Any = None) -> Dict[str, float]:
        """
        Calculate disparate impact (4/5ths rule).
        
        Args:
            y_pred: Predicted labels
            protected_groups: Series indicating group membership
            reference_group: Reference group (if None, uses group with highest approval rate)
            
        Returns:
            Dictionary with disparate impact ratios
        """
        approval_rates = {}
        
        for group in protected_groups.unique():
            group_mask = protected_groups == group
            approval_rates[group] = y_pred[group_mask].mean()
        
        # Determine reference group
        if reference_group is None:
            reference_group = max(approval_rates, key=approval_rates.get)
        
        reference_rate = approval_rates[reference_group]
        
        # Calculate disparate impact ratios
        impact_ratios = {}
        for group, rate in approval_rates.items():
            if group != reference_group and reference_rate > 0:
                impact_ratios[group] = rate / reference_rate
            else:
                impact_ratios[group] = 1.0
        
        # Check 4/5ths rule (disparate impact < 0.8 is problematic)
        violations = {group: ratio < 0.8 for group, ratio in impact_ratios.items()}
        
        return {
            'approval_rates': approval_rates,
            'reference_group': reference_group,
            'impact_ratios': impact_ratios,
            'violations': violations,
            'has_violations': any(violations.values())
        }
    
    def audit_model(self, y_true: np.ndarray, y_pred: np.ndarray,
                   data: pd.DataFrame) -> Dict[str, Any]:
        """
        Comprehensive fairness audit.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            data: DataFrame with protected attributes
            
        Returns:
            Dictionary with all fairness metrics
        """
        audit_results = {}
        
        for attr in self.protected_attributes:
            if attr not in data.columns:
                print(f"Warning: Protected attribute '{attr}' not found in data")
                continue
            
            protected_groups = data[attr]
            
            # Demographic parity
            dp_result = self.demographic_parity(y_pred, protected_groups)
            
            # Equalized odds
            eo_result = self.equalized_odds(y_true, y_pred, protected_groups)
            
            # Disparate impact
            di_result = self.disparate_impact(y_pred, protected_groups)
            
            audit_results[attr] = {
                'demographic_parity': dp_result,
                'equalized_odds': eo_result,
                'disparate_impact': di_result
            }
        
        return audit_results
    
    def generate_fairness_report(self, audit_results: Dict[str, Any]) -> str:
        """
        Generate human-readable fairness report.
        
        Args:
            audit_results: Results from audit_model
            
        Returns:
            Formatted report string
        """
        report_lines = ["=" * 60]
        report_lines.append("FAIRNESS AUDIT REPORT")
        report_lines.append("=" * 60)
        report_lines.append("")
        
        for attr, results in audit_results.items():
            report_lines.append(f"Protected Attribute: {attr}")
            report_lines.append("-" * 60)
            
            # Demographic parity
            dp = results['demographic_parity']
            report_lines.append("\nDemographic Parity:")
            for group, rate in dp['approval_rates'].items():
                report_lines.append(f"  {group}: {rate:.4f} ({rate*100:.2f}%)")
            report_lines.append(f"  Disparity: {dp['disparity']:.4f}")
            
            # Disparate impact
            di = results['disparate_impact']
            report_lines.append("\nDisparate Impact (4/5ths Rule):")
            report_lines.append(f"  Reference Group: {di['reference_group']}")
            for group, ratio in di['impact_ratios'].items():
                status = "VIOLATION" if di['violations'][group] else "OK"
                report_lines.append(f"  {group}: {ratio:.4f} [{status}]")
            
            # Equalized odds
            eo = results['equalized_odds']
            report_lines.append("\nEqualized Odds:")
            for group, metrics in eo['group_metrics'].items():
                report_lines.append(f"  {group}:")
                report_lines.append(f"    TPR: {metrics['tpr']:.4f}")
                report_lines.append(f"    FPR: {metrics['fpr']:.4f}")
            report_lines.append(f"  TPR Disparity: {eo['tpr_disparity']:.4f}")
            report_lines.append(f"  FPR Disparity: {eo['fpr_disparity']:.4f}")
            
            report_lines.append("\n")
        
        report_lines.append("=" * 60)
        
        return "\n".join(report_lines)


if __name__ == "__main__":
    print("FairnessAuditor module loaded successfully")
    print("Use this module to audit model fairness across demographic groups")





