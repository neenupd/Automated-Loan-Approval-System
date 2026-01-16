"""
Feature Engineering Module
Creates meaningful financial ratios and transforms features for model training.
"""

import pandas as pd
import numpy as np
from typing import List, Optional
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """Engineers features from raw loan application data."""
    
    def __init__(self):
        """Initialize the feature engineer."""
        self.feature_names = []
        
    def create_debt_to_income_ratio(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate debt-to-income ratio.
        
        Args:
            df: DataFrame with 'dti' column or 'annual_inc' and total debt
            
        Returns:
            Series with DTI ratio
        """
        if 'dti' in df.columns:
            return df['dti']
        else:
            if 'annual_inc' in df.columns and 'annual_inc' in df.columns:
                return df.get('dti', pd.Series(np.nan, index=df.index))
            return pd.Series(np.nan, index=df.index)
    
    def create_loan_to_income_ratio(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate loan amount to annual income ratio.
        
        Args:
            df: DataFrame with 'loan_amnt' and 'annual_inc' columns
            
        Returns:
            Series with loan-to-income ratio
        """
        if 'loan_amnt' in df.columns and 'annual_inc' in df.columns:
            ratio = df['loan_amnt'] / (df['annual_inc'] + 1e-6)
            return ratio
        return pd.Series(np.nan, index=df.index)
    
    def create_credit_utilization(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate credit utilization ratio.
        
        Args:
            df: DataFrame with 'revol_util' column or calculated from revol_bal and revol_limit
            
        Returns:
            Series with credit utilization ratio
        """
        if 'revol_util' in df.columns:
            return df['revol_util'].replace('%', '', regex=True).astype(float)
        elif 'revol_bal' in df.columns and 'total_rev_hi_lim' in df.columns:
            utilization = df['revol_bal'] / (df['total_rev_hi_lim'] + 1e-6) * 100
            return utilization
        return pd.Series(np.nan, index=df.index)
    
    def create_credit_history_age(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate credit history age in years.
        
        Args:
            df: DataFrame with 'earliest_cr_line' or 'issue_d' column
            
        Returns:
            Series with credit history age
        """
        if 'earliest_cr_line' in df.columns:
            earliest_cr = df['earliest_cr_line'].copy()
            
            if pd.api.types.is_numeric_dtype(earliest_cr):
                return pd.Series(np.nan, index=df.index)
            
            try:
                if earliest_cr.dtype == 'object':
                    earliest_cr = pd.to_datetime(earliest_cr, format='%b-%Y', errors='coerce')
                
                if 'issue_d' in df.columns:
                    issue_d = df['issue_d'].copy()
                    if issue_d.dtype == 'object':
                        issue_d = pd.to_datetime(issue_d, format='%b-%Y', errors='coerce')
                    
                    if pd.api.types.is_datetime64_any_dtype(issue_d) and pd.api.types.is_datetime64_any_dtype(earliest_cr):
                        age = (issue_d - earliest_cr).dt.days / 365.25
                    else:
                        return pd.Series(np.nan, index=df.index)
                else:
                    if pd.api.types.is_datetime64_any_dtype(earliest_cr):
                        age = (pd.Timestamp.now() - earliest_cr).dt.days / 365.25
                    else:
                        return pd.Series(np.nan, index=df.index)
                
                return age.fillna(age.median() if len(age.dropna()) > 0 else 0)
            except (ValueError, TypeError):
                return pd.Series(np.nan, index=df.index)
        
        return pd.Series(np.nan, index=df.index)
    
    def create_fico_average(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate average FICO score from low and high range.
        
        Args:
            df: DataFrame with 'fico_range_low' and 'fico_range_high' columns
            
        Returns:
            Series with average FICO score
        """
        if 'fico_range_low' in df.columns and 'fico_range_high' in df.columns:
            return (df['fico_range_low'] + df['fico_range_high']) / 2
        elif 'fico_range_low' in df.columns:
            return df['fico_range_low']
        elif 'fico_range_high' in df.columns:
            return df['fico_range_high']
        return pd.Series(np.nan, index=df.index)
    
    def create_loan_term_months(self, df: pd.DataFrame) -> pd.Series:
        """
        Convert loan term to months if in string format.
        
        Args:
            df: DataFrame with 'term' column
            
        Returns:
            Series with loan term in months
        """
        if 'term' in df.columns:
            if df['term'].dtype == 'object':
                return df['term'].str.extract('(\d+)').astype(float)
            return df['term']
        return pd.Series(np.nan, index=df.index)
    
    def create_employment_length_years(self, df: pd.DataFrame) -> pd.Series:
        """
        Convert employment length to years.
        
        Args:
            df: DataFrame with 'emp_length' column
            
        Returns:
            Series with employment length in years
        """
        if 'emp_length' in df.columns:
            if df['emp_length'].dtype == 'object':
                emp_years = df['emp_length'].str.extract('(\d+)').astype(float)
                emp_years[df['emp_length'].str.contains('< 1', na=False)] = 0.5
                emp_years[df['emp_length'].str.contains('10\+', na=False)] = 10
                return emp_years.fillna(emp_years.median())
            return df['emp_length']
        return pd.Series(np.nan, index=df.index)
    
    def create_payment_to_income(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate monthly payment to monthly income ratio.
        
        Args:
            df: DataFrame with 'installment' and 'annual_inc' columns
            
        Returns:
            Series with payment-to-income ratio
        """
        if 'installment' in df.columns and 'annual_inc' in df.columns:
            monthly_income = df['annual_inc'] / 12
            ratio = df['installment'] / (monthly_income + 1e-6)
            return ratio
        return pd.Series(np.nan, index=df.index)
    
    def engineer_features(self, df: pd.DataFrame, 
                         exclude_original: bool = False) -> pd.DataFrame:
        """
        Apply all feature engineering transformations.
        
        Args:
            df: Input DataFrame with raw features
            exclude_original: If True, exclude original columns used for engineering
            
        Returns:
            DataFrame with engineered features
        """
        df_eng = df.copy()
        print("Engineering features...")
        
        df_eng['loan_to_income'] = self.create_loan_to_income_ratio(df_eng)
        df_eng['credit_utilization'] = self.create_credit_utilization(df_eng)
        df_eng['credit_history_age'] = self.create_credit_history_age(df_eng)
        df_eng['fico_avg'] = self.create_fico_average(df_eng)
        df_eng['loan_term_months'] = self.create_loan_term_months(df_eng)
        df_eng['emp_length_years'] = self.create_employment_length_years(df_eng)
        df_eng['payment_to_income'] = self.create_payment_to_income(df_eng)
        
        if 'dti' not in df_eng.columns:
            df_eng['dti'] = self.create_debt_to_income_ratio(df_eng)
        
        engineered_cols = [
            'loan_to_income', 'credit_utilization', 'credit_history_age',
            'fico_avg', 'loan_term_months', 'emp_length_years', 'payment_to_income', 'dti'
        ]
        
        for col in engineered_cols:
            if col in df_eng.columns:
                if df_eng[col].isnull().sum() > 0:
                    median_val = df_eng[col].median()
                    df_eng[col].fillna(median_val if not np.isnan(median_val) else 0, inplace=True)
        
        df_eng = df_eng.replace([np.inf, -np.inf], np.nan)
        numeric_cols = df_eng.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df_eng[col].isnull().sum() > 0:
                    median_val = df_eng[col].median()
                    df_eng[col].fillna(median_val if not np.isnan(median_val) else 0, inplace=True)
        
        for col in engineered_cols:
            if col in df_eng.columns:
                if df_eng[col].isnull().sum() > 0:
                    median_val = df_eng[col].median()
                    df_eng[col].fillna(median_val if not np.isnan(median_val) else 0, inplace=True)
        
        if exclude_original:
            cols_to_exclude = [
                'earliest_cr_line', 'issue_d', 'term', 'emp_length',
                'fico_range_low', 'fico_range_high', 'revol_util'
            ]
            cols_to_exclude = [col for col in cols_to_exclude if col in df_eng.columns]
            df_eng = df_eng.drop(columns=cols_to_exclude)
        
        self.feature_names = df_eng.columns.tolist()
        
        print(f"Engineered {len(engineered_cols)} new features")
        print(f"Total features: {len(df_eng.columns)}")
        
        return df_eng
    
    def select_features(self, df: pd.DataFrame, 
                       feature_list: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Select a subset of features for modeling.
        
        Args:
            df: Input DataFrame
            feature_list: List of feature names to select (None for all)
            
        Returns:
            DataFrame with selected features
        """
        if feature_list is None:
            return df
        
        available_features = [f for f in feature_list if f in df.columns]
        missing_features = [f for f in feature_list if f not in df.columns]
        
        if missing_features:
            print(f"Warning: {len(missing_features)} features not found: {missing_features[:5]}")
        
        return df[available_features]

