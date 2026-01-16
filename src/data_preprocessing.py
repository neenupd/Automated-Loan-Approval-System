"""
Data Preprocessing Module
Handles data cleaning, missing value imputation, and initial preprocessing
for the Lending Club loan data.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from typing import Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """Preprocesses raw Lending Club loan data."""
    
    def __init__(self, target_column: str = 'loan_status'):
        """
        Initialize the preprocessor.
        
        Args:
            target_column: Name of the target column (default: 'loan_status')
        """
        self.target_column = target_column
        self.label_encoders = {}
        self.feature_columns = None
        self.categorical_columns = None
        
    def load_data(self, file_path: str, sample_size: Optional[int] = None) -> pd.DataFrame:
        """
        Load data from CSV file, optionally sampling during load to save memory.
        
        Args:
            file_path: Path to the CSV file
            sample_size: Number of rows to sample (None for all data)
            
        Returns:
            DataFrame containing the raw data
        """
        print(f"Loading data from {file_path}...")
        
        if sample_size:
            print(f"Sampling {sample_size} rows during load...")
            total_rows = sum(1 for _ in open(file_path, 'r', encoding='utf-8', errors='ignore')) - 1
            print(f"Total rows in file: {total_rows:,}")
            
            if sample_size >= total_rows:
                df = pd.read_csv(file_path, low_memory=False)
            else:
                skip = sorted(np.random.RandomState(42).choice(
                    range(1, total_rows + 1), 
                    size=total_rows - sample_size, 
                    replace=False
                ))
                df = pd.read_csv(file_path, skiprows=skip, low_memory=False)
        else:
            df = pd.read_csv(file_path, low_memory=False)
        
        print(f"Loaded {len(df)} records with {len(df.columns)} columns")
        return df
    
    def remove_post_origination_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove features that are only available after loan origination
        to avoid target leakage.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with post-origination features removed
        """
        post_origination = [
            'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp',
            'total_rec_int', 'total_rec_late_fee', 'recoveries',
            'collection_recovery_fee', 'last_pymnt_d', 'last_pymnt_amnt',
            'last_credit_pull_d', 'next_pymnt_d', 'out_prncp',
            'out_prncp_inv', 'total_rec_late_fee', 'collection_recovery_fee',
            'last_fico_range_high', 'last_fico_range_low'
        ]
        
        existing_post_orig = [col for col in post_origination if col in df.columns]
        if existing_post_orig:
            df = df.drop(columns=existing_post_orig)
            print(f"Removed {len(existing_post_orig)} post-origination features")
        
        return df
    
    def create_target_variable(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create binary target variable from loan_status.
        Eligible (1): Fully Paid
        Ineligible (0): Charged Off, Default, Late payments
        
        Args:
            df: Input DataFrame with loan_status column
            
        Returns:
            DataFrame with binary target column 'eligible'
        """
        if self.target_column not in df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in data")
        
        df['eligible'] = (df[self.target_column] == 'Fully Paid').astype(int)
        valid_statuses = ['Fully Paid', 'Charged Off', 'Default']
        df = df[df[self.target_column].isin(valid_statuses)].copy()
        
        print(f"Target distribution: {df['eligible'].value_counts().to_dict()}")
        print(f"Eligibility rate: {df['eligible'].mean():.4f}")
        
        return df
    
    def handle_missing_values(self, df: pd.DataFrame, 
                             missing_threshold: float = 0.5) -> pd.DataFrame:
        """
        Handle missing values by removing columns with high missingness
        and imputing remaining missing values.
        
        Args:
            df: Input DataFrame
            missing_threshold: Threshold for dropping columns (default: 0.5)
            
        Returns:
            DataFrame with handled missing values
        """
        missing_ratio = df.isnull().sum() / len(df)
        cols_to_drop = missing_ratio[missing_ratio > missing_threshold].index.tolist()
        
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            print(f"Dropped {len(cols_to_drop)} columns with >{missing_threshold*100}% missing values")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
        
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown', inplace=True)
        
        print(f"Remaining missing values: {df.isnull().sum().sum()}")
        
        return df
    
    def encode_categorical_variables(self, df: pd.DataFrame, 
                                    columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Encode categorical variables using label encoding.
        Excludes date columns to preserve them for feature engineering.
        
        Args:
            df: Input DataFrame
            columns: List of columns to encode (if None, encodes all object columns)
            
        Returns:
            DataFrame with encoded categorical variables
        """
        df_encoded = df.copy()
        date_columns = ['earliest_cr_line', 'issue_d', 'last_pymnt_d', 'next_pymnt_d', 
                       'last_credit_pull_d', 'hardship_start_date', 'hardship_end_date',
                       'payment_plan_start_date', 'debt_settlement_flag_date', 'settlement_date']
        
        if columns is None:
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            if self.target_column in categorical_cols:
                categorical_cols.remove(self.target_column)
            categorical_cols = [col for col in categorical_cols if col not in date_columns]
        else:
            categorical_cols = [col for col in columns if col not in date_columns]
        
        self.categorical_columns = categorical_cols
        
        for col in categorical_cols:
            if col in df_encoded.columns:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                self.label_encoders[col] = le
        
        print(f"Encoded {len(categorical_cols)} categorical variables (excluded {len([c for c in date_columns if c in df.columns])} date columns)")
        
        return df_encoded
    
    def preprocess(self, file_path: str, 
                   sample_size: Optional[int] = None,
                   missing_threshold: float = 0.5) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Complete preprocessing pipeline.
        
        Args:
            file_path: Path to raw data CSV
            sample_size: Number of samples to use (None for all)
            missing_threshold: Threshold for dropping columns with missing values
            
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        df = self.load_data(file_path, sample_size=sample_size)
        df = self.remove_post_origination_features(df)
        df = self.create_target_variable(df)
        df = self.handle_missing_values(df, missing_threshold=missing_threshold)
        
        target = df['eligible']
        features = df.drop(columns=['eligible', self.target_column], errors='ignore')
        features = self.encode_categorical_variables(features)
        self.feature_columns = features.columns.tolist()
        
        print(f"\nPreprocessing complete!")
        print(f"Final dataset: {len(features)} samples, {len(features.columns)} features")
        print(f"Target distribution: {target.value_counts().to_dict()}")
        
        return features, target

