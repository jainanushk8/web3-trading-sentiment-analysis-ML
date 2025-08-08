"""
Data Preprocessing Module for Web3 Trading Analysis
Handles timestamp conversion, data cleaning, and standardization
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone
import pytz
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class TimestampProcessor:
    """Handle timestamp conversion and alignment"""
    
    def __init__(self):
        self.ist_tz = pytz.timezone('Asia/Kolkata')
        self.utc_tz = pytz.UTC
        
    def convert_ist_to_utc(self, timestamp_str: str) -> pd.Timestamp:
        """Convert IST timestamp string to UTC datetime"""
        try:
            # Parse IST timestamp: "02-12-2024 22:50"
            dt = pd.to_datetime(timestamp_str, format='%d-%m-%Y %H:%M')
            # Localize to IST
            dt_ist = self.ist_tz.localize(dt)
            # Convert to UTC
            dt_utc = dt_ist.astimezone(self.utc_tz)
            return dt_utc
        except Exception as e:
            print(f"⚠️ Error converting timestamp '{timestamp_str}': {e}")
            return pd.NaT
    
    def convert_unix_to_utc(self, unix_timestamp: float) -> pd.Timestamp:
        """Convert Unix timestamp to UTC datetime"""
        try:
            # Handle milliseconds vs seconds
            if unix_timestamp > 1e12:  # Milliseconds
                unix_timestamp = unix_timestamp / 1000
            return pd.to_datetime(unix_timestamp, unit='s', utc=True)
        except Exception as e:
            print(f"⚠️ Error converting unix timestamp '{unix_timestamp}': {e}")
            return pd.NaT
    
    def extract_date_only(self, timestamp: pd.Timestamp) -> str:
        """Extract date string in YYYY-MM-DD format"""
        try:
            return timestamp.strftime('%Y-%m-%d')
        except:
            return None
    
    def standardize_timestamps(self, df: pd.DataFrame, timestamp_columns: List[str]) -> pd.DataFrame:
        """Standardize multiple timestamp columns to UTC"""
        df_processed = df.copy()
        
        for col in timestamp_columns:
            if col not in df.columns:
                print(f"⚠️ Column '{col}' not found in dataset")
                continue
                
            print(f"🔄 Processing timestamp column: {col}")
            
            if col == 'Timestamp IST':
                # Convert IST string timestamps
                df_processed[f'{col}_UTC'] = df[col].apply(self.convert_ist_to_utc)
            elif col == 'Timestamp':
                # Convert Unix timestamps
                df_processed[f'{col}_UTC'] = df[col].apply(self.convert_unix_to_utc)
            elif col == 'timestamp':
                # Handle sentiment data unix timestamp
                df_processed[f'{col}_UTC'] = df[col].apply(self.convert_unix_to_utc)
            
        return df_processed

class DataCleaner:
    """Data cleaning and standardization utilities"""
    
    @staticmethod
    def clean_numeric_columns(df: pd.DataFrame, numeric_columns: List[str]) -> pd.DataFrame:
        """Clean and standardize numeric columns"""
        df_cleaned = df.copy()
        
        for col in numeric_columns:
            if col not in df.columns:
                continue
                
            # Convert to numeric, handling errors
            df_cleaned[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Log conversion results
            original_count = df[col].count()
            new_count = df_cleaned[col].count()
            if original_count != new_count:
                print(f"⚠️ {col}: {original_count - new_count} values couldn't be converted to numeric")
        
        return df_cleaned
    
    @staticmethod
    def standardize_categorical_columns(df: pd.DataFrame, categorical_columns: Dict[str, str]) -> pd.DataFrame:
        """Standardize categorical columns"""
        df_cleaned = df.copy()
        
        for col, standard_case in categorical_columns.items():
            if col not in df.columns:
                continue
                
            if standard_case.lower() == 'upper':
                df_cleaned[col] = df[col].astype(str).str.upper()
            elif standard_case.lower() == 'lower':
                df_cleaned[col] = df[col].astype(str).str.lower()
            elif standard_case.lower() == 'title':
                df_cleaned[col] = df[col].astype(str).str.title()
        
        return df_cleaned
    
    @staticmethod
    def remove_duplicates(df: pd.DataFrame, subset_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Remove duplicate rows"""
        original_count = len(df)
        
        if subset_columns:
            df_cleaned = df.drop_duplicates(subset=subset_columns)
        else:
            df_cleaned = df.drop_duplicates()
            
        removed_count = original_count - len(df_cleaned)
        if removed_count > 0:
            print(f"🧹 Removed {removed_count:,} duplicate rows")
        else:
            print("✅ No duplicates found")
            
        return df_cleaned
    
    @staticmethod
    def handle_outliers(df: pd.DataFrame, column: str, method: str = 'iqr', multiplier: float = 1.5) -> pd.DataFrame:
        """Handle outliers in numeric columns"""
        if column not in df.columns:
            print(f"⚠️ Column '{column}' not found")
            return df
            
        df_cleaned = df.copy()
        data = df[column].dropna()
        
        if method.lower() == 'iqr':
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
            # Count outliers
            outliers = ((data < lower_bound) | (data > upper_bound)).sum()
            print(f"📊 {column}: {outliers:,} outliers detected (>{multiplier}×IQR)")
            
            # Option to cap outliers instead of removing
            df_cleaned[f'{column}_outlier_flag'] = ((df[column] < lower_bound) | 
                                                   (df[column] > upper_bound))
        
        return df_cleaned

class DataValidator:
    """Data validation utilities"""
    
    @staticmethod
    def validate_date_ranges(df: pd.DataFrame, date_column: str, 
                           expected_start: str, expected_end: str) -> Dict[str, any]:
        """Validate date ranges in dataset"""
        if date_column not in df.columns:
            return {"error": f"Column '{date_column}' not found"}
        
        dates = pd.to_datetime(df[date_column])
        actual_start = dates.min()
        actual_end = dates.max()
        
        expected_start_dt = pd.to_datetime(expected_start)
        expected_end_dt = pd.to_datetime(expected_end)
        
        validation_results = {
            "actual_start": actual_start,
            "actual_end": actual_end,
            "expected_start": expected_start_dt,
            "expected_end": expected_end_dt,
            "start_match": actual_start.date() == expected_start_dt.date(),
            "end_match": actual_end.date() == expected_end_dt.date(),
            "total_days": (actual_end - actual_start).days,
            "missing_dates": []
        }
        
        # Check for missing dates
        expected_date_range = pd.date_range(start=actual_start.date(), 
                                          end=actual_end.date(), freq='D')
        actual_dates = set(dates.dt.date)
        expected_dates = set(expected_date_range.date)
        missing_dates = expected_dates - actual_dates
        
        validation_results["missing_dates"] = list(missing_dates)
        validation_results["missing_count"] = len(missing_dates)
        
        return validation_results
    
    @staticmethod
    def validate_data_consistency(df: pd.DataFrame, column_checks: Dict[str, Dict]) -> Dict[str, any]:
        """Validate data consistency across columns"""
        validation_results = {}
        
        for column, checks in column_checks.items():
            if column not in df.columns:
                validation_results[column] = {"error": "Column not found"}
                continue
                
            column_results = {}
            
            # Check data type
            if 'expected_type' in checks:
                expected_type = checks['expected_type']
                actual_type = df[column].dtype
                column_results['type_check'] = str(actual_type) == expected_type
            
            # Check value ranges
            if 'min_value' in checks and 'max_value' in checks:
                numeric_data = pd.to_numeric(df[column], errors='coerce').dropna()
                if len(numeric_data) > 0:
                    column_results['range_check'] = {
                        'min_valid': numeric_data.min() >= checks['min_value'],
                        'max_valid': numeric_data.max() <= checks['max_value'],
                        'actual_min': numeric_data.min(),
                        'actual_max': numeric_data.max()
                    }
            
            # Check allowed values
            if 'allowed_values' in checks:
                unique_values = set(df[column].unique())
                allowed_values = set(checks['allowed_values'])
                column_results['value_check'] = {
                    'all_valid': unique_values.issubset(allowed_values),
                    'invalid_values': list(unique_values - allowed_values),
                    'unique_count': len(unique_values)
                }
            
            validation_results[column] = column_results
        
        return validation_results

class WebTradingPreprocessor:
    """Main preprocessing class for Web3 trading data"""
    
    def __init__(self):
        self.timestamp_processor = TimestampProcessor()
        self.data_cleaner = DataCleaner()
        self.data_validator = DataValidator()
        
    def preprocess_sentiment_data(self, sentiment_df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess Fear & Greed Index sentiment data"""
        print("😰😤 **PREPROCESSING SENTIMENT DATA**")
        print("=" * 50)
        
        df_processed = sentiment_df.copy()
        
        # 1. Standardize timestamps
        timestamp_columns = ['timestamp'] if 'timestamp' in df_processed.columns else []
        if timestamp_columns:
            df_processed = self.timestamp_processor.standardize_timestamps(
                df_processed, timestamp_columns)
        
        # 2. Ensure date column is proper datetime
        if 'date' in df_processed.columns:
            df_processed['date_parsed'] = pd.to_datetime(df_processed['date'])
            df_processed['date_standardized'] = df_processed['date_parsed'].dt.strftime('%Y-%m-%d')
        
        # 3. Standardize classification column
        if 'classification' in df_processed.columns:
            df_processed = self.data_cleaner.standardize_categorical_columns(
                df_processed, {'classification': 'title'})
        
        # 4. Clean numeric columns
        numeric_cols = ['value', 'timestamp']
        available_numeric_cols = [col for col in numeric_cols if col in df_processed.columns]
        if available_numeric_cols:
            df_processed = self.data_cleaner.clean_numeric_columns(
                df_processed, available_numeric_cols)
        
        # 5. Remove duplicates
        df_processed = self.data_cleaner.remove_duplicates(df_processed, ['date'])
        
        print(f"✅ Sentiment data preprocessing complete: {df_processed.shape}")
        return df_processed
    
    def preprocess_trading_data(self, trading_df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess Hyperliquid trading data"""
        print("\n📈 **PREPROCESSING TRADING DATA**")
        print("=" * 50)
        
        df_processed = trading_df.copy()
        
        # 1. Standardize timestamps
        timestamp_columns = ['Timestamp IST', 'Timestamp']
        df_processed = self.timestamp_processor.standardize_timestamps(
            df_processed, timestamp_columns)
        
        # 2. Extract trading date for sentiment alignment
        if 'Timestamp IST_UTC' in df_processed.columns:
            df_processed['trading_date'] = df_processed['Timestamp IST_UTC'].dt.strftime('%Y-%m-%d')
        
        # 3. Clean numeric columns
        numeric_cols = ['Execution Price', 'Size Tokens', 'Size USD', 'Start Position', 
                       'Closed PnL', 'Fee', 'Trade ID', 'Timestamp']
        available_numeric_cols = [col for col in numeric_cols if col in df_processed.columns]
        df_processed = self.data_cleaner.clean_numeric_columns(
            df_processed, available_numeric_cols)
        
        # 4. Standardize categorical columns
        categorical_cols = {'Side': 'upper', 'Direction': 'title'}
        df_processed = self.data_cleaner.standardize_categorical_columns(
            df_processed, categorical_cols)
        
        # 5. Handle outliers in key columns
        key_columns = ['Size USD', 'Closed PnL']
        for col in key_columns:
            if col in df_processed.columns:
                df_processed = self.data_cleaner.handle_outliers(df_processed, col)
        
        # 6. Remove duplicates based on key identifiers
        duplicate_check_cols = ['Account', 'Trade ID'] if 'Trade ID' in df_processed.columns else ['Account']
        df_processed = self.data_cleaner.remove_duplicates(df_processed, duplicate_check_cols)
        
        print(f"✅ Trading data preprocessing complete: {df_processed.shape}")
        return df_processed
    
    def validate_processed_data(self, sentiment_df: pd.DataFrame, 
                              trading_df: pd.DataFrame) -> Dict[str, any]:
        """Validate both processed datasets"""
        print("\n🔍 **VALIDATING PROCESSED DATA**")
        print("=" * 50)
        
        validation_results = {}
        
        # Validate sentiment data
        if 'date_standardized' in sentiment_df.columns:
            sentiment_validation = self.data_validator.validate_date_ranges(
                sentiment_df, 'date_standardized', '2018-02-01', '2025-08-08')
            validation_results['sentiment'] = sentiment_validation
        
        # Validate trading data  
        if 'trading_date' in trading_df.columns:
            trading_validation = self.data_validator.validate_date_ranges(
                trading_df, 'trading_date', '2024-01-01', '2025-08-08')
            validation_results['trading'] = trading_validation
        
        # Data consistency checks
        sentiment_checks = {
            'classification': {
                'allowed_values': ['Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed']
            },
            'value': {
                'min_value': 0,
                'max_value': 100
            }
        }
        
        trading_checks = {
            'Side': {
                'allowed_values': ['BUY', 'SELL']
            },
            'Closed PnL': {
                'min_value': -1000000,  # Reasonable loss limit
                'max_value': 1000000    # Reasonable profit limit
            }
        }
        
        validation_results['sentiment_consistency'] = self.data_validator.validate_data_consistency(
            sentiment_df, sentiment_checks)
        validation_results['trading_consistency'] = self.data_validator.validate_data_consistency(
            trading_df, trading_checks)
        
        return validation_results
    
    def create_alignment_summary(self, sentiment_df: pd.DataFrame, 
                               trading_df: pd.DataFrame) -> Dict[str, any]:
        """Create summary of data alignment for merging"""
        print("\n📊 **DATA ALIGNMENT SUMMARY**")
        print("=" * 50)
        
        summary = {}
        
        # Date range analysis
        if 'date_standardized' in sentiment_df.columns:
            sentiment_dates = pd.to_datetime(sentiment_df['date_standardized'])
            summary['sentiment_date_range'] = {
                'start': sentiment_dates.min().strftime('%Y-%m-%d'),
                'end': sentiment_dates.max().strftime('%Y-%m-%d'),
                'total_days': len(sentiment_df),
                'unique_dates': sentiment_dates.nunique()
            }
        
        if 'trading_date' in trading_df.columns:
            trading_dates = pd.to_datetime(trading_df['trading_date'])
            summary['trading_date_range'] = {
                'start': trading_dates.min().strftime('%Y-%m-%d'),
                'end': trading_dates.max().strftime('%Y-%m-%d'),
                'total_trades': len(trading_df),
                'unique_trading_dates': trading_dates.nunique()
            }
        
        # Overlap analysis
        if 'date_standardized' in sentiment_df.columns and 'trading_date' in trading_df.columns:
            sentiment_dates_set = set(sentiment_df['date_standardized'])
            trading_dates_set = set(trading_df['trading_date'])
            
            overlap_dates = sentiment_dates_set.intersection(trading_dates_set)
            summary['overlap_analysis'] = {
                'overlapping_dates': len(overlap_dates),
                'sentiment_only_dates': len(sentiment_dates_set - trading_dates_set),
                'trading_only_dates': len(trading_dates_set - sentiment_dates_set),
                'overlap_percentage': (len(overlap_dates) / len(sentiment_dates_set)) * 100
            }
        
        return summary

if __name__ == "__main__":
    print("🛠️ **Web3 Trading Analysis - Preprocessing Module Loaded**")
    print("Available classes: TimestampProcessor, DataCleaner, DataValidator, WebTradingPreprocessor")
