"""
Utility Functions for Web3 Trading Analysis
Helper functions for data analysis, visualization, and common operations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set style for matplotlib
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class DataProfiler:
    """Comprehensive data profiling utilities"""
    
    @staticmethod
    def basic_info(df, name="Dataset"):
        """Display basic dataset information"""
        print(f"📊 **{name.upper()} - BASIC INFO**")
        print("=" * 50)
        print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
        print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        print(f"Date range: {df.columns[0] if 'date' in str(df.columns[0]).lower() else 'No date column detected'}")
        
        # Column info
        print(f"\n📋 **COLUMNS ({len(df.columns)}):**")
        for i, col in enumerate(df.columns, 1):
            dtype = str(df[col].dtype)
            non_null = df[col].count()
            null_pct = (df[col].isnull().sum() / len(df)) * 100
            print(f"{i:2d}. {col:<25} | {dtype:<12} | {non_null:>7,} non-null ({100-null_pct:5.1f}%)")
    
    @staticmethod
    def missing_values_report(df, name="Dataset"):
        """Comprehensive missing values analysis"""
        print(f"\n🔍 **{name.upper()} - MISSING VALUES ANALYSIS**")
        print("=" * 50)
        
        missing = df.isnull().sum()
        missing_pct = (missing / len(df)) * 100
        
        if missing.sum() == 0:
            print("✅ No missing values detected!")
            return
        
        missing_df = pd.DataFrame({
            'Column': missing.index,
            'Missing_Count': missing.values,
            'Missing_Percentage': missing_pct.values
        }).sort_values('Missing_Count', ascending=False)
        
        missing_df = missing_df[missing_df['Missing_Count'] > 0]
        
        for _, row in missing_df.iterrows():
            print(f"⚠️ {row['Column']:<25} | {row['Missing_Count']:>6,} missing ({row['Missing_Percentage']:5.1f}%)")
    
    @staticmethod
    def data_types_analysis(df, name="Dataset"):
        """Analyze data types and suggest optimizations"""
        print(f"\n🔧 **{name.upper()} - DATA TYPES ANALYSIS**")
        print("=" * 50)
        
        type_counts = df.dtypes.value_counts()
        for dtype, count in type_counts.items():
            print(f"📈 {str(dtype):<15}: {count} columns")
        
        # Check for potential date columns
        date_candidates = []
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['date', 'time', 'timestamp']):
                date_candidates.append(col)
        
        if date_candidates:
            print(f"\n📅 **Potential date columns:** {date_candidates}")
        
        # Check for categorical columns with high cardinality
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            print(f"\n📊 **Categorical columns analysis:**")
            for col in categorical_cols:
                unique_count = df[col].nunique()
                unique_pct = (unique_count / len(df)) * 100
                print(f"   {col:<25}: {unique_count:>6,} unique values ({unique_pct:5.1f}%)")

class TradingAnalyzer:
    """Specialized functions for trading data analysis"""
    
    @staticmethod
    def trading_summary(df):
        """Generate comprehensive trading data summary"""
        print("📈 **TRADING DATA SUMMARY**")
        print("=" * 50)
        
        # Basic trading metrics
        if 'side' in df.columns:
            side_counts = df['side'].value_counts()
            print(f"🔄 **Trade Distribution:**")
            for side, count in side_counts.items():
                pct = (count / len(df)) * 100
                print(f"   {side:<10}: {count:>8,} trades ({pct:5.1f}%)")
        
        # PnL analysis if available
        if 'closedPnL' in df.columns:
            pnl_data = pd.to_numeric(df['closedPnL'], errors='coerce')
            pnl_data = pnl_data.dropna()
            
            if len(pnl_data) > 0:
                print(f"\n💰 **PnL Analysis ({len(pnl_data):,} records):**")
                print(f"   Total PnL: ${pnl_data.sum():,.2f}")
                print(f"   Average PnL: ${pnl_data.mean():,.2f}")
                print(f"   Median PnL: ${pnl_data.median():,.2f}")
                print(f"   Profitable trades: {(pnl_data > 0).sum():,} ({(pnl_data > 0).mean()*100:.1f}%)")
                print(f"   Loss trades: {(pnl_data < 0).sum():,} ({(pnl_data < 0).mean()*100:.1f}%)")
        
        # Symbol analysis
        if 'symbol' in df.columns:
            symbol_counts = df['symbol'].value_counts().head(10)
            print(f"\n🪙 **Top 10 Trading Symbols:**")
            for symbol, count in symbol_counts.items():
                pct = (count / len(df)) * 100
                print(f"   {symbol:<15}: {count:>8,} trades ({pct:5.1f}%)")
        
        # Size analysis
        if 'size' in df.columns:
            size_data = pd.to_numeric(df['size'], errors='coerce').dropna()
            if len(size_data) > 0:
                print(f"\n📊 **Trade Size Analysis:**")
                print(f"   Total volume: {size_data.sum():,.2f}")
                print(f"   Average size: {size_data.mean():,.2f}")
                print(f"   Median size: {size_data.median():,.2f}")
                print(f"   Max size: {size_data.max():,.2f}")

class SentimentAnalyzer:
    """Functions for sentiment data analysis"""
    
    @staticmethod
    def sentiment_summary(df):
        """Analyze Fear & Greed sentiment data"""
        print("😰😤 **SENTIMENT DATA SUMMARY**")
        print("=" * 50)
        
        if 'Classification' in df.columns:
            sentiment_counts = df['Classification'].value_counts()
            print(f"📊 **Sentiment Distribution:**")
            for sentiment, count in sentiment_counts.items():
                pct = (count / len(df)) * 100
                print(f"   {sentiment:<15}: {count:>6,} days ({pct:5.1f}%)")
            
            # Date range analysis
            if 'Date' in df.columns:
                try:
                    df['Date'] = pd.to_datetime(df['Date'])
                    date_range = df['Date'].max() - df['Date'].min()
                    print(f"\n📅 **Time Period:**")
                    print(f"   Start date: {df['Date'].min().strftime('%Y-%m-%d')}")
                    print(f"   End date: {df['Date'].max().strftime('%Y-%m-%d')}")
                    print(f"   Duration: {date_range.days:,} days ({date_range.days/365:.1f} years)")
                except:
                    print(f"⚠️ Date column needs formatting")

class Visualizer:
    """Visualization utilities"""
    
    @staticmethod
    def create_missing_values_heatmap(df, title="Missing Values Heatmap"):
        """Create heatmap for missing values"""
        if df.isnull().sum().sum() == 0:
            print("✅ No missing values to visualize!")
            return None
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(df.isnull(), yticklabels=False, cbar=True, cmap='viridis')
        plt.title(title)
        plt.tight_layout()
        return plt.gcf()
    
    @staticmethod
    def plot_sentiment_distribution(df, title="Sentiment Distribution"):
        """Plot sentiment distribution over time"""
        if 'Classification' in df.columns:
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # Pie chart
            sentiment_counts = df['Classification'].value_counts()
            colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
            axes[0].pie(sentiment_counts.values, labels=sentiment_counts.index, 
                       autopct='%1.1f%%', colors=colors[:len(sentiment_counts)])
            axes[0].set_title('Overall Sentiment Distribution')
            
            # Bar chart
            sentiment_counts.plot(kind='bar', ax=axes[1], color=colors[:len(sentiment_counts)])
            axes[1].set_title('Sentiment Counts')
            axes[1].set_xlabel('Sentiment')
            axes[1].set_ylabel('Count')
            axes[1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            return fig
    
    @staticmethod
    def plot_trading_overview(df, title="Trading Data Overview"):
        """Create trading data overview plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Trade sides distribution
        if 'side' in df.columns:
            side_counts = df['side'].value_counts()
            axes[0, 0].pie(side_counts.values, labels=side_counts.index, autopct='%1.1f%%')
            axes[0, 0].set_title('Buy vs Sell Distribution')
        
        # PnL distribution
        if 'closedPnL' in df.columns:
            pnl_data = pd.to_numeric(df['closedPnL'], errors='coerce').dropna()
            if len(pnl_data) > 0:
                axes[0, 1].hist(pnl_data, bins=50, alpha=0.7, color='skyblue')
                axes[0, 1].set_title('PnL Distribution')
                axes[0, 1].set_xlabel('PnL ($)')
                axes[0, 1].set_ylabel('Frequency')
        
        # Top symbols
        if 'symbol' in df.columns:
            top_symbols = df['symbol'].value_counts().head(10)
            top_symbols.plot(kind='bar', ax=axes[1, 0])
            axes[1, 0].set_title('Top 10 Trading Symbols')
            axes[1, 0].set_xlabel('Symbol')
            axes[1, 0].set_ylabel('Trade Count')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Trade size distribution
        if 'size' in df.columns:
            size_data = pd.to_numeric(df['size'], errors='coerce').dropna()
            if len(size_data) > 0:
                axes[1, 1].hist(size_data, bins=50, alpha=0.7, color='lightgreen')
                axes[1, 1].set_title('Trade Size Distribution')
                axes[1, 1].set_xlabel('Size')
                axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        return fig

def save_figure(fig, filename, results_dir="results/figures"):
    """Save figure with proper error handling"""
    from pathlib import Path
    
    # Create results directory if it doesn't exist
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    
    filepath = Path(results_dir) / filename
    try:
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✅ Figure saved: {filepath}")
        return filepath
    except Exception as e:
        print(f"❌ Error saving figure: {e}")
        return None

def display_progress(current_step, total_steps, description=""):
    """Display progress bar for long operations"""
    from tqdm import tqdm
    import time
    
    progress = (current_step / total_steps) * 100
    bar_length = 30
    filled_length = int(bar_length * current_step // total_steps)
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    
    print(f'\r🔄 Progress: |{bar}| {progress:.1f}% {description}', end='')
    if current_step == total_steps:
        print(" ✅ Complete!")

# Quick data validation functions
def validate_data_quality(df, name="Dataset"):
    """Quick data quality check"""
    print(f"🔍 **DATA QUALITY CHECK - {name.upper()}**")
    print("=" * 50)
    
    issues = []
    
    # Check for duplicates
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        issues.append(f"⚠️ {duplicates:,} duplicate rows")
    
    # Check for missing values
    missing = df.isnull().sum().sum()
    if missing > 0:
        issues.append(f"⚠️ {missing:,} missing values")
    
    # Check for infinite values in numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    infinite_count = 0
    for col in numeric_cols:
        infinite_count += np.isinf(df[col]).sum()
    if infinite_count > 0:
        issues.append(f"⚠️ {infinite_count:,} infinite values")
    
    if not issues:
        print("✅ Data quality looks good!")
    else:
        print("Issues found:")
        for issue in issues:
            print(f"   {issue}")
    
    return len(issues) == 0

if __name__ == "__main__":
    print("🛠️ **Web3 Trading Analysis - Utilities Module Loaded**")
    print("Available classes: DataProfiler, TradingAnalyzer, SentimentAnalyzer, Visualizer")
