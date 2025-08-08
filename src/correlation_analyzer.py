"""
Correlation Analysis Module for Web3 Trading Analysis
Handles sentiment-performance correlation with robust error handling
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class SentimentPerformanceCorrelation:
    """Analyze correlations between sentiment and trading performance"""
    
    def __init__(self):
        self.correlation_results = {}
        
    def validate_required_columns(self, df: pd.DataFrame, required_cols: List[str]) -> Tuple[bool, List[str]]:
        """Safely validate required columns exist"""
        missing_cols = []
        available_cols = list(df.columns)
        
        for col in required_cols:
            if col not in available_cols:
                missing_cols.append(col)
        
        return len(missing_cols) == 0, missing_cols
    
    def safe_correlation_analysis(self, df: pd.DataFrame) -> Dict[str, any]:
        """Perform correlation analysis with error handling"""
        print("🔗 **SENTIMENT-PERFORMANCE CORRELATION ANALYSIS**")
        print("=" * 60)
        
        # Check available columns first
        print("🔍 **Available Columns Check:**")
        available_cols = list(df.columns)
        print(f"   Total columns: {len(available_cols)}")
        
        # Define what we're looking for
        performance_cols = [col for col in available_cols if any(x in col.lower() for x in ['pnl', 'roi', 'profit', 'return'])]
        sentiment_cols = [col for col in available_cols if 'sentiment' in col.lower()]
        regime_cols = [col for col in available_cols if 'regime' in col.lower()]
        
        print(f"   Performance columns found: {len(performance_cols)} - {performance_cols}")
        print(f"   Sentiment columns found: {len(sentiment_cols)} - {sentiment_cols}")
        print(f"   Regime columns found: {len(regime_cols)} - {regime_cols}")
        
        correlation_results = {
            'available_columns': available_cols,
            'performance_columns': performance_cols,
            'sentiment_columns': sentiment_cols,
            'regime_columns': regime_cols,
            'correlations': {},
            'statistical_tests': {}
        }
        
        # Perform correlations only on available numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        print(f"\n📊 **Numeric Columns for Correlation: {len(numeric_cols)}**")
        
        if len(numeric_cols) >= 2:
            try:
                correlation_matrix = df[numeric_cols].corr()
                correlation_results['correlation_matrix'] = correlation_matrix
                
                # Find strongest correlations
                strong_correlations = []
                for i, col1 in enumerate(correlation_matrix.columns):
                    for j, col2 in enumerate(correlation_matrix.columns):
                        if i < j:  # Avoid duplicates
                            corr_val = correlation_matrix.iloc[i, j]
                            if not pd.isna(corr_val) and abs(corr_val) > 0.1:  # Threshold for meaningful correlation
                                strong_correlations.append((col1, col2, corr_val))
                
                # Sort by absolute correlation strength
                strong_correlations.sort(key=lambda x: abs(x[2]), reverse=True)
                correlation_results['strong_correlations'] = strong_correlations[:10]  # Top 10
                
                print(f"✅ Found {len(strong_correlations)} meaningful correlations")
                
            except Exception as e:
                print(f"⚠️ Correlation calculation error: {e}")
                correlation_results['correlation_error'] = str(e)
        
        return correlation_results
    
    def analyze_sentiment_regime_performance(self, df: pd.DataFrame) -> Dict[str, any]:
        """Analyze performance across different sentiment regimes"""
        print("\n🎯 **PERFORMANCE BY SENTIMENT REGIME ANALYSIS**")
        print("=" * 60)
        
        # Check for required columns safely
        required_cols = ['market_regime', 'total_pnl']
        has_required, missing = self.validate_required_columns(df, required_cols)
        
        if not has_required:
            print(f"⚠️ Missing required columns: {missing}")
            # Try alternative column names
            regime_alternatives = [col for col in df.columns if 'regime' in col.lower()]
            pnl_alternatives = [col for col in df.columns if 'pnl' in col.lower()]
            
            print(f"   Available regime columns: {regime_alternatives}")
            print(f"   Available PnL columns: {pnl_alternatives}")
            
            if regime_alternatives and pnl_alternatives:
                regime_col = regime_alternatives[0]
                pnl_col = pnl_alternatives[0]
                print(f"   Using: {regime_col} and {pnl_col}")
            else:
                return {"error": "No suitable columns found for regime analysis"}
        else:
            regime_col = 'market_regime'
            pnl_col = 'total_pnl'
        
        try:
            # Performance by regime
            regime_performance = df.groupby(regime_col)[pnl_col].agg([
                'count', 'mean', 'median', 'std', 'sum'
            ]).round(4)
            
            # Calculate additional metrics
            regime_performance['win_rate'] = df.groupby(regime_col).apply(
                lambda x: (x[pnl_col] > 0).mean()
            ).round(4)
            
            # Statistical significance tests
            regimes = df[regime_col].unique()
            regime_comparisons = {}
            
            if len(regimes) >= 2:
                for i, regime1 in enumerate(regimes):
                    for regime2 in regimes[i+1:]:
                        group1 = df[df[regime_col] == regime1][pnl_col].dropna()
                        group2 = df[df[regime_col] == regime2][pnl_col].dropna()
                        
                        if len(group1) > 0 and len(group2) > 0:
                            try:
                                # T-test for difference in means
                                t_stat, p_value = stats.ttest_ind(group1, group2)
                                regime_comparisons[f"{regime1}_vs_{regime2}"] = {
                                    't_statistic': round(t_stat, 4),
                                    'p_value': round(p_value, 4),
                                    'significant': p_value < 0.05
                                }
                            except Exception as e:
                                print(f"⚠️ T-test error for {regime1} vs {regime2}: {e}")
            
            return {
                'regime_performance': regime_performance,
                'regime_comparisons': regime_comparisons,
                'regime_column_used': regime_col,
                'pnl_column_used': pnl_col
            }
            
        except Exception as e:
            print(f"❌ Regime analysis error: {e}")
            return {"error": str(e)}
    
    def trader_behavior_by_sentiment(self, df: pd.DataFrame) -> Dict[str, any]:
        """Analyze how trader behavior changes with sentiment"""
        print("\n👥 **TRADER BEHAVIOR BY SENTIMENT ANALYSIS**")
        print("=" * 60)
        
        # Safely check for required columns
        behavior_analysis = {}
        
        # Check for sentiment score
        sentiment_cols = [col for col in df.columns if 'sentiment' in col.lower() and 'score' in col.lower()]
        if not sentiment_cols:
            sentiment_cols = [col for col in df.columns if 'sentiment' in col.lower()]
        
        if not sentiment_cols:
            print("⚠️ No sentiment columns found")
            return {"error": "No sentiment columns available"}
        
        sentiment_col = sentiment_cols[0]
        print(f"   Using sentiment column: {sentiment_col}")
        
        # Check for trader identifier
        trader_cols = [col for col in df.columns if col.lower() in ['account', 'trader', 'user']]
        if not trader_cols:
            print("⚠️ No trader identifier column found")
            return {"error": "No trader identifier available"}
        
        trader_col = trader_cols[0]
        print(f"   Using trader column: {trader_col}")
        
        try:
            # Trader activity by sentiment
            trader_sentiment_activity = df.groupby([trader_col, sentiment_col]).size().reset_index(name='trade_days')
            
            # Average trades per sentiment level - FIXED VERSION
            if 'total_pnl' in df.columns:
                avg_activity_by_sentiment = df.groupby(sentiment_col).agg({
                    trader_col: 'nunique',
                    'total_pnl': ['mean', 'sum', 'count']
                })
            else:
                avg_activity_by_sentiment = df.groupby(sentiment_col).agg({
                    trader_col: ['nunique', 'count']
                })
            
            behavior_analysis = {
                'trader_sentiment_activity': trader_sentiment_activity,
                'avg_activity_by_sentiment': avg_activity_by_sentiment,
                'sentiment_column_used': sentiment_col,
                'trader_column_used': trader_col
            }
            
            print(f"✅ Analyzed behavior for {df[trader_col].nunique()} traders across sentiment levels")
            
        except Exception as e:
            print(f"❌ Trader behavior analysis error: {e}")
            behavior_analysis = {"error": str(e)}
        
        return behavior_analysis


class CorrelationVisualizer:
    """Create visualizations for correlation analysis"""
    
    @staticmethod
    def safe_correlation_heatmap(correlation_matrix: pd.DataFrame, title: str = "Correlation Heatmap") -> plt.Figure:
        """Create correlation heatmap with error handling"""
        try:
            if correlation_matrix.empty:
                print("⚠️ Empty correlation matrix, cannot create heatmap")
                return None
            
            # Limit to reasonable size for visualization
            if len(correlation_matrix.columns) > 15:
                # Select most interesting columns
                numeric_variance = correlation_matrix.var().sort_values(ascending=False)
                top_cols = numeric_variance.head(15).index
                correlation_matrix = correlation_matrix.loc[top_cols, top_cols]
                title += " (Top 15 Features)"
            
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Create heatmap
            sns.heatmap(correlation_matrix, 
                       annot=True, 
                       cmap='coolwarm', 
                       center=0,
                       square=True,
                       fmt='.3f',
                       cbar_kws={"shrink": .8},
                       ax=ax)
            
            ax.set_title(title, fontsize=16, fontweight='bold')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            
            return fig
            
        except Exception as e:
            print(f"❌ Heatmap creation error: {e}")
            return None
    
    @staticmethod
    def safe_regime_performance_plot(regime_data: pd.DataFrame, regime_col: str, pnl_col: str) -> plt.Figure:
        """Create performance by regime visualization"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Plot 1: Average PnL by regime
            regime_means = regime_data.groupby(regime_col)[pnl_col].mean()
            axes[0, 0].bar(regime_means.index, regime_means.values, 
                          color=['red' if x < 0 else 'green' for x in regime_means.values])
            axes[0, 0].set_title('Average PnL by Market Regime')
            axes[0, 0].set_ylabel('Average PnL ($)')
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # Plot 2: PnL distribution by regime
            regime_data.boxplot(column=pnl_col, by=regime_col, ax=axes[0, 1])
            axes[0, 1].set_title('PnL Distribution by Market Regime')
            axes[0, 1].set_ylabel('PnL ($)')
            
            # Plot 3: Win rate by regime
            win_rates = regime_data.groupby(regime_col).apply(lambda x: (x[pnl_col] > 0).mean())
            axes[1, 0].bar(win_rates.index, win_rates.values, color='skyblue')
            axes[1, 0].set_title('Win Rate by Market Regime')
            axes[1, 0].set_ylabel('Win Rate')
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            # Plot 4: Trading activity by regime
            activity_counts = regime_data[regime_col].value_counts()
            axes[1, 1].pie(activity_counts.values, labels=activity_counts.index, autopct='%1.1f%%')
            axes[1, 1].set_title('Trading Activity Distribution by Regime')
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            print(f"❌ Regime plot creation error: {e}")
            return None

class CorrelationAnalyzer:
    """Main correlation analysis orchestrator"""
    
    def __init__(self):
        self.sentiment_performance = SentimentPerformanceCorrelation()
        self.visualizer = CorrelationVisualizer()
    
    def comprehensive_correlation_analysis(self, master_df: pd.DataFrame) -> Dict[str, any]:
        """Perform comprehensive correlation analysis with error handling"""
        print("🎯 **COMPREHENSIVE SENTIMENT-PERFORMANCE CORRELATION ANALYSIS**")
        print("=" * 80)
        
        # Validate input data
        if master_df.empty:
            return {"error": "Empty dataset provided"}
        
        print(f"📊 Dataset overview: {master_df.shape[0]:,} records, {master_df.shape[1]} features")
        
        analysis_results = {}
        
        # 1. Basic correlation analysis
        try:
            correlation_analysis = self.sentiment_performance.safe_correlation_analysis(master_df)
            analysis_results['correlation_analysis'] = correlation_analysis
        except Exception as e:
            print(f"❌ Correlation analysis failed: {e}")
            analysis_results['correlation_error'] = str(e)
        
        # 2. Sentiment regime performance analysis
        try:
            regime_analysis = self.sentiment_performance.analyze_sentiment_regime_performance(master_df)
            analysis_results['regime_analysis'] = regime_analysis
        except Exception as e:
            print(f"❌ Regime analysis failed: {e}")
            analysis_results['regime_error'] = str(e)
        
        # 3. Trader behavior analysis
        try:
            behavior_analysis = self.sentiment_performance.trader_behavior_by_sentiment(master_df)
            analysis_results['behavior_analysis'] = behavior_analysis
        except Exception as e:
            print(f"❌ Behavior analysis failed: {e}")
            analysis_results['behavior_error'] = str(e)
        
        return analysis_results

if __name__ == "__main__":
    print("🛠️ **Web3 Trading Analysis - Correlation Analysis Module Loaded**")
    print("Available classes: SentimentPerformanceCorrelation, CorrelationVisualizer, CorrelationAnalyzer")
