"""
Feature Engineering Module for Web3 Trading Analysis
Creates performance metrics, sentiment features, and behavioral indicators
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class TradingPerformanceFeatures:
    """Generate trading performance metrics and features"""
    
    @staticmethod
    def calculate_daily_trader_performance(trading_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate daily performance metrics for each trader"""
        print("📈 **Calculating Daily Trader Performance Metrics**")
        
        # Group by trader and trading date
        daily_performance = trading_df.groupby(['Account', 'trading_date']).agg({
            'Closed PnL': ['sum', 'mean', 'count'],
            'Size USD': ['sum', 'mean', 'max'],
            'Side': lambda x: (x == 'BUY').sum(),  # Count of buy trades
            'Coin': 'nunique',  # Number of unique coins traded
            'Fee': 'sum'
        }).round(4)
        
        # Flatten column names
        daily_performance.columns = [
            'total_pnl', 'avg_pnl_per_trade', 'total_trades',
            'total_volume_usd', 'avg_trade_size_usd', 'max_trade_size_usd',
            'buy_trades_count', 'unique_coins_traded', 'total_fees'
        ]
        
        # Calculate additional metrics
        daily_performance['sell_trades_count'] = daily_performance['total_trades'] - daily_performance['buy_trades_count']
        daily_performance['buy_sell_ratio'] = (daily_performance['buy_trades_count'] / 
                                             (daily_performance['sell_trades_count'] + 0.001))  # Avoid division by zero
        
        # Performance indicators
        daily_performance['profitable_day'] = (daily_performance['total_pnl'] > 0).astype(int)
        daily_performance['net_profit_after_fees'] = daily_performance['total_pnl'] - daily_performance['total_fees']
        daily_performance['roi_percentage'] = ((daily_performance['total_pnl'] / 
                                              daily_performance['total_volume_usd']) * 100).round(4)
        
        # Risk metrics
        daily_performance['pnl_volatility'] = trading_df.groupby(['Account', 'trading_date'])['Closed PnL'].std().fillna(0)
        daily_performance['sharpe_ratio_daily'] = (daily_performance['total_pnl'] / 
                                                  (daily_performance['pnl_volatility'] + 0.001)).round(4)
        
        # Trading activity metrics
        daily_performance['trades_per_coin'] = (daily_performance['total_trades'] / 
                                              daily_performance['unique_coins_traded']).round(2)
        
        print(f"✅ Generated {len(daily_performance)} trader-day performance records")
        return daily_performance.reset_index()
    
    @staticmethod
def calculate_trader_behavior_features(trading_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate trader behavioral characteristics"""
    print("👤 **Analyzing Trader Behavioral Features**")
    
    trader_behavior = trading_df.groupby('Account').agg({
        'Closed PnL': ['sum', 'mean', 'std', 'count'],
        'Size USD': ['sum', 'mean', 'std', 'max'],
        'trading_date': ['nunique', 'min', 'max'],
        'Coin': 'nunique',
        'Side': lambda x: (x == 'BUY').mean(),  # Buy ratio
        'Fee': 'sum'
    })
    
    # Flatten columns
    trader_behavior.columns = [
        'total_career_pnl', 'avg_pnl_per_trade', 'pnl_volatility', 'total_trades',
        'total_career_volume', 'avg_trade_size', 'trade_size_volatility', 'max_trade_size',
        'active_days', 'first_trade_date', 'last_trade_date',
        'unique_coins_traded', 'buy_preference_ratio', 'total_career_fees'
    ]
    
    # Calculate additional behavioral metrics
    trader_behavior['trading_frequency'] = (trader_behavior['total_trades'] / 
                                          trader_behavior['active_days']).round(2)
    
    # FIX: Correct win rate calculation
    win_rates = []
    for account in trader_behavior.index:
        account_trades = trading_df[trading_df['Account'] == account]
        profitable_trades = (account_trades['Closed PnL'] > 0).sum()
        total_trades = len(account_trades)
        win_rate = profitable_trades / total_trades if total_trades > 0 else 0
        win_rates.append(round(win_rate, 4))
    
    trader_behavior['win_rate'] = win_rates
    
    # FIX: Correct profit factor calculation
    profit_factors = []
    for account in trader_behavior.index:
        account_trades = trading_df[trading_df['Account'] == account]
        profits = account_trades[account_trades['Closed PnL'] > 0]['Closed PnL'].sum()
        losses = abs(account_trades[account_trades['Closed PnL'] < 0]['Closed PnL'].sum())
        profit_factor = profits / losses if losses > 0 else (profits if profits > 0 else 0)
        profit_factors.append(round(profit_factor, 4))
    
    trader_behavior['profit_factor'] = profit_factors
    
    trader_behavior['risk_adjusted_return'] = (trader_behavior['total_career_pnl'] / 
                                             trader_behavior['trade_size_volatility']).fillna(0).round(4)
    
    trader_behavior['diversification_score'] = (trader_behavior['unique_coins_traded'] / 
                                               trader_behavior['total_trades']).round(4)
    
    # Trading style classification
    trader_behavior['trader_type'] = 'Unknown'
    trader_behavior.loc[trader_behavior['trading_frequency'] > 10, 'trader_type'] = 'High_Frequency'
    trader_behavior.loc[(trader_behavior['trading_frequency'] <= 10) & 
                      (trader_behavior['trading_frequency'] > 2), 'trader_type'] = 'Active'
    trader_behavior.loc[trader_behavior['trading_frequency'] <= 2, 'trader_type'] = 'Casual'
    
    print(f"✅ Analyzed behavioral features for {len(trader_behavior)} unique traders")
    return trader_behavior.reset_index()

class SentimentFeatures:
    """Generate sentiment-based features"""
    
    @staticmethod
    def create_sentiment_indicators(sentiment_df: pd.DataFrame) -> pd.DataFrame:
        """Create enhanced sentiment features"""
        print("😰😤 **Creating Enhanced Sentiment Features**")
        
        sentiment_enhanced = sentiment_df.copy()
        
        # Numerical sentiment score
        sentiment_mapping = {
            'Extreme Fear': 1,
            'Fear': 2,
            'Neutral': 3,
            'Greed': 4,
            'Extreme Greed': 5
        }
        
        sentiment_enhanced['sentiment_score'] = sentiment_enhanced['classification'].map(sentiment_mapping)
        
        # Sentiment momentum (change from previous day)
        sentiment_enhanced = sentiment_enhanced.sort_values('date_standardized')
        sentiment_enhanced['sentiment_change'] = sentiment_enhanced['sentiment_score'].diff()
        sentiment_enhanced['sentiment_momentum'] = sentiment_enhanced['sentiment_change'].apply(
            lambda x: 'Increasing' if x > 0 else ('Decreasing' if x < 0 else 'Stable')
        )
        
        # Rolling sentiment features (7-day windows)
        sentiment_enhanced['sentiment_7d_avg'] = sentiment_enhanced['sentiment_score'].rolling(window=7, min_periods=1).mean()
        sentiment_enhanced['sentiment_7d_volatility'] = sentiment_enhanced['sentiment_score'].rolling(window=7, min_periods=1).std()
        sentiment_enhanced['sentiment_trend_7d'] = (sentiment_enhanced['sentiment_score'] - 
                                                   sentiment_enhanced['sentiment_7d_avg']).round(4)
        
        # Market regime classification
        sentiment_enhanced['market_regime'] = 'Neutral'
        sentiment_enhanced.loc[sentiment_enhanced['sentiment_score'] <= 2, 'market_regime'] = 'Fear_Dominated'
        sentiment_enhanced.loc[sentiment_enhanced['sentiment_score'] >= 4, 'market_regime'] = 'Greed_Dominated'
        sentiment_enhanced.loc[(sentiment_enhanced['sentiment_score'] > 2) & 
                             (sentiment_enhanced['sentiment_score'] < 4), 'market_regime'] = 'Balanced'
        
        # Extreme event indicators
        sentiment_enhanced['extreme_fear_event'] = (sentiment_enhanced['classification'] == 'Extreme Fear').astype(int)
        sentiment_enhanced['extreme_greed_event'] = (sentiment_enhanced['classification'] == 'Extreme Greed').astype(int)
        
        print(f"✅ Enhanced {len(sentiment_enhanced)} sentiment records with advanced features")
        return sentiment_enhanced
    
    @staticmethod
    def create_sentiment_cycles(sentiment_df: pd.DataFrame) -> pd.DataFrame:
        """Identify sentiment cycles and patterns"""
        print("🔄 **Identifying Sentiment Cycles**")
        
        sentiment_cycles = sentiment_df.copy()
        
        # Identify sentiment cycle phases
        sentiment_cycles['cycle_phase'] = 'Unknown'
        
        # Simple cycle detection based on momentum
        for i in range(1, len(sentiment_cycles)):
            current_score = sentiment_cycles.iloc[i]['sentiment_score']
            prev_score = sentiment_cycles.iloc[i-1]['sentiment_score']
            
            if current_score > prev_score and current_score >= 3:
                sentiment_cycles.iloc[i, sentiment_cycles.columns.get_loc('cycle_phase')] = 'Recovery'
            elif current_score > prev_score and current_score < 3:
                sentiment_cycles.iloc[i, sentiment_cycles.columns.get_loc('cycle_phase')] = 'Bottoming'
            elif current_score < prev_score and current_score <= 3:
                sentiment_cycles.iloc[i, sentiment_cycles.columns.get_loc('cycle_phase')] = 'Declining'
            elif current_score < prev_score and current_score > 3:
                sentiment_cycles.iloc[i, sentiment_cycles.columns.get_loc('cycle_phase')] = 'Topping'
            else:
                sentiment_cycles.iloc[i, sentiment_cycles.columns.get_loc('cycle_phase')] = 'Stable'
        
        # Days since extreme events
        extreme_fear_dates = sentiment_cycles[sentiment_cycles['extreme_fear_event'] == 1].index
        extreme_greed_dates = sentiment_cycles[sentiment_cycles['extreme_greed_event'] == 1].index
        
        sentiment_cycles['days_since_extreme_fear'] = 999
        sentiment_cycles['days_since_extreme_greed'] = 999
        
        for i, row in sentiment_cycles.iterrows():
            if len(extreme_fear_dates) > 0:
                fear_distances = [abs(i - date) for date in extreme_fear_dates if date <= i]
                if fear_distances:
                    sentiment_cycles.at[i, 'days_since_extreme_fear'] = min(fear_distances)
            
            if len(extreme_greed_dates) > 0:
                greed_distances = [abs(i - date) for date in extreme_greed_dates if date <= i]
                if greed_distances:
                    sentiment_cycles.at[i, 'days_since_extreme_greed'] = min(greed_distances)
        
        print(f"✅ Identified sentiment cycles and extreme event distances")
        return sentiment_cycles

class MarketTimingFeatures:
    """Generate market timing and entry/exit features"""
    
    @staticmethod
    def create_timing_features(daily_performance: pd.DataFrame, sentiment_df: pd.DataFrame) -> pd.DataFrame:
        """Create market timing features by merging performance with sentiment"""
        print("⏰ **Creating Market Timing Features**")
        
        # Merge daily performance with sentiment data
        timing_features = daily_performance.merge(
            sentiment_df[['date_standardized', 'sentiment_score', 'market_regime', 'cycle_phase']], 
            left_on='trading_date', 
            right_on='date_standardized', 
            how='left'
        )
        
        # Performance by market regime
        timing_features['performance_in_fear'] = np.where(
            timing_features['market_regime'] == 'Fear_Dominated', 
            timing_features['total_pnl'], 
            0
        )
        
        timing_features['performance_in_greed'] = np.where(
            timing_features['market_regime'] == 'Greed_Dominated', 
            timing_features['total_pnl'], 
            0
        )
        
        timing_features['performance_in_neutral'] = np.where(
            timing_features['market_regime'] == 'Balanced', 
            timing_features['total_pnl'], 
            0
        )
        
        # Timing skill indicators
        timing_features['contrarian_indicator'] = np.where(
            (timing_features['market_regime'] == 'Fear_Dominated') & (timing_features['total_pnl'] > 0) |
            (timing_features['market_regime'] == 'Greed_Dominated') & (timing_features['total_pnl'] < 0),
            1, 0
        )
        
        timing_features['momentum_indicator'] = np.where(
            (timing_features['market_regime'] == 'Greed_Dominated') & (timing_features['total_pnl'] > 0) |
            (timing_features['market_regime'] == 'Fear_Dominated') & (timing_features['total_pnl'] < 0),
            1, 0
        )
        
        print(f"✅ Created timing features for {len(timing_features)} trader-day records")
        return timing_features

class FeatureEngineering:
    """Main feature engineering orchestrator"""
    
    def __init__(self):
        self.trading_features = TradingPerformanceFeatures()
        self.sentiment_features = SentimentFeatures()
        self.timing_features = MarketTimingFeatures()
    
    def create_all_features(self, trading_df: pd.DataFrame, sentiment_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Create all feature sets"""
        print("🎯 **COMPREHENSIVE FEATURE ENGINEERING**")
        print("=" * 60)
        
        # 1. Trading performance features
        daily_performance = self.trading_features.calculate_daily_trader_performance(trading_df)
        trader_behavior = self.trading_features.calculate_trader_behavior_features(trading_df)
        
        # 2. Sentiment features
        sentiment_enhanced = self.sentiment_features.create_sentiment_indicators(sentiment_df)
        sentiment_cycles = self.sentiment_features.create_sentiment_cycles(sentiment_enhanced)
        
        # 3. Market timing features
        timing_features = self.timing_features.create_timing_features(daily_performance, sentiment_cycles)
        
        feature_sets = {
            'daily_performance': daily_performance,
            'trader_behavior': trader_behavior,
            'sentiment_enhanced': sentiment_enhanced,
            'sentiment_cycles': sentiment_cycles,
            'timing_features': timing_features
        }
        
        # Summary
        print(f"\n📊 **FEATURE ENGINEERING SUMMARY:**")
        for name, df in feature_sets.items():
            print(f"   • {name}: {len(df):,} records, {len(df.columns)} features")
        
        return feature_sets
    
    def create_master_dataset(self, feature_sets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Create master dataset with all features"""
        print("\n🔗 **Creating Master Analysis Dataset**")
        
        # Start with timing features (most comprehensive)
        master_df = feature_sets['timing_features'].copy()
        
        # Add trader behavior features
        master_df = master_df.merge(
            feature_sets['trader_behavior'][['Account', 'total_career_pnl', 'win_rate', 
                                           'profit_factor', 'trader_type', 'diversification_score']], 
            on='Account', 
            how='left'
        )
        
        print(f"✅ Master dataset created: {len(master_df):,} records, {len(master_df.columns)} features")
        return master_df

if __name__ == "__main__":
    print("🛠️ **Web3 Trading Analysis - Feature Engineering Module Loaded**")
    print("Available classes: TradingPerformanceFeatures, SentimentFeatures, MarketTimingFeatures, FeatureEngineering")
