import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style for professional-looking charts
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# 1. Top 5 Traders Performance
def create_top_traders_chart():
    traders = ['...01b2afbd', '...f52a9012', '...8bce7713', '...249c4ff1', '...b4ffb9f4']
    profits = [373532.41, 341435.32, 255554.63, 165934.84, 146502.44]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(traders, profits, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
    plt.title('Top 5 Traders by Total Profit', fontsize=16, fontweight='bold')
    plt.xlabel('Trader Account', fontsize=12)
    plt.ylabel('Total Profit ($)', fontsize=12)
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, profits):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5000,
                f'${value:,.0f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('assets/images/top_traders_performance.png', dpi=300, bbox_inches='tight')
    plt.close()

# 2. Model Performance Comparison
def create_model_performance_chart():
    models = ['Random Forest\n(Base)', 'Enhanced\nEnsemble']
    accuracies = [1.0, 1.0]
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(models, accuracies, color=['#2ca02c', '#ff7f0e'])
    plt.title('Machine Learning Model Performance', fontsize=16, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=12)
    plt.ylim(0.9, 1.05)
    
    # Add percentage labels
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.1%}', ha='center', va='bottom', fontweight='bold', fontsize=14)
    
    plt.tight_layout()
    plt.savefig('assets/images/model_performance.png', dpi=300, bbox_inches='tight')
    plt.close()

# 3. Portfolio Strategy Comparison
def create_portfolio_comparison_chart():
    strategies = ['Sentiment\nAdjusted', 'Minimum\nVariance', 'Equal\nWeight']
    sharpe_ratios = [1.267, 1.131, 0.968]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(strategies, sharpe_ratios, 
                   color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    plt.title('Portfolio Strategy Performance (Sharpe Ratios)', fontsize=16, fontweight='bold')
    plt.ylabel('Sharpe Ratio', fontsize=12)
    plt.ylim(0, max(sharpe_ratios) + 0.2)
    
    # Add value labels
    for bar, ratio in zip(bars, sharpe_ratios):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{ratio:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('assets/images/portfolio_strategies.png', dpi=300, bbox_inches='tight')
    plt.close()

# 4. Feature Importance Chart
def create_feature_importance_chart():
    features = ['Profitable Day', 'ROI Percentage', 'Net Profit After Fees', 
                'Sharpe Ratio Daily', 'Avg PnL per Trade']
    importances = [0.2428, 0.2096, 0.1588, 0.1579, 0.1169]
    
    plt.figure(figsize=(10, 6))
    bars = plt.barh(features, importances, color='teal')
    plt.title('Top 5 Feature Importance (Random Forest Model)', fontsize=16, fontweight='bold')
    plt.xlabel('Importance Score', fontsize=12)
    
    # Add percentage labels
    for bar, imp in zip(bars, importances):
        plt.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                f'{imp:.1%}', ha='left', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('assets/images/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()

# Generate all charts
if __name__ == "__main__":
    create_top_traders_chart()
    create_model_performance_chart()
    create_portfolio_comparison_chart()
    create_feature_importance_chart()
    print("✅ All visualizations generated successfully!")
