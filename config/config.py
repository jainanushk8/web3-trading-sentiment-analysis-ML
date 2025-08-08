"""
Web3 Trading Analysis - Configuration Settings
Author: Your Name
Date: August 2025
"""

import os
from pathlib import Path

# Project Structure
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw" 
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Google Drive Dataset URLs
DATASETS = {
    'historical_trader_data': {
        'url': 'https://drive.google.com/file/d/1IAfLZwu6rJzyWKgBToqwSmmVYU6VbjVs/view?usp=sharing',
        'file_id': '1IAfLZwu6rJzyWKgBToqwSmmVYU6VbjVs',
        'filename': 'historical_trader_data.csv',
        'description': 'Hyperliquid trading data with account, symbol, price, size, etc.'
    },
    'fear_greed_index': {
        'url': 'https://drive.google.com/file/d/1PgQC0tO8XN-wqkNyghWc_-mnrYv_nhSf/view?usp=sharing',
        'file_id': '1PgQC0tO8XN-wqkNyghWc_-mnrYv_nhSf',
        'filename': 'fear_greed_index.csv',
        'description': 'Bitcoin market sentiment data with Date and Classification'
    }
}

# File Size Limits (MB)
MAX_LOCAL_FILE_SIZE = 100  # Files larger than this go to Colab only
SAMPLE_SIZE_ROWS = 1000    # Sample size for local testing

# Analysis Parameters
DATE_FORMAT = '%Y-%m-%d'
TIMEZONE = 'UTC'

# Display Settings
FIGURE_SIZE = (12, 8)
DPI = 100
STYLE = 'seaborn-v0_8'

# Create directories if they don't exist
def create_project_structure():
    """Create all necessary directories"""
    directories = [
        DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, FEATURES_DIR,
        MODELS_DIR, RESULTS_DIR, NOTEBOOKS_DIR,
        RESULTS_DIR / "figures", RESULTS_DIR / "reports", 
        RESULTS_DIR / "insights", MODELS_DIR / "saved_models"
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")

if __name__ == "__main__":
    create_project_structure()
    print("🎯 Project structure created successfully!")
