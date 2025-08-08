"""
Web3 Trading Analysis - Main Execution Script
Entry point for the entire project
"""

import sys
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from config.config import create_project_structure
from src.data_loader import quick_data_check, DataLoader

def main():
    """Main execution function"""
    print("🚀 **WEB3 TRADING ANALYSIS PROJECT**")
    print("=" * 50)
    
    # Step 1: Create project structure
    print("\n📁 **STEP 1: Setting up project structure**")
    create_project_structure()
    
    # Step 2: Check datasets
    print("\n📊 **STEP 2: Analyzing datasets**")
    loader = quick_data_check()
    
    # Step 3: Next steps guidance
    print("\n🎯 **STEP 3: Next Actions**")
    print("Choose your next step:")
    print("1. Download Fear & Greed data: loader.smart_download('fear_greed_index')")
    print("2. Download Trading data: loader.smart_download('historical_trader_data')")
    print("3. Create samples for local testing")
    print("4. Move to Colab for heavy processing")
    
    return loader

if __name__ == "__main__":
    loader = main()
    print("\n✅ **Phase 1 Setup Complete!**")
    print("Project ready for development!")
