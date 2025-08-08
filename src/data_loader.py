"""
Data Loading Utilities for Web3 Trading Analysis
Handles Google Drive downloads and file size management
"""

import pandas as pd
import requests
import gdown
import os
from pathlib import Path
from tqdm import tqdm
import sys
sys.path.append(str(Path(__file__).parent.parent))

from config.config import DATASETS, RAW_DATA_DIR, MAX_LOCAL_FILE_SIZE, SAMPLE_SIZE_ROWS

class DataLoader:
    """Smart data loading with size management"""
    
    def __init__(self):
        self.datasets_info = DATASETS
        
    def get_file_size_mb(self, file_id):
        """Get file size from Google Drive without downloading"""
        try:
            url = f"https://drive.google.com/file/d/{file_id}/view"
            response = requests.head(url)
            if 'content-length' in response.headers:
                size_bytes = int(response.headers['content-length'])
                size_mb = size_bytes / (1024 * 1024)
                return round(size_mb, 2)
            else:
                return "Size unknown"
        except Exception as e:
            print(f"⚠️ Could not determine file size: {e}")
            return "Size unknown"
    
    def download_from_gdrive(self, file_id, output_path, max_size_mb=None):
        """Download file from Google Drive with size check"""
        try:
            # Check file size first
            size_mb = self.get_file_size_mb(file_id)
            print(f"📊 File size: {size_mb} MB")
            
            if isinstance(size_mb, float) and max_size_mb and size_mb > max_size_mb:
                print(f"⚠️ File too large ({size_mb}MB > {max_size_mb}MB)")
                return False
            
            # Download file
            url = f"https://drive.google.com/uc?id={file_id}"
            print(f"📥 Downloading to: {output_path}")
            gdown.download(url, str(output_path), quiet=False)
            
            # Verify download
            if output_path.exists():
                actual_size = output_path.stat().st_size / (1024 * 1024)
                print(f"✅ Download successful! Size: {actual_size:.2f}MB")
                return True
            else:
                print("❌ Download failed!")
                return False
                
        except Exception as e:
            print(f"❌ Download error: {e}")
            return False
    
    def load_dataset_info(self):
        """Display information about all datasets"""
        print("📋 **DATASET INFORMATION**")
        print("=" * 50)
        
        for name, info in self.datasets_info.items():
            print(f"\n🗂️ **{name.replace('_', ' ').title()}**")
            print(f"   Description: {info['description']}")
            print(f"   File: {info['filename']}")
            
            # Check file size
            size = self.get_file_size_mb(info['file_id'])
            print(f"   Size: {size} MB")
            
            # Recommendation
            if isinstance(size, float):
                if size > MAX_LOCAL_FILE_SIZE:
                    print(f"   💡 Recommendation: Use Colab (>{MAX_LOCAL_FILE_SIZE}MB)")
                else:
                    print(f"   ✅ Recommendation: Safe for local development")
    
    def create_sample_dataset(self, dataset_name, sample_rows=None):
        """Create a small sample for local testing"""
        if sample_rows is None:
            sample_rows = SAMPLE_SIZE_ROWS
            
        info = self.datasets_info[dataset_name]
        file_path = RAW_DATA_DIR / info['filename']
        sample_path = RAW_DATA_DIR / f"sample_{info['filename']}"
        
        try:
            if file_path.exists():
                print(f"📊 Creating sample from {info['filename']}...")
                df = pd.read_csv(file_path)
                sample_df = df.head(sample_rows)
                sample_df.to_csv(sample_path, index=False)
                
                print(f"✅ Sample created: {len(sample_df)} rows")
                print(f"   Original: {len(df)} rows ({file_path.stat().st_size / (1024*1024):.2f}MB)")
                print(f"   Sample: {len(sample_df)} rows ({sample_path.stat().st_size / 1024:.2f}KB)")
                return sample_path
            else:
                print(f"❌ Original file not found: {file_path}")
                return None
                
        except Exception as e:
            print(f"❌ Sample creation error: {e}")
            return None
    
    def smart_download(self, dataset_name, force_download=False):
        """Smart download with size checking and sampling"""
        info = self.datasets_info[dataset_name]
        file_path = RAW_DATA_DIR / info['filename']
        
        # Check if file already exists
        if file_path.exists() and not force_download:
            print(f"✅ File already exists: {file_path}")
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"   Size: {size_mb:.2f}MB")
            return file_path
        
        # Check file size before download
        size_mb = self.get_file_size_mb(info['file_id'])
        print(f"\n📊 **Analyzing {dataset_name}**")
        print(f"   File size: {size_mb} MB")
        
        if isinstance(size_mb, float):
            if size_mb > MAX_LOCAL_FILE_SIZE:
                print(f"⚠️ **Large file detected!**")
                print(f"   Recommended: Use Google Colab for this dataset")
                print(f"   Local limit: {MAX_LOCAL_FILE_SIZE}MB")
                
                # Ask user preference
                choice = input("   Download anyway? (y/n): ").lower().strip()
                if choice != 'y':
                    print("   💡 Skipping download. Use Colab for processing.")
                    return None
        
        # Proceed with download
        success = self.download_from_gdrive(info['file_id'], file_path, MAX_LOCAL_FILE_SIZE)
        
        if success:
            # Create sample for quick local testing
            self.create_sample_dataset(dataset_name)
            return file_path
        else:
            return None

# Utility functions
def quick_data_check():
    """Quick check of data availability and sizes"""
    loader = DataLoader()
    
    print("🎯 **WEB3 TRADING ANALYSIS - DATA CHECK**")
    print("=" * 60)
    
    # Show dataset information
    loader.load_dataset_info()
    
    # Check if raw data directory exists
    if not RAW_DATA_DIR.exists():
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        print(f"\n📁 Created data directory: {RAW_DATA_DIR}")
    
    return loader

if __name__ == "__main__":
    # Run data check when executed directly
    loader = quick_data_check()
    print("\n🚀 **Ready for data download!**")
    print("Next steps:")
    print("1. Run: python -c \"from src.data_loader import DataLoader; loader = DataLoader(); loader.smart_download('fear_greed_index')\"")
    print("2. Run: python -c \"from src.data_loader import DataLoader; loader = DataLoader(); loader.smart_download('historical_trader_data')\"")
