#!/usr/bin/env python3
"""
Feature Checker Script
======================
Check what features your trained models expect
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path

def check_model_features():
    """Check what features each model expects"""
    print("🔍 Checking Model Feature Requirements")
    print("=" * 50)
    
    model_dir = Path("models")
    segments = ['retail', 'sme', 'corporate']
    
    for segment in segments:
        print(f"\n📊 {segment.upper()} SEGMENT:")
        print("-" * 30)
        
        segment_dir = model_dir / segment
        if not segment_dir.exists():
            print(f"❌ {segment} directory not found")
            continue
        
        # Load preprocessor to see expected features
        preprocessor_path = segment_dir / 'preprocessor.joblib'
        if preprocessor_path.exists():
            try:
                preprocessor = joblib.load(preprocessor_path)
                
                # Get feature names from the preprocessor
                if hasattr(preprocessor, 'transformers_'):
                    print("✅ Preprocessor loaded successfully")
                    
                    all_features = []
                    for name, transformer, features in preprocessor.transformers_:
                        if name != 'remainder':
                            print(f"  {name}: {len(features)} features")
                            all_features.extend(features)
                    
                    print(f"\n📋 Total expected features: {len(all_features)}")
                    print("Features list:")
                    for i, feature in enumerate(sorted(all_features), 1):
                        print(f"  {i:2d}. {feature}")
                    
                else:
                    print("❌ Could not extract features from preprocessor")
                    
            except Exception as e:
                print(f"❌ Error loading preprocessor: {e}")
        else:
            print(f"❌ Preprocessor not found")

def load_sample_data():
    """Load sample from original training data to see features"""
    print("\n\n🔍 Checking Original Training Data Features")
    print("=" * 50)
    
    data_files = {
        'retail': 'data/retail/retail_portfolio.csv',
        'sme': 'data/sme/sme_portfolio.csv',
        'corporate': 'data/corporate/corporate_portfolio.csv'
    }
    
    for segment, file_path in data_files.items():
        print(f"\n📊 {segment.upper()} DATA:")
        print("-" * 30)
        
        if Path(file_path).exists():
            try:
                df = pd.read_csv(file_path)
                
                # Exclude target and ID columns
                exclude_cols = ['is_default', 'customer_id', 'company_id', 'observation_date', 'default_probability']
                feature_cols = [col for col in df.columns if col not in exclude_cols]
                
                print(f"✅ Original data shape: {df.shape}")
                print(f"📋 Available features: {len(feature_cols)}")
                print("Feature list:")
                for i, feature in enumerate(sorted(feature_cols), 1):
                    print(f"  {i:2d}. {feature}")
                    
            except Exception as e:
                print(f"❌ Error loading {file_path}: {e}")
        else:
            print(f"❌ File not found: {file_path}")

def create_simple_test_data():
    """Create simple test data that matches the original structure"""
    print("\n\n🛠️  Creating Simplified Test Data")
    print("=" * 50)
    
    # Load original retail data as template
    retail_path = 'data/retail/retail_portfolio.csv'
    if Path(retail_path).exists():
        df = pd.read_csv(retail_path)
        sample = df.iloc[0:1].copy()  # Take first row as template
        
        # Show what the sample looks like
        print("✅ Sample retail customer from original data:")
        exclude_cols = ['is_default', 'default_probability']
        for col in sample.columns:
            if col not in exclude_cols:
                print(f"  {col}: {sample[col].iloc[0]}")
        
        return sample
    else:
        print("❌ Could not load original retail data")
        return None

if __name__ == "__main__":
    check_model_features()
    load_sample_data()
    sample = create_simple_test_data()