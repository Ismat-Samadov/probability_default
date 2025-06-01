#!/usr/bin/env python3
"""
Test File Upload for PD Model API
=================================
Creates test CSV files and tests upload functionality
"""

import pandas as pd
import requests
import json
from pathlib import Path
import time

def create_test_csv_files():
    """Create test CSV files for all segments"""
    print("🗂️ Creating test CSV files...")
    
    # Create retail test data
    retail_data = []
    for i in range(5):
        retail_data.append({
            'age': 30 + i * 5,
            'income': 50000 + i * 10000,
            'credit_score': 650 + i * 20,
            'debt_to_income': 0.2 + i * 0.1,
            'utilization_rate': 0.2 + i * 0.1,
            'employment_status': 'Full-time',
            'employment_tenure': 2.0 + i,
            'years_at_address': 1.0 + i,
            'num_accounts': 3 + i,
            'monthly_transactions': 30 + i * 5
        })
    
    retail_df = pd.DataFrame(retail_data)
    retail_df.to_csv('test_retail.csv', index=False)
    print(f"✅ Created test_retail.csv with {len(retail_df)} rows")
    
    # Create SME test data
    sme_data = []
    for i in range(5):
        sme_data.append({
            'industry': 'Technology',
            'years_in_business': 5.0 + i,
            'annual_revenue': 1000000 + i * 500000,
            'num_employees': 10 + i * 5,
            'current_ratio': 1.2 + i * 0.2,
            'debt_to_equity': 0.5 + i * 0.3,
            'interest_coverage': 3.0 + i * 2,
            'profit_margin': 0.05 + i * 0.02,
            'operating_cash_flow': 100000 + i * 50000,
            'working_capital': 200000 + i * 100000,
            'credit_utilization': 0.3 + i * 0.1,
            'payment_delays_12m': i,
            'geographic_risk': 'Low',
            'market_competition': 'Medium',
            'management_quality': 7.0 + i * 0.5,
            'days_past_due': 0
        })
    
    sme_df = pd.DataFrame(sme_data)
    sme_df.to_csv('test_sme.csv', index=False)
    print(f"✅ Created test_sme.csv with {len(sme_df)} rows")
    
    # Create corporate test data
    corporate_data = []
    for i in range(5):
        corporate_data.append({
            'industry': 'Technology',
            'annual_revenue': 100000000 + i * 50000000,
            'num_employees': 1000 + i * 500,
            'current_ratio': 1.3 + i * 0.2,
            'debt_to_equity': 0.4 + i * 0.2,
            'times_interest_earned': 5.0 + i * 2,
            'roa': 0.06 + i * 0.02,
            'credit_rating': ['A', 'BBB', 'A', 'AA', 'BBB'][i],
            'market_position': ['Strong', 'Average', 'Strong', 'Leader', 'Average'][i],
            'operating_cash_flow': 10000000 + i * 5000000,
            'free_cash_flow': 8000000 + i * 4000000
        })
    
    corporate_df = pd.DataFrame(corporate_data)
    corporate_df.to_csv('test_corporate.csv', index=False)
    print(f"✅ Created test_corporate.csv with {len(corporate_df)} rows")
    
    return ['test_retail.csv', 'test_sme.csv', 'test_corporate.csv']

def test_file_upload(file_path, segment, api_url="http://localhost:8000"):
    """Test file upload to the API"""
    print(f"\n📤 Testing upload of {file_path} for {segment} segment...")
    
    try:
        with open(file_path, 'rb') as f:
            files = {'file': (file_path, f, 'text/csv')}
            data = {'segment': segment}
            
            response = requests.post(
                f"{api_url}/api/predict/batch",
                files=files,
                data=data,
                timeout=60
            )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Upload successful!")
            print(f"   Total predictions: {result['total_predictions']}")
            print(f"   Successful: {result['successful_predictions']}")
            print(f"   Failed: {result['failed_predictions']}")
            print(f"   Processing time: {result['processing_time_seconds']:.2f}s")
            
            if result['summary_statistics']:
                stats = result['summary_statistics']
                print(f"   Average PD: {stats.get('avg_pd_score', 0):.4f}")
                print(f"   Risk grades: {stats.get('risk_grade_distribution', {})}")
            
            return True
        else:
            print(f"❌ Upload failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Upload error: {e}")
        return False

def test_api_health(api_url="http://localhost:8000"):
    """Test API health"""
    print(f"\n🔍 Testing API health at {api_url}...")
    
    try:
        response = requests.get(f"{api_url}/api/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API is {data['status']}")
            
            for segment, info in data['models'].items():
                status = "✅" if info['loaded'] else "❌"
                print(f"   {status} {segment.upper()}: {info['model_count']} models loaded")
            
            return True
        else:
            print(f"❌ API health check failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ API health check error: {e}")
        return False

def test_single_prediction(api_url="http://localhost:8000"):
    """Test single predictions"""
    print(f"\n🧪 Testing single predictions...")
    
    # Test retail prediction
    retail_data = {
        "age": 35,
        "income": 75000,
        "credit_score": 720,
        "debt_to_income": 0.30,
        "utilization_rate": 0.25,
        "employment_status": "Full-time"
    }
    
    try:
        response = requests.post(
            f"{api_url}/api/predict/retail",
            json=retail_data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Retail prediction successful!")
            print(f"   PD Score: {result['pd_score']:.4f}")
            print(f"   Risk Grade: {result['risk_grade']}")
            print(f"   IFRS 9 Stage: {result['ifrs9_stage']}")
            return True
        else:
            print(f"❌ Retail prediction failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Retail prediction error: {e}")
        return False

def cleanup_test_files(file_list):
    """Clean up test files"""
    print(f"\n🧹 Cleaning up test files...")
    
    for file_path in file_list:
        try:
            Path(file_path).unlink()
            print(f"✅ Deleted {file_path}")
        except Exception as e:
            print(f"❌ Could not delete {file_path}: {e}")

def main():
    """Main test function"""
    print("🧪 PD Model API File Upload Test")
    print("=" * 50)
    
    api_url = "http://localhost:8000"
    
    # Step 1: Check API health
    if not test_api_health(api_url):
        print("\n❌ API is not healthy. Please start the API first:")
        print("   python app.py")
        return
    
    # Step 2: Test single prediction
    if not test_single_prediction(api_url):
        print("\n❌ Single prediction test failed")
        return
    
    # Step 3: Create test files
    test_files = create_test_csv_files()
    
    # Step 4: Test file uploads
    segments = ['retail', 'sme', 'corporate']
    upload_results = []
    
    for file_path, segment in zip(test_files, segments):
        success = test_file_upload(file_path, segment, api_url)
        upload_results.append(success)
        time.sleep(1)  # Small delay between uploads
    
    # Step 5: Results summary
    print(f"\n{'='*50}")
    print("📋 TEST RESULTS SUMMARY")
    print(f"{'='*50}")
    
    successful_uploads = sum(upload_results)
    total_uploads = len(upload_results)
    
    print(f"✅ API Health: OK")
    print(f"✅ Single Prediction: OK")
    print(f"📤 File Uploads: {successful_uploads}/{total_uploads} successful")
    
    for i, (file_path, segment, success) in enumerate(zip(test_files, segments, upload_results)):
        status = "✅" if success else "❌"
        print(f"   {status} {segment.capitalize()}: {file_path}")
    
    if successful_uploads == total_uploads:
        print(f"\n🎉 All tests passed! File upload functionality is working correctly.")
    else:
        print(f"\n⚠️  Some tests failed. Check the error messages above.")
    
    # Step 6: Cleanup
    cleanup_test_files(test_files)

if __name__ == "__main__":
    main()