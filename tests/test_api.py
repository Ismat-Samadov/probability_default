#!/usr/bin/env python3
"""
Test Script to Verify PD Model API Fixes
========================================
Run this script to test that the fixes work correctly
"""

import requests
import json
import time
import pandas as pd
from pathlib import Path

def test_api_health():
    """Test API health endpoint"""
    print("🔍 Testing API Health...")
    try:
        response = requests.get("http://localhost:8000/api/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API Health: {data['status']}")
            for segment, info in data['models'].items():
                status = "✅" if info['loaded'] else "❌"
                print(f"  {status} {segment.upper()}: {info['model_count']} models")
            return True
        else:
            print(f"❌ API Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ API Health check error: {e}")
        return False

def test_retail_prediction():
    """Test retail customer prediction"""
    print("\n🏠 Testing Retail Prediction...")
    
    retail_data = {
        "age": 35,
        "income": 75000,
        "credit_score": 720,
        "debt_to_income": 0.30,
        "utilization_rate": 0.25,
        "employment_status": "Full-time",
        "employment_tenure": 3.0,
        "years_at_address": 2.0,
        "num_accounts": 5,
        "monthly_transactions": 45
    }
    
    try:
        response = requests.post(
            "http://localhost:8000/api/predict/retail",
            json=retail_data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Retail prediction successful!")
            print(f"  PD Score: {result['pd_score']:.4f}")
            print(f"  Risk Grade: {result['risk_grade']}")
            print(f"  IFRS 9 Stage: {result['ifrs9_stage']}")
            return True
        else:
            print(f"❌ Retail prediction failed: {response.status_code}")
            print(f"  Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Retail prediction error: {e}")
        return False

def test_sme_prediction():
    """Test SME company prediction"""
    print("\n🏢 Testing SME Prediction...")
    
    sme_data = {
        "industry": "Professional Services",
        "years_in_business": 8.0,
        "annual_revenue": 1500000,
        "num_employees": 20,
        "current_ratio": 1.6,
        "debt_to_equity": 1.2,
        "interest_coverage": 5.0,
        "profit_margin": 0.08,
        "operating_cash_flow": 150000,
        "working_capital": 200000,
        "credit_utilization": 0.35,
        "payment_delays_12m": 1,
        "geographic_risk": "Low",
        "market_competition": "Medium",
        "management_quality": 7.0,
        "days_past_due": 0
    }
    
    try:
        response = requests.post(
            "http://localhost:8000/api/predict/sme",
            json=sme_data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ SME prediction successful!")
            print(f"  PD Score: {result['pd_score']:.4f}")
            print(f"  Risk Grade: {result['risk_grade']}")
            print(f"  IFRS 9 Stage: {result['ifrs9_stage']}")
            return True
        else:
            print(f"❌ SME prediction failed: {response.status_code}")
            print(f"  Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ SME prediction error: {e}")
        return False

def test_corporate_prediction():
    """Test corporate entity prediction"""
    print("\n🏗️ Testing Corporate Prediction...")
    
    corporate_data = {
        "industry": "Technology",
        "annual_revenue": 1000000000,
        "num_employees": 5000,
        "current_ratio": 1.5,
        "debt_to_equity": 0.8,
        "times_interest_earned": 8.0,
        "roa": 0.08,
        "credit_rating": "A",
        "market_position": "Strong",
        "operating_cash_flow": 150000000,
        "free_cash_flow": 100000000
    }
    
    try:
        response = requests.post(
            "http://localhost:8000/api/predict/corporate",
            json=corporate_data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Corporate prediction successful!")
            print(f"  PD Score: {result['pd_score']:.4f}")
            print(f"  Risk Grade: {result['risk_grade']}")
            print(f"  IFRS 9 Stage: {result['ifrs9_stage']}")
            return True
        else:
            print(f"❌ Corporate prediction failed: {response.status_code}")
            print(f"  Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Corporate prediction error: {e}")
        return False

def test_web_interface():
    """Test web interface endpoints"""
    print("\n🌐 Testing Web Interface...")
    
    endpoints = [
        ("Home", "http://localhost:8000/"),
        ("Retail Form", "http://localhost:8000/retail"),
        ("SME Form", "http://localhost:8000/sme"),
        ("Corporate Form", "http://localhost:8000/corporate"),
        ("Batch Form", "http://localhost:8000/batch")
    ]
    
    success_count = 0
    for name, url in endpoints:
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                print(f"✅ {name}: OK")
                success_count += 1
            else:
                print(f"❌ {name}: {response.status_code}")
        except Exception as e:
            print(f"❌ {name}: {e}")
    
    return success_count == len(endpoints)

def create_test_csv():
    """Create a test CSV file for batch processing"""
    print("\n📄 Creating test CSV file...")
    
    # Create sample retail data
    test_data = []
    for i in range(5):
        test_data.append({
            "age": 30 + i * 5,
            "income": 50000 + i * 10000,
            "credit_score": 650 + i * 20,
            "debt_to_income": 0.2 + i * 0.1,
            "utilization_rate": 0.2 + i * 0.1,
            "employment_status": "Full-time",
            "employment_tenure": 2.0 + i,
            "years_at_address": 1.0 + i,
            "num_accounts": 3 + i,
            "monthly_transactions": 30 + i * 5
        })
    
    df = pd.DataFrame(test_data)
    csv_path = Path("test_batch_data.csv")
    df.to_csv(csv_path, index=False)
    print(f"✅ Created test CSV: {csv_path}")
    return csv_path

def test_batch_processing():
    """Test batch processing endpoint"""
    print("\n📊 Testing Batch Processing...")
    
    # Create test CSV
    csv_path = create_test_csv()
    
    try:
        with open(csv_path, 'rb') as f:
            files = {'file': ('test_data.csv', f, 'text/csv')}
            data = {'segment': 'retail'}
            
            response = requests.post(
                "http://localhost:8000/api/predict/batch",
                files=files,
                data=data,
                timeout=60
            )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Batch processing successful!")
            print(f"  Total predictions: {result['total_predictions']}")
            print(f"  Successful: {result['successful_predictions']}")
            print(f"  Failed: {result['failed_predictions']}")
            print(f"  Processing time: {result['processing_time_seconds']:.2f}s")
            
            # Clean up
            csv_path.unlink()
            return True
        else:
            print(f"❌ Batch processing failed: {response.status_code}")
            print(f"  Error: {response.text}")
            # Clean up
            csv_path.unlink()
            return False
            
    except Exception as e:
        print(f"❌ Batch processing error: {e}")
        # Clean up
        if csv_path.exists():
            csv_path.unlink()
        return False

def run_all_tests():
    """Run all tests"""
    print("🧪 Running PD Model API Tests")
    print("=" * 50)
    
    tests = [
        ("API Health", test_api_health),
        ("Retail Prediction", test_retail_prediction),
        ("SME Prediction", test_sme_prediction),
        ("Corporate Prediction", test_corporate_prediction),
        ("Web Interface", test_web_interface),
        ("Batch Processing", test_batch_processing)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20}")
        start_time = time.time()
        
        try:
            success = test_func()
            duration = time.time() - start_time
            results.append((test_name, success, duration))
        except Exception as e:
            duration = time.time() - start_time
            print(f"❌ {test_name} failed with error: {e}")
            results.append((test_name, False, duration))
    
    # Print summary
    print(f"\n{'='*50}")
    print("📋 TEST SUMMARY")
    print(f"{'='*50}")
    
    passed = 0
    total = len(results)
    
    for test_name, success, duration in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status:8} | {test_name:20} | {duration:.2f}s")
        if success:
            passed += 1
    
    print(f"{'='*50}")
    print(f"RESULT: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All tests passed! The fixes are working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")
    
    return passed == total

if __name__ == "__main__":
    print("🚀 Starting PD Model API Test Suite")
    print("Please make sure the API is running on http://localhost:8000")
    print("Run: python app.py")
    
    input("\nPress Enter to start tests...")
    
    success = run_all_tests()
    
    if success:
        exit(0)
    else:
        exit(1)