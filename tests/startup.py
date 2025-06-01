#!/usr/bin/env python3
"""
Quick test to verify the PD Model API is working
"""

import requests
import json

# Test the API health
def test_health():
    try:
        response = requests.get("https://probability-default.onrender.com//api/health")
        print("Health Check:", response.json())
        return response.status_code == 200
    except:
        print("API not reachable")
        return False

# Test retail prediction
def test_retail():
    data = {
        "age": 35,
        "income": 75000,
        "credit_score": 720,
        "debt_to_income": 0.30,
        "utilization_rate": 0.25,
        "employment_status": "Full-time"
    }
    
    try:
        response = requests.post("https://probability-default.onrender.com//api/predict/retail", json=data)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Retail Test: PD={result['pd_score']:.4f}, Grade={result['risk_grade']}")
            return True
        else:
            print(f"❌ Retail Test Failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Retail Test Error: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing PD Model API")
    print("=" * 30)
    
    if test_health():
        print("✅ API Health OK")
        if test_retail():
            print("✅ All tests passed!")
        else:
            print("❌ Some tests failed")
    else:
        print("❌ API not healthy")