#!/usr/bin/env python3
"""
PD Model API Startup Script
===========================
This script helps you start the PD Model API with proper checks
"""

import subprocess
import sys
import time
import requests
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required. Current version:", sys.version)
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} is compatible")
    return True

def check_virtual_environment():
    """Check if virtual environment is active"""
    print("📦 Checking virtual environment...")
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Virtual environment is active")
        return True
    else:
        print("⚠️  Virtual environment not detected (recommended but not required)")
        return True

def check_requirements():
    """Check if required packages are installed"""
    print("📋 Checking required packages...")
    required_packages = [
        'fastapi', 'uvicorn', 'pandas', 'numpy', 'scikit-learn', 
        'joblib', 'jinja2', 'python-multipart'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n📥 Installing missing packages: {', '.join(missing_packages)}")
        try:
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install'
            ] + missing_packages)
            print("✅ All packages installed successfully")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to install packages. Please install manually:")
            print(f"pip install {' '.join(missing_packages)}")
            return False
    
    return True

def check_models():
    """Check if trained models exist"""
    print("🤖 Checking trained models...")
    
    model_dir = Path("models")
    if not model_dir.exists():
        print("❌ Models directory not found")
        print("   Please run: python train_models.py")
        return False
    
    segments = ['retail', 'sme', 'corporate']
    all_models_exist = True
    
    for segment in segments:
        segment_dir = model_dir / segment
        if not segment_dir.exists():
            print(f"❌ {segment} models directory missing")
            all_models_exist = False
            continue
            
        required_files = [
            'logistic_model.joblib',
            'random_forest_model.joblib',
            'preprocessor.joblib'
        ]
        
        missing_files = []
        for file in required_files:
            if not (segment_dir / file).exists():
                missing_files.append(file)
        
        if missing_files:
            print(f"❌ {segment}: missing {', '.join(missing_files)}")
            all_models_exist = False
        else:
            print(f"✅ {segment}: all model files present")
    
    if not all_models_exist:
        print("\n🏋️ To train models, run:")
        print("   python data/generator.py")
        print("   python train_models.py")
        return False
    
    return True

def check_data():
    """Check if training data exists"""
    print("📊 Checking training data...")
    
    data_dir = Path("data")
    if not data_dir.exists():
        print("❌ Data directory not found")
        print("   Please run: python data/generator.py")
        return False
    
    required_data_files = [
        'data/retail/retail_portfolio.csv',
        'data/sme/sme_portfolio.csv',
        'data/corporate/corporate_portfolio.csv',
        'data/macroeconomic/macro_data.csv'
    ]
    
    missing_files = []
    for file_path in required_data_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
        else:
            print(f"✅ {file_path}")
    
    if missing_files:
        print(f"❌ Missing data files: {', '.join(missing_files)}")
        print("\n📊 To generate data, run:")
        print("   python data/generator.py")
        return False
    
    return True

def check_templates_and_static():
    """Check if templates and static files exist"""
    print("🎨 Checking templates and static files...")
    
    required_dirs = ['templates', 'static']
    for dir_name in required_dirs:
        if not Path(dir_name).exists():
            print(f"❌ {dir_name} directory missing")
            return False
        print(f"✅ {dir_name} directory exists")
    
    return True

def start_api_server():
    """Start the API server"""
    print("🚀 Starting PD Model API server...")
    print("   URL: http://localhost:8000")
    print("   API Docs: http://localhost:8000/api/docs")
    print("   Press Ctrl+C to stop\n")
    
    try:
        # Start uvicorn server
        subprocess.run([
            sys.executable, '-m', 'uvicorn', 
            'app:app', 
            '--host', '0.0.0.0', 
            '--port', '8000',
            '--reload',
            '--log-level', 'info'
        ])
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        return False
    
    return True

def wait_for_api():
    """Wait for API to be ready"""
    print("⏳ Waiting for API to start...")
    for i in range(30):  # Wait up to 30 seconds
        try:
            response = requests.get("http://localhost:8000/api/health", timeout=2)
            if response.status_code == 200:
                print("✅ API is ready!")
                return True
        except:
            pass
        time.sleep(1)
        print(f"   Attempt {i+1}/30...")
    
    print("❌ API failed to start within 30 seconds")
    return False

def run_quick_test():
    """Run a quick test of the API"""
    print("🧪 Running quick API test...")
    
    test_data = {
        "age": 35,
        "income": 75000,
        "credit_score": 720,
        "debt_to_income": 0.30,
        "utilization_rate": 0.25,
        "employment_status": "Full-time"
    }
    
    try:
        response = requests.post(
            "http://localhost:8000/api/predict/retail",
            json=test_data,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Test successful! PD Score: {result['pd_score']:.4f}")
            return True
        else:
            print(f"❌ Test failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False

def main():
    """Main startup function"""
    print("🏦 PD Model API Startup Script")
    print("=" * 50)
    
    # Run all checks
    checks = [
        ("Python Version", check_python_version),
        ("Virtual Environment", check_virtual_environment),
        ("Required Packages", check_requirements),
        ("Training Data", check_data),
        ("Trained Models", check_models),
        ("Templates & Static", check_templates_and_static)
    ]
    
    print("🔍 Running startup checks...")
    all_checks_passed = True
    
    for check_name, check_func in checks:
        print(f"\n{'='*20}")
        try:
            if not check_func():
                all_checks_passed = False
        except Exception as e:
            print(f"❌ {check_name} check failed: {e}")
            all_checks_passed = False
    
    print(f"\n{'='*50}")
    if all_checks_passed:
        print("✅ All checks passed! Starting API server...")
        
        # Give user option to continue
        response = input("\nPress Enter to start the server (or Ctrl+C to cancel): ")
        
        # Start the server
        start_api_server()
        
    else:
        print("❌ Some checks failed. Please fix the issues above before starting the API.")
        print("\n🔧 Common fixes:")
        print("   1. Install packages: pip install -r requirements.txt")
        print("   2. Generate data: python data/generator.py")
        print("   3. Train models: python train_models.py")
        return False
    
    return True

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)