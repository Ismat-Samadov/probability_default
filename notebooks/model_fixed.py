#!/usr/bin/env python3
"""
Fixed Advanced PD Model Training Script
======================================
Addresses data quality issues and infinite values
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime, timedelta
import joblib
import os
from pathlib import Path

# Machine Learning
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, FunctionTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix, 
    roc_curve, precision_recall_curve, brier_score_loss,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Statistical Tests
from scipy import stats
from scipy.stats import ks_2samp, chi2_contingency

# Configuration
warnings.filterwarnings('ignore')
try:
    plt.style.use('seaborn-v0_8')
except:
    try:
        plt.style.use('seaborn')
    except:
        plt.style.use('default')
sns.set_palette("husl")

# Set random seeds for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("🏦 Fixed Advanced PD Model Training Framework")
print("=" * 60)

class PDModelConfig:
    """Configuration class for PD model training"""
    
    # Data paths
    DATA_DIR = Path("data")
    RETAIL_PATH = DATA_DIR / "retail" / "retail_portfolio.csv"
    SME_PATH = DATA_DIR / "sme" / "sme_portfolio.csv"
    CORPORATE_PATH = DATA_DIR / "corporate" / "corporate_portfolio.csv"
    MACRO_PATH = DATA_DIR / "macroeconomic" / "macro_data.csv"
    
    # Model paths
    MODEL_DIR = Path("models")
    MODEL_DIR.mkdir(exist_ok=True)
    
    # Regulatory thresholds
    BASEL_MIN_PD = 0.0003
    IFRS9_STAGE1_THRESHOLD = 0.01
    IFRS9_STAGE2_THRESHOLD = 0.05
    
    # Model validation thresholds
    MIN_AUC = 0.7
    MAX_GINI_DECLINE = 0.05
    PSI_THRESHOLD = 0.1
    
    # Cross-validation settings
    CV_FOLDS = 5
    TEST_SIZE = 0.2
    VALIDATION_SIZE = 0.2

class ModelUtilities:
    """Utility functions for model training and validation"""
    
    @staticmethod
    def calculate_gini(y_true, y_prob):
        """Calculate Gini coefficient"""
        auc = roc_auc_score(y_true, y_prob)
        return 2 * auc - 1
    
    @staticmethod
    def calculate_ks_statistic(y_true, y_prob):
        """Calculate Kolmogorov-Smirnov statistic"""
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        return np.max(tpr - fpr)

def load_data():
    """Load and validate datasets"""
    print("📊 Loading datasets...")
    
    config = PDModelConfig()
    
    retail_data = pd.read_csv(config.RETAIL_PATH)
    sme_data = pd.read_csv(config.SME_PATH)
    corporate_data = pd.read_csv(config.CORPORATE_PATH)
    macro_data = pd.read_csv(config.MACRO_PATH)
    
    datasets = {
        'retail': retail_data,
        'sme': sme_data,
        'corporate': corporate_data,
        'macro': macro_data
    }
    
    print(f"✅ Data loaded successfully!")
    print(f"Retail: {len(retail_data):,} customers")
    print(f"SME: {len(sme_data):,} companies")
    print(f"Corporate: {len(corporate_data):,} companies")
    print(f"Macro: {len(macro_data)} time periods")
    
    # Basic statistics
    print("\n📈 Default Rates by Segment:")
    for name, data in datasets.items():
        if 'is_default' in data.columns:
            default_rate = data['is_default'].mean()
            print(f"{name.upper()}: {default_rate:.2%}")
    
    return datasets

def clean_infinite_values(df):
    """Clean infinite and extreme values from dataframe"""
    # Replace infinite values with NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Fill NaN values in numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        if df[col].isnull().any():
            # Use median for most columns, but 0 for ratio columns
            if any(keyword in col.lower() for keyword in ['ratio', 'rate', 'margin', 'coverage']):
                df[col] = df[col].fillna(0)
            else:
                df[col] = df[col].fillna(df[col].median())
    
    # Cap extreme values (beyond 99.9th percentile) for numerical columns
    for col in numerical_cols:
        if col not in ['customer_id', 'company_id', 'is_default']:
            if df[col].nunique() > 1:  # Only cap if there's variance
                q99_9 = df[col].quantile(0.999)
                q00_1 = df[col].quantile(0.001)
                df[col] = np.clip(df[col], q00_1, q99_9)
    
    return df

def engineer_features(datasets):
    """Engineer features for all segments"""
    print("🔧 Starting Feature Engineering...")
    
    # Retail features
    print("🔧 Engineering retail features...")
    retail_data = datasets['retail'].copy()
    
    # Safe financial ratios
    retail_data['debt_service_ratio'] = np.where(
        retail_data['income'] > 0, 
        retail_data['current_debt'] / np.maximum(retail_data['income'] / 12, 1), 
        0
    )
    retail_data['savings_to_income_ratio'] = np.where(
        retail_data['income'] > 0,
        retail_data['savings_balance'] / retail_data['income'],
        0
    )
    retail_data['available_credit'] = np.maximum(0, retail_data['credit_limit'] - retail_data['current_debt'])
    retail_data['available_credit_ratio'] = np.where(
        retail_data['credit_limit'] > 0,
        retail_data['available_credit'] / retail_data['credit_limit'],
        0
    )
    
    # Additional features
    retail_data['age_squared'] = retail_data['age'] ** 2
    retail_data['high_utilization'] = (retail_data['utilization_rate'] > 0.8).astype(int)
    retail_data['high_dti'] = (retail_data['debt_to_income'] > 0.4).astype(int)
    
    # Clean retail data
    retail_data = clean_infinite_values(retail_data)
    
    # SME features
    print("🔧 Engineering SME features...")
    sme_data = datasets['sme'].copy()
    
    # Safe financial indicators
    sme_data['revenue_per_employee'] = np.where(
        sme_data['num_employees'] > 0,
        sme_data['annual_revenue'] / sme_data['num_employees'],
        0
    )
    sme_data['cash_flow_margin'] = np.where(
        sme_data['annual_revenue'] > 0,
        sme_data['operating_cash_flow'] / sme_data['annual_revenue'],
        0
    )
    
    # Risk indicators
    sme_data['payment_risk_score'] = sme_data['payment_delays_12m'] * 10 + sme_data['days_past_due'] / 30
    
    # Clean SME data
    sme_data = clean_infinite_values(sme_data)
    
    # Corporate features
    print("🔧 Engineering corporate features...")
    corp_data = datasets['corporate'].copy()
    
    # Safe financial ratios
    corp_data['cash_generation_ability'] = np.where(
        corp_data['annual_revenue'] > 0,
        corp_data['free_cash_flow'] / corp_data['annual_revenue'],
        0
    )
    corp_data['market_cap_to_revenue'] = np.where(
        (corp_data['is_public'] == 1) & (corp_data['annual_revenue'] > 0),
        np.minimum(corp_data['market_cap'] / corp_data['annual_revenue'], 100),
        0
    )
    
    # Scale indicators
    corp_data['company_scale'] = (
        np.log1p(np.maximum(1, corp_data['annual_revenue'])) + 
        np.log1p(np.maximum(1, corp_data['num_employees']))
    )
    
    # Clean corporate data
    corp_data = clean_infinite_values(corp_data)
    
    print("✅ Feature engineering completed!")
    print(f"Retail features: {retail_data.shape[1]} columns")
    print(f"SME features: {sme_data.shape[1]} columns")
    print(f"Corporate features: {corp_data.shape[1]} columns")
    
    return {
        'retail': retail_data,
        'sme': sme_data,
        'corporate': corp_data
    }

def prepare_model_data(data, segment, target_col='is_default'):
    """Prepare data for modeling with robust preprocessing"""
    print(f"🎯 Preparing features for {segment} segment...")
    
    # Separate features and target
    y = data[target_col]
    
    # Select features (exclude ID, target, and dates)
    exclude_cols = [target_col, 'customer_id', 'company_id', 'observation_date', 'default_probability']
    feature_cols = [col for col in data.columns if col not in exclude_cols]
    X = data[feature_cols].copy()
    
    print(f"  🔍 Initial data shape: {X.shape}")
    
    # Handle missing values
    missing_cols = X.columns[X.isnull().any()].tolist()
    if missing_cols:
        print(f"  🔧 Filling missing values in {len(missing_cols)} columns...")
        for col in missing_cols:
            if X[col].dtype in ['object', 'category']:
                X[col] = X[col].fillna('Unknown')
            else:
                X[col] = X[col].fillna(X[col].median())
    
    # Handle categorical variables
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove constant features
    constant_features = []
    for col in numerical_features:
        if X[col].nunique() <= 1:
            constant_features.append(col)
    
    if constant_features:
        print(f"  🗑️  Removing {len(constant_features)} constant features...")
        X = X.drop(constant_features, axis=1)
        numerical_features = [col for col in numerical_features if col not in constant_features]
    
    print(f"📊 Final features summary:")
    print(f"  - Shape: {X.shape}")
    print(f"  - Numerical: {len(numerical_features)}")
    print(f"  - Categorical: {len(categorical_features)}")
    print(f"  - Target positive rate: {y.mean():.2%}")
    
    return X, y, numerical_features, categorical_features

def create_preprocessor(numerical_features, categorical_features):
    """Create preprocessing pipeline"""
    transformers = []
    
    if numerical_features:
        transformers.append(('num', StandardScaler(), numerical_features))
    
    if categorical_features:
        transformers.append(('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), 
                            categorical_features))
    
    if not transformers:
        transformers.append(('passthrough', FunctionTransformer(), []))
    
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='drop'
    )
    return preprocessor

def train_model_segment(X, y, segment):
    """Train model for a specific segment"""
    print(f"\n🤖 Training {segment.upper()} model...")
    
    # Split data
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=RANDOM_STATE
    )
    
    print(f"📊 Data split:")
    print(f"  Training: {len(X_train):,} samples ({y_train.mean():.2%} default)")
    print(f"  Validation: {len(X_val):,} samples ({y_val.mean():.2%} default)")
    print(f"  Test: {len(X_test):,} samples ({y_test.mean():.2%} default)")
    
    # Train models
    models = {}
    
    # Logistic Regression
    print("  Training Logistic Regression...")
    log_model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000, class_weight='balanced')
    log_model.fit(X_train, y_train)
    models['logistic'] = log_model
    
    # Random Forest
    print("  Training Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=100, max_depth=10, min_samples_split=50,
        random_state=RANDOM_STATE, class_weight='balanced', n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    models['random_forest'] = rf_model
    
    # Make predictions
    predictions = {}
    for name, model in models.items():
        pred_proba = model.predict_proba(X_val)[:, 1]
        predictions[name] = pred_proba
    
    # Ensemble prediction (simple average)
    ensemble_pred = np.mean(list(predictions.values()), axis=0)
    
    # Calculate metrics
    utils = ModelUtilities()
    ensemble_metrics = {
        'auc': roc_auc_score(y_val, ensemble_pred),
        'gini': utils.calculate_gini(y_val, ensemble_pred),
        'ks': utils.calculate_ks_statistic(y_val, ensemble_pred),
        'brier': brier_score_loss(y_val, ensemble_pred),
    }
    
    print(f"\n🎯 Validation Results:")
    print(f"  AUC: {ensemble_metrics['auc']:.4f}")
    print(f"  Gini: {ensemble_metrics['gini']:.4f}")
    print(f"  KS: {ensemble_metrics['ks']:.4f}")
    
    # Test set evaluation
    test_predictions = {}
    for name, model in models.items():
        test_pred = model.predict_proba(X_test)[:, 1]
        test_predictions[name] = test_pred
    
    ensemble_test_pred = np.mean(list(test_predictions.values()), axis=0)
    
    test_metrics = {
        'auc': roc_auc_score(y_test, ensemble_test_pred),
        'gini': utils.calculate_gini(y_test, ensemble_test_pred),
        'ks': utils.calculate_ks_statistic(y_test, ensemble_test_pred),
    }
    
    print(f"\n📊 Test Set Performance:")
    print(f"  AUC: {test_metrics['auc']:.4f}")
    print(f"  Gini: {test_metrics['gini']:.4f}")
    print(f"  KS: {test_metrics['ks']:.4f}")
    
    return {
        'models': models,
        'validation_metrics': ensemble_metrics,
        'test_metrics': test_metrics,
        'test_data': (X_test, y_test, ensemble_test_pred)
    }

def main():
    """Main training pipeline"""
    print("🚀 Starting PD Model Training Pipeline...")
    
    # Load data
    datasets = load_data()
    
    # Engineer features
    engineered_data = engineer_features(datasets)
    
    # Train models for each segment
    results = {}
    
    for segment_name, segment_data in engineered_data.items():
        try:
            # Prepare data
            X, y, num_features, cat_features = prepare_model_data(segment_data, segment_name)
            
            # Create preprocessor
            preprocessor = create_preprocessor(num_features, cat_features)
            
            # Fit and transform data
            X_processed = preprocessor.fit_transform(X)
            
            # Train model
            segment_results = train_model_segment(X_processed, y, segment_name)
            segment_results['preprocessor'] = preprocessor
            
            results[segment_name] = segment_results
            
            # Save model
            config = PDModelConfig()
            segment_dir = config.MODEL_DIR / segment_name
            segment_dir.mkdir(exist_ok=True)
            
            for model_name, model in segment_results['models'].items():
                model_path = segment_dir / f'{model_name}_model.joblib'
                joblib.dump(model, model_path)
            
            # Save preprocessor
            preprocessor_path = segment_dir / 'preprocessor.joblib'
            joblib.dump(preprocessor, preprocessor_path)
            
            print(f"✅ {segment_name.upper()} model saved successfully!")
            
        except Exception as e:
            print(f"❌ Error training {segment_name} model: {e}")
            continue
    
    # Create summary visualization
    print("\n📈 Creating Performance Summary...")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    segments = ['retail', 'sme', 'corporate']
    colors = ['skyblue', 'lightcoral', 'lightgreen']
    
    for i, segment in enumerate(segments):
        if segment in results:
            # ROC Curve
            X_test, y_test, y_pred = results[segment]['test_data']
            fpr, tpr, _ = roc_curve(y_test, y_pred)
            auc = results[segment]['test_metrics']['auc']
            
            axes[i].plot(fpr, tpr, color=colors[i], lw=2, label=f'AUC = {auc:.3f}')
            axes[i].plot([0, 1], [0, 1], 'k--', lw=1)
            axes[i].set_title(f'{segment.upper()} ROC Curve')
            axes[i].set_xlabel('False Positive Rate')
            axes[i].set_ylabel('True Positive Rate')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        else:
            axes[i].text(0.5, 0.5, 'Model\nNot Available', ha='center', va='center')
            axes[i].set_title(f'{segment.upper()} Model')
    
    plt.tight_layout()
    plt.savefig(config.MODEL_DIR / 'model_performance_summary.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print final summary
    print(f"\n🎉 TRAINING COMPLETED!")
    print("=" * 50)
    
    total_models = len(results)
    avg_auc = np.mean([results[s]['test_metrics']['auc'] for s in results.keys()])
    
    print(f"✅ Successfully trained {total_models}/3 models")
    print(f"📊 Average AUC: {avg_auc:.4f}")
    print(f"📁 Models saved to: {config.MODEL_DIR}")
    print(f"🚀 Ready for deployment!")

if __name__ == "__main__":
    main()