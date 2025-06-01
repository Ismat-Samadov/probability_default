#!/usr/bin/env python3
"""
Complete Fixed PD Model API with Exact Feature Engineering
=========================================================
FastAPI application with exact feature engineering matching trained models
"""

from fastapi import FastAPI, HTTPException, Request, Form, UploadFile, File
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Union
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime
import json
import io
from contextlib import asynccontextmanager
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDModelPredictor:
    """Production PD model predictor with EXACT feature engineering"""
    
    def __init__(self, model_dir="models"):
        self.model_dir = Path(model_dir)
        self.models = {}
        self.preprocessors = {}
        self.load_models()
        
        # Regulatory constants
        self.BASEL_MIN_PD = 0.0003  # 3 basis points
        self.IFRS9_STAGE1_THRESHOLD = 0.01
        self.IFRS9_STAGE2_THRESHOLD = 0.05
        
        # Default macroeconomic values
        self.DEFAULT_MACRO = {
            'gdp_growth': 0.025,
            'unemployment_rate': 0.055,
            'interest_rate': 0.035,
            'inflation_rate': 0.02,
            'credit_spread': 0.015,
            'vix': 20.0
        }
    
    def load_models(self):
        """Load all trained models"""
        segments = ['retail', 'sme', 'corporate']
        
        for segment in segments:
            segment_dir = self.model_dir / segment
            if segment_dir.exists():
                try:
                    # Load preprocessor
                    preprocessor_path = segment_dir / 'preprocessor.joblib'
                    if preprocessor_path.exists():
                        self.preprocessors[segment] = joblib.load(preprocessor_path)
                    
                    # Load models
                    self.models[segment] = {}
                    self.models[segment]['logistic'] = joblib.load(segment_dir / 'logistic_model.joblib')
                    self.models[segment]['random_forest'] = joblib.load(segment_dir / 'random_forest_model.joblib')
                    
                    logger.info(f"✅ Loaded {segment} models successfully")
                    
                except Exception as e:
                    logger.error(f"❌ Error loading {segment} models: {e}")
    
    def safe_division(self, numerator, denominator, default_value=0):
        """Safe division handling all edge cases"""
        if isinstance(numerator, (pd.Series, np.ndarray)) or isinstance(denominator, (pd.Series, np.ndarray)):
            denominator = np.where(np.abs(denominator) < 1e-10, 1e-10, denominator)
            result = numerator / denominator
            if isinstance(result, (pd.Series, np.ndarray)):
                return np.where(np.isfinite(result), result, default_value)
            return result if np.isfinite(result) else default_value
        else:
            if abs(denominator) < 1e-10:
                return default_value
            result = numerator / denominator
            return result if np.isfinite(result) else default_value
    
    def clean_infinite_values(self, df):
        """Clean infinite and extreme values"""
        df = df.copy()
        
        # Replace infinite values with NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN values
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df[col].isnull().any():
                if any(keyword in col.lower() for keyword in ['ratio', 'rate', 'margin', 'coverage']):
                    df[col] = df[col].fillna(0)
                else:
                    df[col] = df[col].fillna(df[col].median())
        
        # Cap extreme values
        for col in numerical_cols:
            if col not in ['customer_id', 'company_id', 'is_default']:
                if df[col].nunique() > 1:
                    q99_9 = df[col].quantile(0.999)
                    q00_1 = df[col].quantile(0.001)
                    df[col] = np.clip(df[col], q00_1, q99_9)
        
        return df
    
    def add_macroeconomic_features(self, df, segment):
        """Add macroeconomic features exactly as in training"""
        df = df.copy()
        
        # Add default macro features
        for feature, value in self.DEFAULT_MACRO.items():
            if feature not in df.columns:
                df[feature] = value
        
        # Calculate macro risk factor EXACTLY as in training
        if segment == 'retail':
            df['macro_risk_factor'] = (
                df['unemployment_rate'] * 2 + 
                df['vix'] / 50
            )
        elif segment == 'sme':
            df['macro_risk_factor'] = (
                -df['gdp_growth'] * 3 +
                df['credit_spread'] * 20 +
                df['vix'] / 40
            )
        elif segment == 'corporate':
            df['macro_risk_factor'] = (
                df['interest_rate'] * 5 +
                df['credit_spread'] * 15 +
                df['vix'] / 60
            )
        
        return df

    def engineer_retail_features(self, df):
        """Engineer EXACT retail features matching training"""
        df = df.copy()
        logger.info(f"Starting retail feature engineering with columns: {list(df.columns)}")
        
        # Ensure required base columns exist
        required_base = ['age', 'income', 'debt_to_income', 'utilization_rate']
        for col in required_base:
            if col not in df.columns:
                raise ValueError(f"Required column missing: {col}")
        
        # Add missing base columns with defaults if not present
        if 'savings_balance' not in df.columns:
            df['savings_balance'] = df['income'] * 0.3
        if 'credit_limit' not in df.columns:
            df['credit_limit'] = df['income'] * 0.8
        if 'current_debt' not in df.columns:
            df['current_debt'] = df['income'] * df['debt_to_income']
        if 'checking_balance' not in df.columns:
            df['checking_balance'] = df['income'] * 0.05
        if 'monthly_transactions' not in df.columns:
            df['monthly_transactions'] = 45
        if 'avg_transaction_amount' not in df.columns:
            df['avg_transaction_amount'] = df['income'] / np.maximum(df['monthly_transactions'], 1)
        
        # EXACT feature engineering as in training script
        # 1. debt_service_ratio
        df['debt_service_ratio'] = np.where(
            df['income'] > 0, 
            df['current_debt'] / np.maximum(df['income'] / 12, 1), 
            0
        )
        
        # 2. savings_to_income_ratio
        df['savings_to_income_ratio'] = np.where(
            df['income'] > 0,
            df['savings_balance'] / df['income'],
            0
        )
        
        # 3. available_credit
        df['available_credit'] = np.maximum(0, df['credit_limit'] - df['current_debt'])
        
        # 4. available_credit_ratio
        df['available_credit_ratio'] = np.where(
            df['credit_limit'] > 0,
            df['available_credit'] / df['credit_limit'],
            0
        )
        
        # 5. age_squared
        df['age_squared'] = df['age'] ** 2
        
        # 6. high_utilization
        df['high_utilization'] = (df['utilization_rate'] > 0.8).astype(int)
        
        # 7. high_dti
        df['high_dti'] = (df['debt_to_income'] > 0.4).astype(int)
        
        # Add credit score if missing (calculated later)
        if 'credit_score' not in df.columns:
            df['credit_score'] = 720  # Will be recalculated
        
        # Add other required fields with defaults
        defaults = {
            'gender': 'M',
            'education_level': 'Bachelor',
            'employment_status': 'Full-time',
            'employment_tenure': 3.0,
            'years_at_address': 2.0,
            'homeowner': 1,
            'credit_history_length': 84,
            'num_accounts': 5,
            'payment_history': 0.95,
            'recent_inquiries': 1,
            'months_since_last_delinquency': 999.0,
            'num_delinquent_accounts': 0,
            'num_bank_accounts': 2
        }
        
        for col, default_val in defaults.items():
            if col not in df.columns:
                df[col] = default_val
        
        logger.info(f"Retail features after engineering: {list(df.columns)}")
        return self.clean_infinite_values(df)
    
    def engineer_sme_features(self, df):
            """Engineer EXACT SME features matching training"""
            df = df.copy()
            logger.info(f"Starting SME feature engineering with columns: {list(df.columns)}")
            
            # Required base columns
            required_base = ['annual_revenue', 'num_employees']
            for col in required_base:
                if col not in df.columns:
                    raise ValueError(f"Required column missing: {col}")
            
            # Add missing columns with defaults
            defaults = {
                'operating_cash_flow': lambda x: x['annual_revenue'] * 0.1,
                'asset_turnover': 1.5,
                'primary_bank_relationship_years': 5.0,
                'num_banking_products': 3,
                'credit_line_amount': 500000,
                'outstanding_loans': 200000,
                'payment_delays_12m': 0,  # Default: no payment delays
                'days_past_due': 0,       # Default: not past due
                'years_in_business': 5.0,
                'interest_coverage': 5.0,
                'credit_utilization': 0.3,
                'geographic_risk': 'Medium',
                'market_competition': 'Medium',
                'management_quality': 7.0,
                'working_capital': lambda x: x['annual_revenue'] * 0.15
            }
            
            for col, default_val in defaults.items():
                if col not in df.columns:
                    if callable(default_val):
                        df[col] = default_val(df)
                    else:
                        df[col] = default_val
            
            # EXACT feature engineering as in training
            # 1. revenue_per_employee
            df['revenue_per_employee'] = np.where(
                df['num_employees'] > 0,
                df['annual_revenue'] / df['num_employees'],
                0
            )
            
            # 2. cash_flow_margin
            df['cash_flow_margin'] = np.where(
                df['annual_revenue'] > 0,
                df['operating_cash_flow'] / df['annual_revenue'],
                0
            )
            
            # 3. payment_risk_score
            df['payment_risk_score'] = df['payment_delays_12m'] * 10 + df['days_past_due'] / 30
            
            logger.info(f"SME features after engineering: {list(df.columns)}")
            return self.clean_infinite_values(df)
    
    def engineer_corporate_features(self, df):
        """Engineer EXACT corporate features matching training"""
        df = df.copy()
        logger.info(f"Starting corporate feature engineering with columns: {list(df.columns)}")
        
        # Ensure required columns
        required_base = ['annual_revenue', 'num_employees', 'free_cash_flow']
        for col in required_base:
            if col not in df.columns:
                raise ValueError(f"Required column missing: {col}")
        
        # Add missing columns with defaults
        if 'market_cap' not in df.columns:
            df['market_cap'] = df['annual_revenue'] * 2
        if 'is_public' not in df.columns:
            df['is_public'] = 1
        if 'years_established' not in df.columns:
            df['years_established'] = 25
        
        # Add other defaults
        defaults = {
            'quick_ratio': 1.2,
            'cash_ratio': 0.8,
            'debt_to_assets': 0.4,
            'net_profit_margin': 0.1,
            'roe': 0.15,
            'asset_turnover': 1.5,
            'inventory_turnover': 8,
            'num_banking_relationships': 5,
            'primary_bank_relationship_years': 10.0,
            'total_credit_facilities': 100000000,
            'committed_facilities': 80000000,
            'utilization_rate': 0.3,
            'outstanding_debt': 50000000,
            'geographic_diversification': 'Regional',
            'regulatory_environment': 'Medium',
            'esg_score': 65
        }
        
        for col, default_val in defaults.items():
            if col not in df.columns:
                df[col] = default_val
        
        # EXACT feature engineering as in training
        # 1. cash_generation_ability
        df['cash_generation_ability'] = np.where(
            df['annual_revenue'] > 0,
            df['free_cash_flow'] / df['annual_revenue'],
            0
        )
        
        # 2. market_cap_to_revenue
        df['market_cap_to_revenue'] = np.where(
            (df['is_public'] == 1) & (df['annual_revenue'] > 0),
            np.minimum(df['market_cap'] / df['annual_revenue'], 100),
            0
        )
        
        # 3. company_scale
        df['company_scale'] = (
            np.log1p(np.maximum(1, df['annual_revenue'])) + 
            np.log1p(np.maximum(1, df['num_employees']))
        )
        
        logger.info(f"Corporate features after engineering: {list(df.columns)}")
        return self.clean_infinite_values(df)

    def calculate_credit_scores(self, df, segment):
        """Calculate credit scores exactly as in training"""
        df = df.copy()
        
        if segment == 'retail':
            # FICO-like calculation
            score = pd.Series([300.0] * len(df), index=df.index)
            
            # Payment history (35%)
            payment_history = df.get('payment_history', 0.95)
            if isinstance(payment_history, (int, float)):
                payment_history = pd.Series([payment_history] * len(df))
            score += payment_history * 315
            
            # Credit utilization (30%)
            utilization_penalty = np.where(df['utilization_rate'] < 0.3, 0, (df['utilization_rate'] - 0.3) * 200)
            score += (1 - df['utilization_rate']) * 270 - utilization_penalty
            
            # Credit history length (15%)
            credit_history = df.get('credit_history_length', 84)
            if isinstance(credit_history, (int, float)):
                credit_history = pd.Series([credit_history] * len(df))
            score += np.minimum(credit_history / 240, 1) * 135
            
            # Credit mix (10%)
            num_accounts = df.get('num_accounts', 5)
            if isinstance(num_accounts, (int, float)):
                num_accounts = pd.Series([num_accounts] * len(df))
            score += np.minimum(num_accounts / 10, 1) * 90
            
            # Recent inquiries (10%)
            recent_inquiries = df.get('recent_inquiries', 1)
            if isinstance(recent_inquiries, (int, float)):
                recent_inquiries = pd.Series([recent_inquiries] * len(df))
            inquiry_penalty = np.minimum(recent_inquiries * 15, 90)
            score += 90 - inquiry_penalty
            
            # DTI adjustment
            dti_penalty = np.where(df['debt_to_income'] > 0.4, (df['debt_to_income'] - 0.4) * 100, 0)
            score -= dti_penalty
            
            df['credit_score'] = np.clip(score, 300, 850).astype(int)
            
        elif segment == 'sme':
            # SME credit score
            score = pd.Series([300.0] * len(df), index=df.index)
            
            # Profitability (25%)
            score += np.clip(df['profit_margin'] * 1000 + 50, 0, 125)
            
            # Liquidity (20%)
            score += np.clip((df['current_ratio'] - 1) * 40 + 60, 20, 100)
            
            # Leverage (20%)
            score += np.clip(100 - df['debt_to_equity'] * 30, 20, 100)
            
            # Interest coverage (15%)
            score += np.clip(df['interest_coverage'] * 8, 0, 75)
            
            # Business tenure (10%)
            score += np.minimum(df['years_in_business'] * 2.5, 50)
            
            # Payment behavior (10%)
            score += np.clip(50 - df['payment_delays_12m'] * 10, 0, 50)
            
            # Credit utilization penalty
            util_penalty = np.where(df['credit_utilization'] > 0.5, (df['credit_utilization'] - 0.5) * 100, 0)
            score -= util_penalty
            
            df['sme_credit_score'] = np.clip(score, 300, 850).astype(int)
            
        elif segment == 'corporate':
            # Corporate credit score from rating
            rating_scores = {
                'AAA': 850, 'AA+': 820, 'AA': 800, 'AA-': 780,
                'A+': 760, 'A': 740, 'A-': 720, 'BBB+': 700,
                'BBB': 680, 'BBB-': 660, 'BB+': 640, 'BB': 620,
                'BB-': 600, 'B+': 580, 'B': 560, 'B-': 540,
                'CCC+': 520, 'CCC': 500, 'CCC-': 480
            }
            
            credit_rating = df.get('credit_rating', 'A')
            if isinstance(credit_rating, str):
                credit_rating = pd.Series([credit_rating] * len(df))
            
            base_scores = credit_rating.map(rating_scores).fillna(500)
            
            # Adjustments
            coverage_adj = np.clip((df['times_interest_earned'] - 5) * 5, -50, 50)
            leverage_adj = np.clip((1 - df['debt_to_equity']) * 20, -40, 40)
            profitability_adj = np.clip(df['roa'] * 200, -30, 30)
            liquidity_adj = np.clip((df['current_ratio'] - 1.5) * 10, -20, 20)
            
            # Stability adjustment
            stability_adj = np.clip(df['years_established'] / 5, 0, 20)
            
            # Market position
            position_map = {'Leader': 15, 'Strong': 5, 'Average': 0, 'Weak': -15}
            market_position = df.get('market_position', 'Strong')
            if isinstance(market_position, str):
                market_position = pd.Series([market_position] * len(df))
            position_adj = market_position.map(position_map).fillna(0)
            
            final_scores = (base_scores + coverage_adj + leverage_adj + 
                          profitability_adj + liquidity_adj + stability_adj + position_adj)
            
            df['corporate_credit_score'] = np.clip(final_scores, 300, 850).astype(int)
        
        return df
    
    def prepare_model_data(self, data, segment):
        """Prepare data exactly matching training expectations"""
        exclude_cols = ['is_default', 'customer_id', 'company_id', 'observation_date', 'default_probability']
        feature_cols = [col for col in data.columns if col not in exclude_cols]
        X = data[feature_cols].copy()
        
        logger.info(f"Prepared {segment} data with {len(feature_cols)} features: {feature_cols}")
        return X
    
    def predict_pd(self, data: pd.DataFrame, segment: str) -> Dict:
        """Predict PD with exact feature engineering"""
        if segment not in self.models or not self.models[segment]:
            raise ValueError(f"No models available for segment: {segment}")
        
        try:
            logger.info(f"Starting prediction for {segment} with {len(data)} records")
            original_columns = list(data.columns)
            logger.info(f"Input columns: {original_columns}")
            
            # 1. Add macroeconomic features
            data_with_macro = self.add_macroeconomic_features(data.copy(), segment)
            logger.info(f"After macro features: {data_with_macro.shape}")
            
            # 2. Apply feature engineering
            if segment == 'retail':
                data_engineered = self.engineer_retail_features(data_with_macro)
            elif segment == 'sme':
                data_engineered = self.engineer_sme_features(data_with_macro)
            elif segment == 'corporate':
                data_engineered = self.engineer_corporate_features(data_with_macro)
            else:
                raise ValueError(f"Unknown segment: {segment}")
            
            logger.info(f"After feature engineering: {data_engineered.shape}")
            
            # 3. Calculate credit scores
            data_with_scores = self.calculate_credit_scores(data_engineered, segment)
            logger.info(f"After credit score calculation: {data_with_scores.shape}")
            
            # 4. Prepare model data
            X = self.prepare_model_data(data_with_scores, segment)
            logger.info(f"Final feature matrix: {X.shape}")
            logger.info(f"Final features: {list(X.columns)}")
            
            # 5. Get expected features from preprocessor
            if segment in self.preprocessors:
                expected_features = []
                for name, transformer, features in self.preprocessors[segment].transformers_:
                    if name != 'remainder':
                        expected_features.extend(features)
                
                logger.info(f"Expected features ({len(expected_features)}): {expected_features}")
                
                # Check for missing features
                missing_features = set(expected_features) - set(X.columns)
                if missing_features:
                    logger.warning(f"Missing features: {missing_features}")
                    # Add missing features with defaults
                    for feature in missing_features:
                        if feature in ['industry', 'education_level', 'employment_status', 'gender']:
                            X[feature] = 'Unknown'
                        elif 'risk' in feature.lower():
                            X[feature] = 'Medium'
                        else:
                            X[feature] = 0
                    logger.info(f"Added missing features with defaults")
                
                # Reorder columns to match expected order
                X = X.reindex(columns=expected_features, fill_value=0)
                logger.info(f"Reordered features to match training: {X.shape}")
            
            # 6. Preprocess
            if segment in self.preprocessors:
                X_processed = self.preprocessors[segment].transform(X)
                logger.info(f"Preprocessed data: {X_processed.shape}")
            else:
                X_processed = X.values
                logger.info("No preprocessor available, using raw values")
            
            # 7. Predict
            logistic_pred = self.models[segment]['logistic'].predict_proba(X_processed)[:, 1]
            rf_pred = self.models[segment]['random_forest'].predict_proba(X_processed)[:, 1]
            
            # 8. Ensemble
            ensemble_pred = (logistic_pred + rf_pred) / 2
            
            # 9. Apply Basel III minimum floor
            ensemble_pred = np.maximum(ensemble_pred, self.BASEL_MIN_PD)
            
            logger.info(f"Prediction completed successfully for {segment}")
            
            return {
                'pd_scores': ensemble_pred.tolist(),
                'logistic_predictions': logistic_pred.tolist(),
                'random_forest_predictions': rf_pred.tolist(),
                'model_details': {
                    'segment': segment,
                    'basel_floor_applied': True,
                    'ensemble_method': 'simple_average'
                }
            }
            
        except Exception as e:
            logger.error(f"Prediction error for {segment}: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    def predict_with_staging(self, data: pd.DataFrame, segment: str) -> Dict:
        """Predict PD with IFRS 9 staging and risk grades"""
        prediction_result = self.predict_pd(data, segment)
        pd_scores = np.array(prediction_result['pd_scores'])
        
        # IFRS 9 stages
        stages = np.where(pd_scores <= self.IFRS9_STAGE1_THRESHOLD, 1,
                         np.where(pd_scores <= self.IFRS9_STAGE2_THRESHOLD, 2, 3))
        
        # Risk grades
        risk_grades = self._assign_risk_grades(pd_scores)
        
        return {
            'pd_scores': pd_scores.tolist(),
            'ifrs9_stages': stages.tolist(),
            'risk_grades': risk_grades,
            'logistic_predictions': prediction_result['logistic_predictions'],
            'random_forest_predictions': prediction_result['random_forest_predictions'],
            'basel_compliant': True,
            'prediction_timestamp': datetime.now().isoformat(),
            'model_details': prediction_result['model_details']
        }
    
    def _assign_risk_grades(self, pd_scores: np.ndarray) -> List[str]:
        """Assign risk grades based on PD scores"""
        conditions = [
            pd_scores <= 0.0025,   # AAA
            pd_scores <= 0.005,    # AA
            pd_scores <= 0.01,     # A
            pd_scores <= 0.025,    # BBB
            pd_scores <= 0.05,     # BB
            pd_scores <= 0.1,      # B
            pd_scores <= 0.25,     # CCC
            pd_scores <= 0.5,      # CC
        ]
        
        grades = ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC', 'CC', 'C']
        return np.select(conditions, grades[:-1], default=grades[-1]).tolist()

# Initialize predictor
predictor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    try:
        predictor = PDModelPredictor()
        logger.info("🚀 PD Model API started successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize models: {e}")
        predictor = None
    
    yield
    logger.info("🛑 PD Model API shutting down")

# FastAPI app
app = FastAPI(
    title="Advanced PD Model API",
    description="Enterprise-grade Probability of Default scoring with web interface",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    lifespan=lifespan
)

# Templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Request/Response Models (keeping existing ones)
class RetailCustomer(BaseModel):
    age: int = Field(..., ge=18, le=100)
    income: float = Field(..., gt=0)
    credit_score: int = Field(..., ge=300, le=850)
    debt_to_income: float = Field(..., ge=0, le=5)
    utilization_rate: float = Field(..., ge=0, le=1)
    employment_status: str = Field(default="Full-time")
    employment_tenure: float = Field(default=3.0)
    years_at_address: float = Field(default=2.0)
    num_accounts: int = Field(default=5)
    monthly_transactions: int = Field(default=45)

class SMECompany(BaseModel):
    industry: str = Field(...)
    years_in_business: float = Field(..., ge=0.1, le=100)
    annual_revenue: float = Field(..., gt=0)
    num_employees: int = Field(..., ge=1, le=500)
    current_ratio: float = Field(..., ge=0, le=10)
    debt_to_equity: float = Field(..., ge=0, le=10)
    interest_coverage: float = Field(..., ge=0, le=100)
    profit_margin: float = Field(..., ge=-1, le=1)
    operating_cash_flow: float = Field(...)
    working_capital: float = Field(...)
    credit_utilization: float = Field(..., ge=0, le=1)
    payment_delays_12m: int = Field(..., ge=0, le=12)
    geographic_risk: str = Field(default="Medium")
    market_competition: str = Field(default="Medium")
    management_quality: float = Field(..., ge=1, le=10)
    days_past_due: int = Field(default=0)

class CorporateEntity(BaseModel):
    industry: str = Field(...)
    annual_revenue: float = Field(..., gt=0)
    num_employees: int = Field(..., ge=1)
    current_ratio: float = Field(..., ge=0, le=10)
    debt_to_equity: float = Field(..., ge=0, le=10)
    times_interest_earned: float = Field(..., ge=0, le=100)
    roa: float = Field(..., ge=-1, le=1)
    credit_rating: str = Field(...)
    market_position: str = Field(...)
    operating_cash_flow: float = Field(...)
    free_cash_flow: float = Field(...)

class PDResponse(BaseModel):
    customer_id: Optional[str] = None
    segment: str
    pd_score: float
    risk_grade: str
    ifrs9_stage: int
    basel_compliant: bool
    prediction_timestamp: str
    model_version: str = "1.0.0"
    model_details: Dict

class BatchResponse(BaseModel):
    total_predictions: int
    successful_predictions: int
    failed_predictions: int
    processing_time_seconds: float
    results: List[Dict]
    summary_statistics: Dict

# Web Interface Routes
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/retail", response_class=HTMLResponse)
async def retail_form(request: Request):
    return templates.TemplateResponse("retail.html", {"request": request})

@app.get("/sme", response_class=HTMLResponse)
async def sme_form(request: Request):
    return templates.TemplateResponse("sme.html", {"request": request})

@app.get("/corporate", response_class=HTMLResponse)
async def corporate_form(request: Request):
    return templates.TemplateResponse("corporate.html", {"request": request})

@app.get("/batch", response_class=HTMLResponse)
async def batch_form(request: Request):
    return templates.TemplateResponse("batch.html", {"request": request})

# API Health Check
@app.get("/api/health")
async def health_check():
    if predictor is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    model_status = {}
    for segment in ['retail', 'sme', 'corporate']:
        model_status[segment] = {
            'loaded': segment in predictor.models and len(predictor.models[segment]) > 0,
            'model_count': len(predictor.models.get(segment, {})),
            'preprocessor_loaded': segment in predictor.preprocessors
        }
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models": model_status,
        "regulatory_compliance": {
            "basel_iii": True,
            "ifrs_9": True,
            "minimum_pd_floor": predictor.BASEL_MIN_PD
        }
    }

# API Prediction Endpoints
@app.post("/api/predict/retail", response_model=PDResponse)
async def predict_retail_api(customer: RetailCustomer, customer_id: Optional[str] = None):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        data = pd.DataFrame([customer.dict()])
        result = predictor.predict_with_staging(data, 'retail')
        
        return PDResponse(
            customer_id=customer_id,
            segment='retail',
            pd_score=result['pd_scores'][0],
            risk_grade=result['risk_grades'][0],
            ifrs9_stage=result['ifrs9_stages'][0],
            basel_compliant=result['basel_compliant'],
            prediction_timestamp=result['prediction_timestamp'],
            model_details=result['model_details']
        )
        
    except Exception as e:
        logger.error(f"Retail prediction error: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/api/predict/sme", response_model=PDResponse)
async def predict_sme_api(company: SMECompany, company_id: Optional[str] = None):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        data = pd.DataFrame([company.dict()])
        result = predictor.predict_with_staging(data, 'sme')
        
        return PDResponse(
            customer_id=company_id,
            segment='sme',
            pd_score=result['pd_scores'][0],
            risk_grade=result['risk_grades'][0],
            ifrs9_stage=result['ifrs9_stages'][0],
            basel_compliant=result['basel_compliant'],
            prediction_timestamp=result['prediction_timestamp'],
            model_details=result['model_details']
        )
        
    except Exception as e:
        logger.error(f"SME prediction error: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/api/predict/corporate", response_model=PDResponse)
async def predict_corporate_api(entity: CorporateEntity, entity_id: Optional[str] = None):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        data = pd.DataFrame([entity.dict()])
        result = predictor.predict_with_staging(data, 'corporate')
        
        return PDResponse(
            customer_id=entity_id,
            segment='corporate',
            pd_score=result['pd_scores'][0],
            risk_grade=result['risk_grades'][0],
            ifrs9_stage=result['ifrs9_stages'][0],
            basel_compliant=result['basel_compliant'],
            prediction_timestamp=result['prediction_timestamp'],
            model_details=result['model_details']
        )
        
    except Exception as e:
        logger.error(f"Corporate prediction error: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/api/predict/batch", response_model=BatchResponse)
async def predict_batch(file: UploadFile = File(...), segment: str = Form(...)):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    start_time = datetime.now()
    
    try:
        # Enhanced file validation
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
            
        if not file.filename.lower().endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are supported")
        
        if segment not in ['retail', 'sme', 'corporate']:
            raise HTTPException(status_code=400, detail="Invalid segment")
        
        # Read file with better error handling
        try:
            contents = await file.read()
            if len(contents) == 0:
                raise HTTPException(status_code=400, detail="Empty file")
                
            # Try different encodings
            try:
                content_str = contents.decode('utf-8')
            except UnicodeDecodeError:
                try:
                    content_str = contents.decode('latin-1')
                except UnicodeDecodeError:
                    content_str = contents.decode('utf-8', errors='ignore')
            
            df = pd.read_csv(io.StringIO(content_str))
            
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to read CSV file: {str(e)}")
        
        if len(df) == 0:
            raise HTTPException(status_code=400, detail="CSV file contains no data")
        
        if len(df) > 10000:
            raise HTTPException(status_code=400, detail="File too large. Maximum 10,000 rows allowed")
        
        logger.info(f"Processing batch file: {len(df)} rows for {segment}")
        
        # Process predictions
        results = []
        successful_predictions = 0
        failed_predictions = 0
        
        for idx, row in df.iterrows():
            try:
                row_data = pd.DataFrame([row.to_dict()])
                prediction_result = predictor.predict_with_staging(row_data, segment)
                
                result = {
                    'row_index': idx,
                    'pd_score': prediction_result['pd_scores'][0],
                    'risk_grade': prediction_result['risk_grades'][0],
                    'ifrs9_stage': prediction_result['ifrs9_stages'][0],
                    'success': True
                }
                results.append(result)
                successful_predictions += 1
                
            except Exception as e:
                result = {
                    'row_index': idx,
                    'error': str(e),
                    'success': False
                }
                results.append(result)
                failed_predictions += 1
                logger.warning(f"Failed to process row {idx}: {e}")
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Calculate summary statistics
        successful_results = [r for r in results if r['success']]
        if successful_results:
            pd_scores = [r['pd_score'] for r in successful_results]
            risk_grades = [r['risk_grade'] for r in successful_results]
            stages = [r['ifrs9_stage'] for r in successful_results]
            
            summary_stats = {
                'avg_pd_score': float(np.mean(pd_scores)),
                'median_pd_score': float(np.median(pd_scores)),
                'min_pd_score': float(np.min(pd_scores)),
                'max_pd_score': float(np.max(pd_scores)),
                'risk_grade_distribution': pd.Series(risk_grades).value_counts().to_dict(),
                'ifrs9_stage_distribution': pd.Series(stages).value_counts().to_dict()
            }
        else:
            summary_stats = {}
        
        logger.info(f"Batch processing completed: {successful_predictions} successful, {failed_predictions} failed")
        
        return BatchResponse(
            total_predictions=len(df),
            successful_predictions=successful_predictions,
            failed_predictions=failed_predictions,
            processing_time_seconds=processing_time,
            results=results,
            summary_statistics=summary_stats
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")

# Form submission endpoints
@app.post("/predict/retail")
async def predict_retail_form(
    request: Request,
    age: int = Form(...),
    income: float = Form(...),
    credit_score: int = Form(...),
    debt_to_income: float = Form(...),
    utilization_rate: float = Form(...),
    employment_status: str = Form(...),
    employment_tenure: float = Form(default=3.0),
    years_at_address: float = Form(default=2.0),
    num_accounts: int = Form(default=5),
    monthly_transactions: int = Form(default=45)
):
    try:
        customer = RetailCustomer(
            age=age,
            income=income,
            credit_score=credit_score,
            debt_to_income=debt_to_income,
            utilization_rate=utilization_rate,
            employment_status=employment_status,
            employment_tenure=employment_tenure,
            years_at_address=years_at_address,
            num_accounts=num_accounts,
            monthly_transactions=monthly_transactions
        )
        
        result = await predict_retail_api(customer)
        
        return templates.TemplateResponse("results.html", {
            "request": request,
            "result": result,
            "segment": "Retail Customer"
        })
        
    except Exception as e:
        logger.error(f"Retail form error: {e}")
        return templates.TemplateResponse("error.html", {
            "request": request,
            "error": str(e)
        })

@app.post("/predict/sme")
async def predict_sme_form(
    request: Request,
    industry: str = Form(...),
    years_in_business: float = Form(...),
    annual_revenue: float = Form(...),
    num_employees: int = Form(...),
    current_ratio: float = Form(...),
    debt_to_equity: float = Form(...),
    interest_coverage: float = Form(...),
    profit_margin: float = Form(...),
    operating_cash_flow: float = Form(...),
    working_capital: float = Form(...),
    credit_utilization: float = Form(...),
    payment_delays_12m: int = Form(...),
    geographic_risk: str = Form(...),
    market_competition: str = Form(...),
    management_quality: float = Form(...),
    days_past_due: int = Form(...)
):
    try:
        company = SMECompany(
            industry=industry,
            years_in_business=years_in_business,
            annual_revenue=annual_revenue,
            num_employees=num_employees,
            current_ratio=current_ratio,
            debt_to_equity=debt_to_equity,
            interest_coverage=interest_coverage,
            profit_margin=profit_margin,
            operating_cash_flow=operating_cash_flow,
            working_capital=working_capital,
            credit_utilization=credit_utilization,
            payment_delays_12m=payment_delays_12m,
            geographic_risk=geographic_risk,
            market_competition=market_competition,
            management_quality=management_quality,
            days_past_due=days_past_due
        )
        
        result = await predict_sme_api(company)
        
        return templates.TemplateResponse("results.html", {
            "request": request,
            "result": result,
            "segment": "SME Company"
        })
        
    except Exception as e:
        logger.error(f"SME form error: {e}")
        return templates.TemplateResponse("error.html", {
            "request": request,
            "error": str(e)
        })

@app.post("/predict/corporate")
async def predict_corporate_form(
    request: Request,
    industry: str = Form(...),
    annual_revenue: float = Form(...),
    num_employees: int = Form(...),
    current_ratio: float = Form(...),
    debt_to_equity: float = Form(...),
    times_interest_earned: float = Form(...),
    roa: float = Form(...),
    credit_rating: str = Form(...),
    market_position: str = Form(...),
    operating_cash_flow: float = Form(...),
    free_cash_flow: float = Form(...)
):
    try:
        entity = CorporateEntity(
            industry=industry,
            annual_revenue=annual_revenue,
            num_employees=num_employees,
            current_ratio=current_ratio,
            debt_to_equity=debt_to_equity,
            times_interest_earned=times_interest_earned,
            roa=roa,
            credit_rating=credit_rating,
            market_position=market_position,
            operating_cash_flow=operating_cash_flow,
            free_cash_flow=free_cash_flow
        )
        
        result = await predict_corporate_api(entity)
        
        return templates.TemplateResponse("results.html", {
            "request": request,
            "result": result,
            "segment": "Corporate Entity"
        })
        
    except Exception as e:
        logger.error(f"Corporate form error: {e}")
        return templates.TemplateResponse("error.html", {
            "request": request,
            "error": str(e)
        })

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc: HTTPException):
    return templates.TemplateResponse("error.html", {
        "request": request,
        "error": "Page not found"
    }, status_code=404)

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc: Exception):
    logger.error(f"Internal server error: {exc}")
    return templates.TemplateResponse("error.html", {
        "request": request,
        "error": "Internal server error"
    }, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )