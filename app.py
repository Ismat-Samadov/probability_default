#!/usr/bin/env python3
"""
Fixed PD Model API with Complete Feature Engineering
==================================================
FastAPI application with proper feature handling for all trained models
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

# Global variables for models
models = {}
preprocessors = {}

class PDModelPredictor:
    """Production PD model predictor with complete feature engineering"""
    
    def __init__(self, model_dir="models"):
        self.model_dir = Path(model_dir)
        self.models = {}
        self.preprocessors = {}
        self.load_models()
        
        # Regulatory constants
        self.BASEL_MIN_PD = 0.0003  # 3 basis points
        self.IFRS9_STAGE1_THRESHOLD = 0.01
        self.IFRS9_STAGE2_THRESHOLD = 0.05
        
        # Default macroeconomic values (based on typical conditions)
        self.DEFAULT_MACRO = {
            'gdp_growth': 0.025,       # 2.5% GDP growth
            'unemployment_rate': 0.055, # 5.5% unemployment
            'interest_rate': 0.035,     # 3.5% interest rate
            'inflation_rate': 0.02,     # 2% inflation
            'credit_spread': 0.015,     # 1.5% credit spread
            'vix': 20.0                 # VIX volatility index
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
    
    def clean_infinite_values(self, df):
        """Clean infinite and extreme values from dataframe"""
        df = df.copy()
        
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
        
        # Cap extreme values for numerical columns
        for col in numerical_cols:
            if col not in ['customer_id', 'company_id', 'is_default']:
                if df[col].nunique() > 1:
                    q99_9 = df[col].quantile(0.999)
                    q00_1 = df[col].quantile(0.001)
                    df[col] = np.clip(df[col], q00_1, q99_9)
        
        return df
    
    def add_macroeconomic_features(self, df, segment):
        """Add macroeconomic features and calculated risk factors"""
        df = df.copy()
        
        # Add default macroeconomic features
        for feature, value in self.DEFAULT_MACRO.items():
            if feature not in df.columns:
                df[feature] = value
        
        # Calculate macro risk factor based on segment
        if segment == 'retail':
            # Unemployment affects retail customers more
            df['macro_risk_factor'] = (
                df['unemployment_rate'] * 2 + 
                df['vix'] / 50
            )
        elif segment == 'sme':
            # SMEs sensitive to credit conditions and GDP
            df['macro_risk_factor'] = (
                -df['gdp_growth'] * 3 +
                df['credit_spread'] * 20 +
                df['vix'] / 40
            )
        elif segment == 'corporate':
            # Corporates affected by interest rates and credit spreads
            df['macro_risk_factor'] = (
                df['interest_rate'] * 5 +
                df['credit_spread'] * 15 +
                df['vix'] / 60
            )
        
        return df
    
    def calculate_credit_scores(self, df, segment):
        """Calculate credit scores based on segment"""
        df = df.copy()
        
        if segment == 'retail':
            # Simplified FICO-like score calculation
            score = pd.Series([300] * len(df), index=df.index)
            
            # Payment history proxy (assume good if not specified)
            payment_history = df.get('payment_history', pd.Series([0.95] * len(df)))
            if not isinstance(payment_history, pd.Series):
                payment_history = pd.Series([payment_history] * len(df))
            score += payment_history * 315
            
            # Credit utilization (30% of score)
            utilization = df['utilization_rate']
            utilization_penalty = np.where(utilization < 0.3, 0, (utilization - 0.3) * 200)
            score += (1 - utilization) * 270 - utilization_penalty
            
            # Credit history length (15% of score)
            credit_history = df.get('credit_history_length', pd.Series([84] * len(df)))
            if not isinstance(credit_history, pd.Series):
                credit_history = pd.Series([credit_history] * len(df))
            score += np.minimum(credit_history / 240, 1) * 135
            
            # Credit mix (10% of score)
            num_accounts = df.get('num_accounts', pd.Series([5] * len(df)))
            if not isinstance(num_accounts, pd.Series):
                num_accounts = pd.Series([num_accounts] * len(df))
            score += np.minimum(num_accounts / 10, 1) * 90
            
            # Recent inquiries (10% of score)
            recent_inquiries = df.get('recent_inquiries', pd.Series([1] * len(df)))
            if not isinstance(recent_inquiries, pd.Series):
                recent_inquiries = pd.Series([recent_inquiries] * len(df))
            inquiry_penalty = np.minimum(recent_inquiries * 15, 90)
            score += 90 - inquiry_penalty
            
            # Debt-to-income adjustment
            dti_penalty = np.where(df['debt_to_income'] > 0.4, (df['debt_to_income'] - 0.4) * 100, 0)
            score -= dti_penalty
            
            df['credit_score'] = np.clip(score, 300, 850)
            
        elif segment == 'sme':
            # SME credit score calculation
            score = pd.Series([300] * len(df), index=df.index)
            
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
            
            df['sme_credit_score'] = np.clip(score, 300, 850)
            
        elif segment == 'corporate':
            # Corporate credit score based on rating
            rating_scores = {
                'AAA': 850, 'AA+': 820, 'AA': 800, 'AA-': 780,
                'A+': 760, 'A': 740, 'A-': 720, 'BBB+': 700,
                'BBB': 680, 'BBB-': 660, 'BB+': 640, 'BB': 620,
                'BB-': 600, 'B+': 580, 'B': 560, 'B-': 540,
                'CCC+': 520, 'CCC': 500, 'CCC-': 480
            }
            
            # Get base score from rating
            base_scores = df['credit_rating'].map(rating_scores).fillna(500)
            
            # Financial adjustments
            coverage_adj = np.clip((df['times_interest_earned'] - 5) * 5, -50, 50)
            leverage_adj = np.clip((1 - df['debt_to_equity']) * 20, -40, 40)
            profitability_adj = np.clip(df['roa'] * 200, -30, 30)
            liquidity_adj = np.clip((df['current_ratio'] - 1.5) * 10, -20, 20)
            
            # Market position adjustment
            position_adj_map = {'Leader': 15, 'Strong': 5, 'Average': 0, 'Weak': -15}
            position_adj = df['market_position'].map(position_adj_map).fillna(0)
            
            final_scores = (base_scores + coverage_adj + leverage_adj + 
                           profitability_adj + liquidity_adj + position_adj)
            
            df['corporate_credit_score'] = np.clip(final_scores, 300, 850)
            
            # Add industry if missing
            if 'industry' not in df.columns:
                df['industry'] = 'Technology'  # Default industry
        
        return df
    
    def engineer_retail_features(self, df):
        """Engineer retail features (exact copy from training)"""
        df = df.copy()
        
        # Safe financial ratios
        df['debt_service_ratio'] = np.where(
            df['income'] > 0, 
            df['current_debt'] / np.maximum(df['income'] / 12, 1), 
            0
        )
        df['savings_to_income_ratio'] = np.where(
            df['income'] > 0,
            df['savings_balance'] / df['income'],
            0
        )
        df['available_credit'] = np.maximum(0, df['credit_limit'] - df['current_debt'])
        df['available_credit_ratio'] = np.where(
            df['credit_limit'] > 0,
            df['available_credit'] / df['credit_limit'],
            0
        )
        
        # Additional features
        df['age_squared'] = df['age'] ** 2
        df['high_utilization'] = (df['utilization_rate'] > 0.8).astype(int)
        df['high_dti'] = (df['debt_to_income'] > 0.4).astype(int)
        
        return self.clean_infinite_values(df)
    
    def engineer_sme_features(self, df):
        """Engineer SME features (exact copy from training)"""
        df = df.copy()
        
        # Safe financial indicators
        df['revenue_per_employee'] = np.where(
            df['num_employees'] > 0,
            df['annual_revenue'] / df['num_employees'],
            0
        )
        df['cash_flow_margin'] = np.where(
            df['annual_revenue'] > 0,
            df['operating_cash_flow'] / df['annual_revenue'],
            0
        )
        
        # Risk indicators
        df['payment_risk_score'] = df['payment_delays_12m'] * 10 + df['days_past_due'] / 30
        
        return self.clean_infinite_values(df)
    
    def engineer_corporate_features(self, df):
        """Engineer corporate features (exact copy from training)"""
        df = df.copy()
        
        # Safe financial ratios
        df['cash_generation_ability'] = np.where(
            df['annual_revenue'] > 0,
            df['free_cash_flow'] / df['annual_revenue'],
            0
        )
        df['market_cap_to_revenue'] = np.where(
            (df['is_public'] == 1) & (df['annual_revenue'] > 0),
            np.minimum(df['market_cap'] / df['annual_revenue'], 100),
            0
        )
        
        # Scale indicators
        df['company_scale'] = (
            np.log1p(np.maximum(1, df['annual_revenue'])) + 
            np.log1p(np.maximum(1, df['num_employees']))
        )
        
        return self.clean_infinite_values(df)
    
    def prepare_model_data(self, data, segment):
        """Prepare data for modeling"""
        # Select features (exclude ID, target, and dates)
        exclude_cols = ['is_default', 'customer_id', 'company_id', 'observation_date', 'default_probability']
        feature_cols = [col for col in data.columns if col not in exclude_cols]
        X = data[feature_cols].copy()
        return X
    
    def predict_pd(self, data: pd.DataFrame, segment: str) -> Dict:
        """Predict PD for given data and segment"""
        if segment not in self.models or not self.models[segment]:
            raise ValueError(f"No models available for segment: {segment}")
        
        try:
            # Add macroeconomic features
            data_with_macro = self.add_macroeconomic_features(data.copy(), segment)
            
            # Calculate credit scores
            data_with_scores = self.calculate_credit_scores(data_with_macro, segment)
            
            # Apply feature engineering
            if segment == 'retail':
                data_engineered = self.engineer_retail_features(data_with_scores)
            elif segment == 'sme':
                data_engineered = self.engineer_sme_features(data_with_scores)
            elif segment == 'corporate':
                data_engineered = self.engineer_corporate_features(data_with_scores)
            else:
                raise ValueError(f"Unknown segment: {segment}")
            
            # Prepare data
            X = self.prepare_model_data(data_engineered, segment)
            
            # Preprocess
            if segment in self.preprocessors:
                X_processed = self.preprocessors[segment].transform(X)
            else:
                X_processed = X.values
            
            # Get predictions from both models
            logistic_pred = self.models[segment]['logistic'].predict_proba(X_processed)[:, 1]
            rf_pred = self.models[segment]['random_forest'].predict_proba(X_processed)[:, 1]
            
            # Ensemble prediction (average)
            ensemble_pred = (logistic_pred + rf_pred) / 2
            
            # Apply Basel III minimum floor
            ensemble_pred = np.maximum(ensemble_pred, self.BASEL_MIN_PD)
            
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
            logger.error(f"Data shape: {data.shape}")
            logger.error(f"Data columns: {list(data.columns)}")
            raise
    
    def predict_with_staging(self, data: pd.DataFrame, segment: str) -> Dict:
        """Predict PD with IFRS 9 staging and risk grades"""
        prediction_result = self.predict_pd(data, segment)
        pd_scores = np.array(prediction_result['pd_scores'])
        
        # Determine IFRS 9 stages
        stages = np.where(pd_scores <= self.IFRS9_STAGE1_THRESHOLD, 1,
                         np.where(pd_scores <= self.IFRS9_STAGE2_THRESHOLD, 2, 3))
        
        # Assign risk grades
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
    # Startup
    global predictor
    try:
        predictor = PDModelPredictor()
        logger.info("🚀 PD Model API started successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize models: {e}")
        predictor = None
    
    yield
    
    # Shutdown
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

# Setup Jinja2 templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Request/Response Models
class RetailCustomer(BaseModel):
    """Retail customer data model"""
    age: int = Field(..., ge=18, le=100, description="Customer age")
    income: float = Field(..., gt=0, description="Annual income")
    credit_score: int = Field(..., ge=300, le=850, description="Credit score")
    debt_to_income: float = Field(..., ge=0, le=5, description="Debt-to-income ratio")
    utilization_rate: float = Field(..., ge=0, le=1, description="Credit utilization rate")
    employment_status: str = Field(default="Full-time", description="Employment status")
    gender: str = Field(default="M", description="Gender")
    education_level: str = Field(default="Bachelor", description="Education level")
    employment_tenure: float = Field(default=3.0, description="Employment tenure")
    years_at_address: float = Field(default=2.0, description="Years at address")
    homeowner: int = Field(default=1, description="Homeowner (1=yes, 0=no)")
    credit_history_length: int = Field(default=84, description="Credit history length")
    num_accounts: int = Field(default=5, description="Number of accounts")
    payment_history: float = Field(default=0.95, description="Payment history")
    recent_inquiries: int = Field(default=1, description="Recent inquiries")
    months_since_last_delinquency: float = Field(default=999, description="Months since delinquency")
    num_delinquent_accounts: int = Field(default=0, description="Delinquent accounts")
    num_bank_accounts: int = Field(default=2, description="Bank accounts")
    monthly_transactions: int = Field(default=45, description="Monthly transactions")

class SMECompany(BaseModel):
    """SME company data model"""
    industry: str = Field(..., description="Industry sector")
    years_in_business: float = Field(..., ge=0.1, le=100, description="Years in business")
    annual_revenue: float = Field(..., gt=0, description="Annual revenue")
    num_employees: int = Field(..., ge=1, le=500, description="Number of employees")
    current_ratio: float = Field(..., ge=0, le=10, description="Current ratio")
    debt_to_equity: float = Field(..., ge=0, le=10, description="Debt-to-equity ratio")
    interest_coverage: float = Field(..., ge=0, le=100, description="Interest coverage ratio")
    profit_margin: float = Field(..., ge=-1, le=1, description="Profit margin")
    operating_cash_flow: float = Field(..., description="Operating cash flow")
    working_capital: float = Field(..., description="Working capital")
    credit_utilization: float = Field(..., ge=0, le=1, description="Credit utilization rate")
    payment_delays_12m: int = Field(..., ge=0, le=12, description="Payment delays in 12 months")
    geographic_risk: str = Field(default="Medium", description="Geographic risk level")
    market_competition: str = Field(default="Medium", description="Market competition level")
    management_quality: float = Field(..., ge=1, le=10, description="Management quality score")
    days_past_due: int = Field(default=0, description="Days past due")
    # Calculated fields
    asset_turnover: float = Field(default=1.5, description="Asset turnover")
    primary_bank_relationship_years: float = Field(default=5.0, description="Banking relationship years")
    num_banking_products: int = Field(default=3, description="Number of banking products")
    credit_line_amount: float = Field(default=500000, description="Credit line amount")
    outstanding_loans: float = Field(default=200000, description="Outstanding loans")

class CorporateEntity(BaseModel):
    """Corporate entity data model"""
    annual_revenue: float = Field(..., gt=0, description="Annual revenue")
    num_employees: int = Field(..., ge=1, description="Number of employees")
    current_ratio: float = Field(..., ge=0, le=10, description="Current ratio")
    debt_to_equity: float = Field(..., ge=0, le=10, description="Debt-to-equity ratio")
    times_interest_earned: float = Field(..., ge=0, le=100, description="Interest coverage ratio")
    roa: float = Field(..., ge=-1, le=1, description="Return on assets")
    credit_rating: str = Field(..., description="Credit rating")
    market_position: str = Field(..., description="Market position")
    operating_cash_flow: float = Field(..., description="Operating cash flow")
    free_cash_flow: float = Field(..., description="Free cash flow")
    # Calculated and default fields
    years_established: int = Field(default=25, description="Years established")
    is_public: int = Field(default=1, description="Public company indicator")
    market_cap: float = Field(default=1000000000, description="Market capitalization")
    quick_ratio: float = Field(default=1.2, description="Quick ratio")
    cash_ratio: float = Field(default=0.8, description="Cash ratio")
    debt_to_assets: float = Field(default=0.4, description="Debt to assets")
    net_profit_margin: float = Field(default=0.1, description="Net profit margin")
    roe: float = Field(default=0.15, description="Return on equity")
    asset_turnover: float = Field(default=1.5, description="Asset turnover")
    inventory_turnover: float = Field(default=8, description="Inventory turnover")
    num_banking_relationships: int = Field(default=5, description="Banking relationships")
    primary_bank_relationship_years: float = Field(default=10.0, description="Primary bank relationship")
    total_credit_facilities: float = Field(default=100000000, description="Total credit facilities")
    committed_facilities: float = Field(default=80000000, description="Committed facilities")
    utilization_rate: float = Field(default=0.3, description="Utilization rate")
    outstanding_debt: float = Field(default=50000000, description="Outstanding debt")
    geographic_diversification: str = Field(default="Regional", description="Geographic diversification")
    regulatory_environment: str = Field(default="Medium", description="Regulatory environment")
    esg_score: int = Field(default=65, description="ESG score")

class PDResponse(BaseModel):
    """PD prediction response"""
    customer_id: Optional[str] = None
    segment: str
    pd_score: float = Field(..., description="Probability of default")
    risk_grade: str = Field(..., description="Risk grade (AAA to C)")
    ifrs9_stage: int = Field(..., description="IFRS 9 stage (1, 2, or 3)")
    basel_compliant: bool = Field(..., description="Basel III compliance")
    prediction_timestamp: str = Field(..., description="Prediction timestamp")
    model_version: str = "1.0.0"
    model_details: Dict = Field(..., description="Model execution details")

class BatchResponse(BaseModel):
    """Batch prediction response"""
    total_predictions: int
    successful_predictions: int
    failed_predictions: int
    processing_time_seconds: float
    results: List[Dict]
    summary_statistics: Dict

# Web Interface Routes
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page with model selection"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/retail", response_class=HTMLResponse)
async def retail_form(request: Request):
    """Retail customer scoring form"""
    return templates.TemplateResponse("retail.html", {"request": request})

@app.get("/sme", response_class=HTMLResponse)  
async def sme_form(request: Request):
    """SME company scoring form"""
    return templates.TemplateResponse("sme.html", {"request": request})

@app.get("/corporate", response_class=HTMLResponse)
async def corporate_form(request: Request):
    """Corporate entity scoring form"""
    return templates.TemplateResponse("corporate.html", {"request": request})

@app.get("/batch", response_class=HTMLResponse)
async def batch_form(request: Request):
    """Batch scoring form"""
    return templates.TemplateResponse("batch.html", {"request": request})

# API Routes
@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
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
    """API endpoint for retail customer prediction"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        # Convert to DataFrame with all required features
        customer_dict = customer.dict()
        
        # Add calculated fields
        customer_dict['credit_limit'] = customer_dict['income'] * 0.8
        customer_dict['current_debt'] = customer_dict['income'] * customer_dict['debt_to_income']
        customer_dict['savings_balance'] = customer_dict['income'] * 0.3
        customer_dict['checking_balance'] = customer_dict['income'] * 0.05
        customer_dict['avg_transaction_amount'] = customer_dict['income'] / customer_dict['monthly_transactions']
        
        data = pd.DataFrame([customer_dict])
        
        # Make prediction
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
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/api/predict/sme", response_model=PDResponse)
async def predict_sme_api(company: SMECompany, company_id: Optional[str] = None):
    """API endpoint for SME company prediction"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        # Convert to DataFrame
        company_dict = company.dict()
        data = pd.DataFrame([company_dict])
        
        # Make prediction
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
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/api/predict/corporate", response_model=PDResponse)
async def predict_corporate_api(entity: CorporateEntity, entity_id: Optional[str] = None):
    """API endpoint for corporate entity prediction"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        # Convert to DataFrame
        entity_dict = entity.dict()
        data = pd.DataFrame([entity_dict])
        
        # Make prediction
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
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/api/predict/batch", response_model=BatchResponse)
async def predict_batch(file: UploadFile = File(...), segment: str = Form(...)):
    """Batch prediction endpoint"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    start_time = datetime.now()
    
    try:
        # Validate file type
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are supported")
        
        # Validate segment
        if segment not in ['retail', 'sme', 'corporate']:
            raise HTTPException(status_code=400, detail="Invalid segment")
        
        # Read CSV file
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        if len(df) == 0:
            raise HTTPException(status_code=400, detail="Empty CSV file")
        
        # Process predictions
        results = []
        successful_predictions = 0
        failed_predictions = 0
        
        for idx, row in df.iterrows():
            try:
                # Convert row to DataFrame
                row_data = pd.DataFrame([row.to_dict()])
                
                # Make prediction
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
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Calculate summary statistics
        successful_results = [r for r in results if r['success']]
        if successful_results:
            pd_scores = [r['pd_score'] for r in successful_results]
            risk_grades = [r['risk_grade'] for r in successful_results]
            stages = [r['ifrs9_stage'] for r in successful_results]
            
            summary_stats = {
                'avg_pd_score': np.mean(pd_scores),
                'median_pd_score': np.median(pd_scores),
                'min_pd_score': np.min(pd_scores),
                'max_pd_score': np.max(pd_scores),
                'risk_grade_distribution': pd.Series(risk_grades).value_counts().to_dict(),
                'ifrs9_stage_distribution': pd.Series(stages).value_counts().to_dict()
            }
        else:
            summary_stats = {}
        
        return BatchResponse(
            total_predictions=len(df),
            successful_predictions=successful_predictions,
            failed_predictions=failed_predictions,
            processing_time_seconds=processing_time,
            results=results,
            summary_statistics=summary_stats
        )
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")

# Form submission endpoints (for web interface)
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
    """Handle retail form submission"""
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
        
        # Call API endpoint
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
    """Handle SME form submission"""
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
        
        # Call API endpoint
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
    """Handle corporate form submission"""
    try:
        entity = CorporateEntity(
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
        
        # Call API endpoint
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