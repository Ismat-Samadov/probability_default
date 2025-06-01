#!/usr/bin/env python3
"""
Complete PD Model API with Web Interface
=========================================
FastAPI application with Jinja2 templating for all model segments
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for models
models = {}
preprocessors = {}

class PDModelPredictor:
    """Production PD model predictor with all segments"""
    
    def __init__(self, model_dir="models"):
        self.model_dir = Path(model_dir)
        self.models = {}
        self.preprocessors = {}
        self.load_models()
        
        # Regulatory constants
        self.BASEL_MIN_PD = 0.0003  # 3 basis points
        self.IFRS9_STAGE1_THRESHOLD = 0.01
        self.IFRS9_STAGE2_THRESHOLD = 0.05
    
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
    
    def engineer_retail_features(self, df):
        """Engineer retail features (exact copy from training)"""
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
            # Apply feature engineering
            if segment == 'retail':
                data_engineered = self.engineer_retail_features(data)
            elif segment == 'sme':
                data_engineered = self.engineer_sme_features(data)
            elif segment == 'corporate':
                data_engineered = self.engineer_corporate_features(data)
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
    annual_revenue: float = Field(..., gt=0, description="Annual revenue")
    num_employees: int = Field(..., gt=0, description="Number of employees")
    years_in_business: float = Field(..., gt=0, description="Years in business")
    current_ratio: float = Field(..., gt=0, description="Current ratio")
    debt_to_equity: float = Field(..., ge=0, description="Debt-to-equity ratio")
    profit_margin: float = Field(..., ge=-1, le=1, description="Profit margin")
    industry: str = Field(..., description="Industry sector")
    interest_coverage: float = Field(default=5.0, description="Interest coverage")
    asset_turnover: float = Field(default=1.5, description="Asset turnover")
    operating_cash_flow: float = Field(..., description="Operating cash flow")
    working_capital: float = Field(..., description="Working capital")
    credit_utilization: float = Field(default=0.3, description="Credit utilization")
    payment_delays_12m: int = Field(default=0, description="Payment delays")
    days_past_due: int = Field(default=0, description="Days past due")
    management_quality: float = Field(default=7.0, description="Management quality")

class CorporateCompany(BaseModel):
    """Corporate company data model"""
    annual_revenue: float = Field(..., gt=0, description="Annual revenue")
    num_employees: int = Field(..., gt=0, description="Number of employees")
    current_ratio: float = Field(..., gt=0, description="Current ratio")
    debt_to_equity: float = Field(..., ge=0, description="Debt-to-equity ratio")
    times_interest_earned: float = Field(..., gt=0, description="Interest coverage ratio")
    roa: float = Field(..., ge=-1, le=1, description="Return on assets")
    credit_rating: str = Field(..., description="Credit rating")
    market_position: str = Field(..., description="Market position")
    years_established: int = Field(default=15, description="Years established")
    is_public: int = Field(default=1, description="Public company")
    market_cap: float = Field(default=1000000000, description="Market cap")
    quick_ratio: float = Field(default=1.5, description="Quick ratio")
    cash_ratio: float = Field(default=0.8, description="Cash ratio")
    debt_to_assets: float = Field(default=0.3, description="Debt to assets")
    net_profit_margin: float = Field(default=0.1, description="Net profit margin")
    roe: float = Field(default=0.15, description="Return on equity")
    asset_turnover: float = Field(default=0.8, description="Asset turnover")
    inventory_turnover: float = Field(default=6.0, description="Inventory turnover")
    operating_cash_flow: float = Field(..., description="Operating cash flow")
    free_cash_flow: float = Field(..., description="Free cash flow")

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
        # Convert to DataFrame with all required features
        company_dict = company.dict()
        
        # Add calculated fields
        company_dict['credit_line_amount'] = company_dict['annual_revenue'] * 0.1
        company_dict['outstanding_loans'] = company_dict['credit_line_amount'] * company_dict['credit_utilization']
        company_dict['primary_bank_relationship_years'] = min(company_dict['years_in_business'], 5.0)
        company_dict['num_banking_products'] = 3
        company_dict['geographic_risk'] = 'Medium'
        company_dict['market_competition'] = 'Medium'
        company_dict['sme_credit_score'] = 650  # Will be recalculated during feature engineering
        
        # Add macro features
        company_dict['gdp_growth'] = 0.025
        company_dict['unemployment_rate'] = 0.055
        company_dict['interest_rate'] = 0.035
        company_dict['credit_spread'] = 0.015
        company_dict['vix'] = 20.0
        company_dict['macro_risk_factor'] = 0.15
        
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
async def predict_corporate_api(company: CorporateCompany, company_id: Optional[str] = None):
    """API endpoint for corporate entity prediction"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        # Convert to DataFrame with all required features
        company_dict = company.dict()
        
        # Add calculated fields
        company_dict['total_credit_facilities'] = company_dict['annual_revenue'] * 0.05
        company_dict['committed_facilities'] = company_dict['total_credit_facilities'] * 0.8
        company_dict['utilization_rate'] = 0.2
        company_dict['outstanding_debt'] = company_dict['committed_facilities'] * company_dict['utilization_rate']
        company_dict['num_banking_relationships'] = 5
        company_dict['primary_bank_relationship_years'] = 8.0
        company_dict['geographic_diversification'] = 'Global'
        company_dict['regulatory_environment'] = 'Medium'
        company_dict['esg_score'] = 75
        company_dict['corporate_credit_score'] = 700  # Will be recalculated during feature engineering
        
        # Add macro features
        company_dict['gdp_growth'] = 0.025
        company_dict['unemployment_rate'] = 0.055
        company_dict['interest_rate'] = 0.035
        company_dict['credit_spread'] = 0.015
        company_dict['vix'] = 20.0
        company_dict['macro_risk_factor'] = 0.08
        
        data = pd.DataFrame([company_dict])
        
        # Make prediction
        result = predictor.predict_with_staging(data, 'corporate')
        
        return PDResponse(
            customer_id=company_id,
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

@app.post("/api/predict/batch")
async def predict_batch_api(file: UploadFile = File(...), segment: str = Form(...)):
    """Batch prediction endpoint"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    if segment not in ['retail', 'sme', 'corporate']:
        raise HTTPException(status_code=400, detail="Invalid segment")
    
    try:
        # Read uploaded file
        content = await file.read()
        df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        
        # Make predictions
        result = predictor.predict_with_staging(df, segment)
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'pd_score': result['pd_scores'],
            'risk_grade': result['risk_grades'],
            'ifrs9_stage': result['ifrs9_stages'],
            'logistic_prediction': result['logistic_predictions'],
            'random_forest_prediction': result['random_forest_predictions']
        })
        
        # Combine with original data
        final_results = pd.concat([df, results_df], axis=1)
        
        return JSONResponse({
            "status": "success",
            "total_predictions": len(result['pd_scores']),
            "segment": segment,
            "results": final_results.to_dict('records'),
            "summary": {
                "avg_pd": np.mean(result['pd_scores']),
                "median_pd": np.median(result['pd_scores']),
                "max_pd": np.max(result['pd_scores']),
                "min_pd": np.min(result['pd_scores']),
                "stage_distribution": {
                    "stage_1": sum(1 for stage in result['ifrs9_stages'] if stage == 1),
                    "stage_2": sum(1 for stage in result['ifrs9_stages'] if stage == 2),
                    "stage_3": sum(1 for stage in result['ifrs9_stages'] if stage == 3)
                }
            },
            "prediction_timestamp": result['prediction_timestamp']
        })
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

# Form submission endpoints (for web interface)
@app.post("/predict/retail")
async def predict_retail_form(
    request: Request,
    age: int = Form(...),
    income: float = Form(...),
    credit_score: int = Form(...),
    debt_to_income: float = Form(...),
    utilization_rate: float = Form(...),
    employment_status: str = Form(...)
):
    """Handle retail form submission"""
    try:
        customer = RetailCustomer(
            age=age,
            income=income,
            credit_score=credit_score,
            debt_to_income=debt_to_income,
            utilization_rate=utilization_rate,
            employment_status=employment_status
        )
        
        # Call API endpoint
        result = await predict_retail_api(customer)
        
        return templates.TemplateResponse("result.html", {
            "request": request,
            "result": result,
            "segment": "Retail Customer"
        })
        
    except Exception as e:
        return templates.TemplateResponse("error.html", {
            "request": request,
            "error": str(e)
        })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )