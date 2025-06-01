#!/usr/bin/env python3
"""
CSV Template Creator for PD Model API
====================================
Creates properly formatted CSV templates for batch processing
"""

import pandas as pd
from pathlib import Path

def create_retail_template():
    """Create retail customer CSV template"""
    print("📄 Creating retail customer template...")
    
    # Template with sample data
    template_data = [
        {
            'age': 35,
            'income': 75000,
            'credit_score': 720,
            'debt_to_income': 0.30,
            'utilization_rate': 0.25,
            'employment_status': 'Full-time',
            'employment_tenure': 3.0,
            'years_at_address': 2.0,
            'num_accounts': 5,
            'monthly_transactions': 45
        },
        {
            'age': 42,
            'income': 95000,
            'credit_score': 780,
            'debt_to_income': 0.20,
            'utilization_rate': 0.15,
            'employment_status': 'Full-time',
            'employment_tenure': 7.0,
            'years_at_address': 5.0,
            'num_accounts': 8,
            'monthly_transactions': 60
        },
        {
            'age': 28,
            'income': 55000,
            'credit_score': 650,
            'debt_to_income': 0.45,
            'utilization_rate': 0.70,
            'employment_status': 'Part-time',
            'employment_tenure': 1.5,
            'years_at_address': 1.0,
            'num_accounts': 3,
            'monthly_transactions': 25
        }
    ]
    
    df = pd.DataFrame(template_data)
    df.to_csv('retail_template.csv', index=False)
    
    print(f"✅ Created retail_template.csv with {len(df)} sample rows")
    print("   Required columns:")
    for col in df.columns:
        print(f"   - {col}")
    
    return 'retail_template.csv'

def create_sme_template():
    """Create SME company CSV template"""
    print("\n📄 Creating SME company template...")
    
    template_data = [
        {
            'industry': 'Technology',
            'years_in_business': 8.0,
            'annual_revenue': 2000000,
            'num_employees': 25,
            'current_ratio': 1.6,
            'debt_to_equity': 1.2,
            'interest_coverage': 5.0,
            'profit_margin': 0.08,
            'operating_cash_flow': 180000,
            'working_capital': 300000,
            'credit_utilization': 0.35,
            'payment_delays_12m': 1,
            'geographic_risk': 'Low',
            'market_competition': 'Medium',
            'management_quality': 7.5,
            'days_past_due': 0
        },
        {
            'industry': 'Manufacturing',
            'years_in_business': 15.0,
            'annual_revenue': 5000000,
            'num_employees': 75,
            'current_ratio': 2.1,
            'debt_to_equity': 0.8,
            'interest_coverage': 8.0,
            'profit_margin': 0.12,
            'operating_cash_flow': 650000,
            'working_capital': 800000,
            'credit_utilization': 0.25,
            'payment_delays_12m': 0,
            'geographic_risk': 'Low',
            'market_competition': 'High',
            'management_quality': 8.5,
            'days_past_due': 0
        },
        {
            'industry': 'Retail Trade',
            'years_in_business': 3.0,
            'annual_revenue': 800000,
            'num_employees': 12,
            'current_ratio': 1.1,
            'debt_to_equity': 2.5,
            'interest_coverage': 2.0,
            'profit_margin': 0.03,
            'operating_cash_flow': 40000,
            'working_capital': 100000,
            'credit_utilization': 0.80,
            'payment_delays_12m': 3,
            'geographic_risk': 'Medium',
            'market_competition': 'High',
            'management_quality': 6.0,
            'days_past_due': 30
        }
    ]
    
    df = pd.DataFrame(template_data)
    df.to_csv('sme_template.csv', index=False)
    
    print(f"✅ Created sme_template.csv with {len(df)} sample rows")
    print("   Required columns:")
    for col in df.columns:
        print(f"   - {col}")
    
    return 'sme_template.csv'

def create_corporate_template():
    """Create corporate entity CSV template"""
    print("\n📄 Creating corporate entity template...")
    
    template_data = [
        {
            'industry': 'Technology',
            'annual_revenue': 1000000000,
            'num_employees': 5000,
            'current_ratio': 1.5,
            'debt_to_equity': 0.8,
            'times_interest_earned': 8.0,
            'roa': 0.08,
            'credit_rating': 'A',
            'market_position': 'Strong',
            'operating_cash_flow': 150000000,
            'free_cash_flow': 100000000
        },
        {
            'industry': 'Healthcare',
            'annual_revenue': 2500000000,
            'num_employees': 12000,
            'current_ratio': 2.2,
            'debt_to_equity': 0.4,
            'times_interest_earned': 15.0,
            'roa': 0.12,
            'credit_rating': 'AA',
            'market_position': 'Leader',
            'operating_cash_flow': 400000000,
            'free_cash_flow': 300000000
        },
        {
            'industry': 'Energy',
            'annual_revenue': 800000000,
            'num_employees': 3000,
            'current_ratio': 1.2,
            'debt_to_equity': 1.5,
            'times_interest_earned': 3.0,
            'roa': 0.04,
            'credit_rating': 'BBB',
            'market_position': 'Average',
            'operating_cash_flow': 80000000,
            'free_cash_flow': 50000000
        }
    ]
    
    df = pd.DataFrame(template_data)
    df.to_csv('corporate_template.csv', index=False)
    
    print(f"✅ Created corporate_template.csv with {len(df)} sample rows")
    print("   Required columns:")
    for col in df.columns:
        print(f"   - {col}")
    
    return 'corporate_template.csv'

def create_empty_templates():
    """Create empty templates with just headers"""
    print("\n📄 Creating empty templates (headers only)...")
    
    # Retail empty template
    retail_columns = [
        'age', 'income', 'credit_score', 'debt_to_income', 'utilization_rate',
        'employment_status', 'employment_tenure', 'years_at_address', 
        'num_accounts', 'monthly_transactions'
    ]
    pd.DataFrame(columns=retail_columns).to_csv('retail_empty_template.csv', index=False)
    
    # SME empty template
    sme_columns = [
        'industry', 'years_in_business', 'annual_revenue', 'num_employees',
        'current_ratio', 'debt_to_equity', 'interest_coverage', 'profit_margin',
        'operating_cash_flow', 'working_capital', 'credit_utilization',
        'payment_delays_12m', 'geographic_risk', 'market_competition',
        'management_quality', 'days_past_due'
    ]
    pd.DataFrame(columns=sme_columns).to_csv('sme_empty_template.csv', index=False)
    
    # Corporate empty template
    corporate_columns = [
        'industry', 'annual_revenue', 'num_employees', 'current_ratio',
        'debt_to_equity', 'times_interest_earned', 'roa', 'credit_rating',
        'market_position', 'operating_cash_flow', 'free_cash_flow'
    ]
    pd.DataFrame(columns=corporate_columns).to_csv('corporate_empty_template.csv', index=False)
    
    print("✅ Created empty templates:")
    print("   - retail_empty_template.csv")
    print("   - sme_empty_template.csv") 
    print("   - corporate_empty_template.csv")

def create_field_descriptions():
    """Create a field description file"""
    print("\n📄 Creating field descriptions...")
    
    descriptions = {
        'RETAIL FIELDS': {
            'age': 'Customer age (18-100 years)',
            'income': 'Annual gross income in USD',
            'credit_score': 'FICO score (300-850)',
            'debt_to_income': 'Total debt / annual income (0.0-5.0)',
            'utilization_rate': 'Credit used / credit available (0.0-1.0)',
            'employment_status': 'Full-time, Part-time, Self-employed, Unemployed, Retired',
            'employment_tenure': 'Years with current employer',
            'years_at_address': 'Years at current address',
            'num_accounts': 'Number of credit accounts',
            'monthly_transactions': 'Average monthly transactions'
        },
        'SME FIELDS': {
            'industry': 'Business industry sector',
            'years_in_business': 'Years since company establishment',
            'annual_revenue': 'Annual revenue in USD',
            'num_employees': 'Total number of employees',
            'current_ratio': 'Current assets / current liabilities',
            'debt_to_equity': 'Total debt / total equity',
            'interest_coverage': 'EBIT / interest expense',
            'profit_margin': 'Net profit / revenue (-1.0 to 1.0)',
            'operating_cash_flow': 'Annual operating cash flow',
            'working_capital': 'Current assets - current liabilities',
            'credit_utilization': 'Used credit / available credit (0.0-1.0)',
            'payment_delays_12m': 'Payment delays in past 12 months',
            'geographic_risk': 'Low, Medium, High',
            'market_competition': 'Low, Medium, High',
            'management_quality': 'Management score (1-10)',
            'days_past_due': 'Current days past due (0, 30, 60, 90, 120)'
        },
        'CORPORATE FIELDS': {
            'industry': 'Corporate industry sector',
            'annual_revenue': 'Annual revenue in USD (minimum $50M)',
            'num_employees': 'Total number of employees',
            'current_ratio': 'Current assets / current liabilities',
            'debt_to_equity': 'Total debt / total equity',
            'times_interest_earned': 'EBIT / interest expense',
            'roa': 'Return on assets (-1.0 to 1.0)',
            'credit_rating': 'AAA, AA+, AA, AA-, A+, A, A-, BBB+, BBB, BBB-, BB+, BB, BB-, B+, B, B-',
            'market_position': 'Leader, Strong, Average, Weak',
            'operating_cash_flow': 'Annual operating cash flow',
            'free_cash_flow': 'Operating cash flow - capital expenditures'
        }
    }
    
    with open('field_descriptions.txt', 'w') as f:
        f.write("PD MODEL API - FIELD DESCRIPTIONS\n")
        f.write("=" * 50 + "\n\n")
        
        for section, fields in descriptions.items():
            f.write(f"{section}\n")
            f.write("-" * len(section) + "\n")
            for field, desc in fields.items():
                f.write(f"{field:25} : {desc}\n")
            f.write("\n")
    
    print("✅ Created field_descriptions.txt")

def main():
    """Main function to create all templates"""
    print("🗂️ PD Model API - CSV Template Creator")
    print("=" * 50)
    
    # Create sample templates
    retail_file = create_retail_template()
    sme_file = create_sme_template()
    corporate_file = create_corporate_template()
    
    # Create empty templates
    create_empty_templates()
    
    # Create field descriptions
    create_field_descriptions()
    
    print(f"\n{'='*50}")
    print("✅ ALL TEMPLATES CREATED SUCCESSFULLY!")
    print(f"{'='*50}")
    
    print("\n📋 Sample Templates (with example data):")
    print("   - retail_template.csv")
    print("   - sme_template.csv")
    print("   - corporate_template.csv")
    
    print("\n📋 Empty Templates (headers only):")
    print("   - retail_empty_template.csv")
    print("   - sme_empty_template.csv")
    print("   - corporate_empty_template.csv")
    
    print("\n📋 Documentation:")
    print("   - field_descriptions.txt")
    
    print("\n💡 Usage Instructions:")
    print("1. Use sample templates to see data format")
    print("2. Use empty templates to add your own data")
    print("3. Check field_descriptions.txt for field details")
    print("4. Upload CSV files to http://localhost:8000/batch")
    
    print("\n🚀 Start the API with: python app.py")

if __name__ == "__main__":
    main()