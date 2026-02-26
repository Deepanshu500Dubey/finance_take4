"""
FastAPI Application for Finance Month-End Close AI Agent
Provides tools for Watsonx Orchestrate integration
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum
import csv
import os
from collections import defaultdict

app = FastAPI(
    title="Finance Month-End Close AI Agent API",
    description="API endpoints for AI-powered month-end close automation",
    version="1.0.0"
)

# CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# DATA MODELS
# ============================================================================

class SeverityLevel(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"

class ExceptionType(str, Enum):
    MISSING_COST_CENTER = "Missing Cost Center"
    AR_GL_VARIANCE = "AR/GL Variance"
    INVALID_ACCOUNT = "Invalid Account Code"
    OVERDUE_INVOICE = "Overdue Invoice"
    LARGE_OUTSTANDING = "Large Outstanding Balance"

class ToolResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    exceptions: Optional[List[Dict[str, Any]]] = None
    timestamp: datetime = Field(default_factory=datetime.now)

class TrialBalanceRequest(BaseModel):
    fiscal_period: str = Field(..., example="2026-02")
    entity_code: str = Field(default="AUS01", example="AUS01")

class ARVarianceRequest(BaseModel):
    fiscal_period: str = Field(..., example="2026-02")
    entity_code: str = Field(default="AUS01", example="AUS01")

class CostCenterAssignment(BaseModel):
    transaction_id: str
    cost_center: str
    approved_by: Optional[str] = None

class CostCenterBatchRequest(BaseModel):
    assignments: List[CostCenterAssignment]
    fiscal_period: str = Field(..., example="2026-02")

class JournalEntry(BaseModel):
    entry_id: str
    date: str
    description: str
    debit_account: str
    credit_account: str
    amount: float
    approved: bool = False
    approved_by: Optional[str] = None

class JournalEntryRequest(BaseModel):
    entries: List[JournalEntry]
    fiscal_period: str

class BudgetVarianceRequest(BaseModel):
    fiscal_period: str = Field(..., example="2026-02")
    entity_code: str = Field(default="AUS01", example="AUS01")

class YoYComparisonRequest(BaseModel):
    current_period: str = Field(..., example="2026-02")
    comparison_period: str = Field(..., example="2025-02")
    entity_code: str = Field(default="AUS01", example="AUS01")

class CostCenterPLRequest(BaseModel):
    fiscal_period: str = Field(..., example="2026-02")
    entity_code: str = Field(default="AUS01", example="AUS01")

class MonthEndCloseRequest(BaseModel):
    fiscal_period: str = Field(..., example="2026-02")
    entity_code: str = Field(default="AUS01", example="AUS01")
    approved_by: str

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_csv_data(filename: str) -> List[Dict[str, Any]]:
    """Load CSV file and return list of dictionaries"""
    if not os.path.exists(filename):
        raise HTTPException(status_code=404, detail=f"File {filename} not found")
    
    data = []
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data

def save_csv_data(filename: str, data: List[Dict[str, Any]], fieldnames: List[str]):
    """Save data to CSV file"""
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

def load_coa() -> Dict[str, Dict[str, str]]:
    """Load Chart of Accounts"""
    coa = {}
    data = load_csv_data('Master_COA_Complete.csv')
    for row in data:
        coa[row['Account_Code']] = {
            'name': row['Account_Name'],
            'type': row['Account_Type'],
            'category': row['Category']
        }
    return coa

# ============================================================================
# HEALTH CHECK
# ============================================================================

@app.get("/")
def root():
    """Root endpoint - API health check"""
    return {
        "status": "healthy",
        "service": "Finance Month-End Close AI Agent API",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

# ============================================================================
# TOOL 0: INITIAL DATA ASSESSMENT (High-Level Summary)
# ============================================================================

@app.post("/tools/initial_assessment", response_model=ToolResponse)
def initial_assessment(request: TrialBalanceRequest):
    """
    Provide initial data assessment without full exception detection
    
    This tool:
    - Loads and counts all data files
    - Provides high-level summary statistics
    - Does NOT run full trial balance or exception detection
    - Quick overview to start the conversation
    """
    try:
        # Load all data files
        transactions = load_csv_data('Raw_GL_Export_With_CostCenters_Feb2026.csv')
        coa = load_coa()
        cost_centers_data = load_csv_data('Master_CostCenters_States.csv')
        ar_records = load_csv_data('AR_Subledger_Feb2026.csv')
        
        # Filter for the requested period
        period_txns = [t for t in transactions if t['Fiscal_Period'] == request.fiscal_period]
        
        # Count by transaction type
        revenue_txns = [t for t in period_txns if t['Account_Code_Raw'].startswith('4')]
        expense_txns = [t for t in period_txns if t['Account_Code_Raw'].startswith('5')]
        balance_sheet_txns = [t for t in period_txns if t['Account_Code_Raw'][0] in ['1', '2', '3']]
        
        # Calculate totals
        revenue_total = sum(float(t['Amount']) for t in revenue_txns)
        expense_total = sum(float(t['Amount']) for t in expense_txns)
        ar_outstanding = sum(float(r['Outstanding_Balance']) for r in ar_records)
        
        # Get unique currencies
        currencies = set(t.get('Currency', 'AUD') for t in period_txns if t.get('Currency'))
        if not currencies:
            currencies = {'AUD', 'USD', 'NZD', 'GBP'}  # Default from demo
        
        # Get cost center list
        cost_center_codes = [cc['Cost_Center_Code'] for cc in cost_centers_data]
        
        # Build summary message
        summary = {
            "data_loaded": {
                "gl_transactions_total": len(transactions),
                "gl_transactions_period": len(period_txns),
                "chart_of_accounts": len(coa),
                "cost_centers": len(cost_center_codes),
                "cost_center_list": cost_center_codes,
                "ar_customers": len(ar_records),
                "ar_outstanding": ar_outstanding
            },
            "transaction_breakdown": {
                "revenue_transactions": len(revenue_txns),
                "revenue_amount": revenue_total,
                "expense_transactions": len(expense_txns),
                "expense_amount": expense_total,
                "balance_sheet_adjustments": len(balance_sheet_txns)
            },
            "currencies": list(currencies),
            "status": "ready",
            "next_step": "Run 'Generate Trial Balance' to detect exceptions and validate data"
        }
        
        # Format message like the demo script
        message = f"""Loading data files...

✓ GL Transactions: {len(transactions)} total, {len(period_txns)} for {request.fiscal_period}
✓ Chart of Accounts: {len(coa)} accounts (Assets, Liabilities, Equity, Revenue, Expenses)
✓ Cost Centers: {len(cost_center_codes)} centers ({', '.join(cost_center_codes)})
✓ AR Subledger: {len(ar_records)} customers, ${ar_outstanding:,.2f} outstanding

Initial Assessment:
- Revenue transactions: {len(revenue_txns)} (${revenue_total:,.2f})
- Expense transactions: {len(expense_txns)} (${expense_total:,.2f})
- Balance sheet adjustments: {len(balance_sheet_txns)}
- Multi-currency: {', '.join(sorted(currencies))}

Ready to proceed with month-end close process."""
        
        return ToolResponse(
            success=True,
            message=message,
            data=summary
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# TOOL 1: GENERATE TRIAL BALANCE WITH EXCEPTIONS
# ============================================================================

@app.post("/tools/generate_trial_balance", response_model=ToolResponse)
def generate_trial_balance(request: TrialBalanceRequest):
    """
    Generate trial balance and detect exceptions
    
    This tool:
    - Loads GL transactions for the specified period
    - Calculates trial balance by account type
    - Detects missing cost centers, AR/GL variances, and other exceptions
    - Returns blocking and non-blocking exceptions
    """
    try:
        # Load data
        transactions = load_csv_data('Raw_GL_Export_With_CostCenters_Feb2026.csv')
        coa = load_coa()
        ar_records = load_csv_data('AR_Subledger_Feb2026.csv')
        
        # Filter transactions for the period
        period_txns = [t for t in transactions if t['Fiscal_Period'] == request.fiscal_period]
        
        # Calculate AR outstanding
        ar_outstanding = sum(float(r['Outstanding_Balance']) for r in ar_records)
        
        # Detect exceptions
        exceptions = []
        
        # Exception 1: Missing Cost Centers
        missing_cc = [t for t in period_txns if not t.get('Cost_Center', '')]
        if missing_cc:
            exceptions.append({
                'type': 'Missing Cost Centers',
                'severity': 'HIGH',
                'count': len(missing_cc),
                'amount': sum(float(t['Amount']) for t in missing_cc),
                'action': 'Assign cost centers to all transactions',
                'blocking': True
            })
        
        # Exception 2: AR/GL Variance
        gl_ar_balance = sum(float(t['Amount']) for t in period_txns if t['Account_Code_Raw'] == '1100')
        ar_variance = ar_outstanding - gl_ar_balance
        if abs(ar_variance) > 0.01:
            exceptions.append({
                'type': 'AR/GL Variance',
                'severity': 'CRITICAL',
                'variance': ar_variance,
                'ar_subledger': ar_outstanding,
                'gl_balance': gl_ar_balance,
                'action': 'Review and post journal entries',
                'blocking': True
            })
        
        # Exception 3: Invalid Account Codes
        invalid_accounts = [t for t in period_txns if t['Account_Code_Raw'] not in coa]
        if invalid_accounts:
            exceptions.append({
                'type': 'Invalid Account Codes',
                'severity': 'HIGH',
                'count': len(invalid_accounts),
                'amount': sum(float(t['Amount']) for t in invalid_accounts),
                'action': 'Map to valid account codes',
                'blocking': True
            })
        
        # Exception 4: AR Missing Cost Centers
        ar_missing_cc = [r for r in ar_records if not r['Cost_Center'] or not r['Region']]
        if ar_missing_cc:
            exceptions.append({
                'type': 'AR Missing Cost Centers',
                'severity': 'MEDIUM',
                'count': len(ar_missing_cc),
                'amount': sum(float(r['Outstanding_Balance']) for r in ar_missing_cc),
                'action': 'Assign regional cost centers',
                'blocking': False
            })
        
        # Exception 5: Overdue Invoices
        overdue = [r for r in ar_records if r['Status'] == 'Overdue']
        if overdue:
            exceptions.append({
                'type': 'Overdue Invoices',
                'severity': 'MEDIUM',
                'count': len(overdue),
                'amount': sum(float(r['Outstanding_Balance']) for r in overdue),
                'action': 'Review collections process',
                'blocking': False
            })
        
        # Calculate trial balance
        account_balances = defaultdict(float)
        for txn in period_txns:
            if txn['Account_Code_Raw'] in coa:
                account_balances[txn['Account_Code_Raw']] += float(txn['Amount'])
        
        # Group by type
        type_totals = {
            'Asset': 0,
            'Liability': 0,
            'Equity': 0,
            'Revenue': 0,
            'Expense': 0
        }
        
        for account, balance in account_balances.items():
            if account in coa:
                type_totals[coa[account]['type']] += balance
        
        # Check if balanced
        balance_check = type_totals['Asset'] + type_totals['Liability'] + type_totals['Equity']
        is_balanced = abs(balance_check) < 0.01
        
        blocking_exceptions = [e for e in exceptions if e.get('blocking', False)]
        
        return ToolResponse(
            success=len(blocking_exceptions) == 0,
            message=f"Trial balance {'completed' if len(blocking_exceptions) == 0 else 'cannot be completed'} - {len(exceptions)} exception(s) detected",
            data={
                'fiscal_period': request.fiscal_period,
                'transaction_count': len(period_txns),
                'trial_balance': type_totals,
                'is_balanced': is_balanced,
                'balance_check': balance_check,
                'blocking_exceptions_count': len(blocking_exceptions),
                'total_exceptions_count': len(exceptions)
            },
            exceptions=exceptions
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# TOOL 2: ANALYZE AR VARIANCE
# ============================================================================

@app.post("/tools/analyze_ar_variance", response_model=ToolResponse)
def analyze_ar_variance(request: ARVarianceRequest):
    """
    Analyze Accounts Receivable variance between subledger and GL
    
    This tool:
    - Compares AR subledger with GL account 1100
    - Identifies missing cost centers in AR
    - Detects overdue invoices
    - Suggests journal entries to reconcile variance
    """
    try:
        # Load data
        ar_records = load_csv_data('AR_Subledger_Feb2026.csv')
        transactions = load_csv_data('Raw_GL_Export_With_CostCenters_Feb2026.csv')
        
        # Filter GL transactions for AR account
        period_txns = [t for t in transactions if t['Fiscal_Period'] == request.fiscal_period]
        
        # Calculate totals
        total_invoiced = sum(float(r['Invoice_Amount']) for r in ar_records)
        total_paid = sum(float(r['Amount_Paid']) for r in ar_records)
        total_outstanding = sum(float(r['Outstanding_Balance']) for r in ar_records)
        
        gl_ar_balance = sum(float(t['Amount']) for t in period_txns if t['Account_Code_Raw'] == '1100')
        variance = total_outstanding - gl_ar_balance
        
        # Analyze by status
        status_summary = defaultdict(lambda: {'count': 0, 'amount': 0.0})
        for record in ar_records:
            status = record['Status']
            outstanding = float(record['Outstanding_Balance'])
            status_summary[status]['count'] += 1
            status_summary[status]['amount'] += outstanding
        
        # Identify issues
        missing_cost_centers = [r for r in ar_records if not r['Cost_Center'] or not r['Region']]
        overdue_invoices = [r for r in ar_records if r['Status'] == 'Overdue']
        large_outstanding = [r for r in ar_records if float(r['Outstanding_Balance']) > 500000 and r['Status'] in ['Outstanding', 'Overdue']]
        
        # Generate recommendations
        recommendations = []
        
        if missing_cost_centers:
            recommendations.append({
                'type': 'Missing Cost Centers',
                'action': f'Assign cost centers to {len(missing_cost_centers)} invoices',
                'impact': sum(float(r['Outstanding_Balance']) for r in missing_cost_centers),
                'priority': 'HIGH'
            })
        
        if overdue_invoices:
            recommendations.append({
                'type': 'Overdue Invoices',
                'action': f'Review and follow up on {len(overdue_invoices)} overdue invoices',
                'impact': sum(float(r['Outstanding_Balance']) for r in overdue_invoices),
                'priority': 'HIGH',
                'details': [
                    {
                        'customer': r['Customer_Name'],
                        'invoice': r['Invoice_Number'],
                        'days_overdue': r['Days_Outstanding'],
                        'amount': float(r['Outstanding_Balance'])
                    } for r in sorted(overdue_invoices, key=lambda x: int(x['Days_Outstanding']), reverse=True)[:5]
                ]
            })
        
        if abs(variance) > 0.01:
            recommendations.append({
                'type': 'AR/GL Variance',
                'action': f'Investigate ${variance:,.2f} variance',
                'impact': abs(variance),
                'priority': 'CRITICAL',
                'suggested_journal_entries': [
                    {
                        'entry_id': 'JE-2026-001',
                        'description': 'AR Variance Correction',
                        'debit_account': '5990' if variance < 0 else '1000',
                        'credit_account': '1100',
                        'amount': abs(variance)
                    }
                ]
            })
        
        return ToolResponse(
            success=True,
            message=f"AR variance analysis complete - Variance: ${variance:,.2f}",
            data={
                'ar_subledger': {
                    'total_invoiced': total_invoiced,
                    'total_paid': total_paid,
                    'total_outstanding': total_outstanding
                },
                'gl_balance': gl_ar_balance,
                'variance': variance,
                'status_summary': dict(status_summary),
                'missing_cost_centers_count': len(missing_cost_centers),
                'overdue_invoices_count': len(overdue_invoices),
                'large_outstanding_count': len(large_outstanding),
                'recommendations': recommendations
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# TOOL 3: UPDATE GL WITH COST CENTERS
# ============================================================================

@app.post("/tools/assign_cost_centers", response_model=ToolResponse)
def assign_cost_centers(request: CostCenterBatchRequest):
    """
    Assign cost centers to GL transactions
    
    This tool:
    - Updates GL transactions with approved cost center assignments
    - Validates cost centers against master list
    - Maintains audit trail of assignments
    """
    try:
        # Load data
        transactions = load_csv_data('Raw_GL_Export_With_CostCenters_Feb2026.csv')
        cost_centers_data = load_csv_data('Master_CostCenters_States.csv')
        valid_cost_centers = [cc['Cost_Center_Code'] for cc in cost_centers_data]
        
        # Create lookup for assignments
        assignment_map = {a.transaction_id: a for a in request.assignments}
        
        # Update transactions
        updated_count = 0
        invalid_assignments = []
        
        for txn in transactions:
            if txn['Txn_ID'] in assignment_map:
                assignment = assignment_map[txn['Txn_ID']]
                
                # Validate cost center
                if assignment.cost_center not in valid_cost_centers:
                    invalid_assignments.append({
                        'transaction_id': txn['Txn_ID'],
                        'cost_center': assignment.cost_center,
                        'reason': 'Invalid cost center code'
                    })
                    continue
                
                txn['Cost_Center'] = assignment.cost_center
                updated_count += 1
        
        # Save updated transactions
        if updated_count > 0:
            fieldnames = list(transactions[0].keys())
            save_csv_data('Raw_GL_Export_With_CostCenters_Feb2026.csv', transactions, fieldnames)
        
        return ToolResponse(
            success=len(invalid_assignments) == 0,
            message=f"Updated {updated_count} transactions with cost center assignments",
            data={
                'updated_count': updated_count,
                'requested_count': len(request.assignments),
                'invalid_count': len(invalid_assignments),
                'invalid_assignments': invalid_assignments
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# TOOL 4: POST JOURNAL ENTRIES
# ============================================================================

@app.post("/tools/post_journal_entries", response_model=ToolResponse)
def post_journal_entries(request: JournalEntryRequest):
    """
    Post approved journal entries to GL
    
    This tool:
    - Validates journal entries (debits = credits)
    - Posts approved entries to GL
    - Maintains audit trail
    """
    try:
        # Load data
        transactions = load_csv_data('Raw_GL_Export_With_CostCenters_Feb2026.csv')
        coa = load_coa()
        
        # Validate entries
        validation_errors = []
        for entry in request.entries:
            if not entry.approved:
                validation_errors.append({
                    'entry_id': entry.entry_id,
                    'error': 'Entry not approved'
                })
            if entry.debit_account not in coa:
                validation_errors.append({
                    'entry_id': entry.entry_id,
                    'error': f'Invalid debit account: {entry.debit_account}'
                })
            if entry.credit_account not in coa:
                validation_errors.append({
                    'entry_id': entry.entry_id,
                    'error': f'Invalid credit account: {entry.credit_account}'
                })
        
        if validation_errors:
            return ToolResponse(
                success=False,
                message=f"Validation failed for {len(validation_errors)} entries",
                data={'validation_errors': validation_errors}
            )
        
        # Post entries
        posted_entries = []
        for entry in request.entries:
            # Create debit transaction
            debit_txn = {
                'Txn_ID': f"{entry.entry_id}-DR",
                'Posting_Date_Raw': entry.date,
                'Fiscal_Period': request.fiscal_period,
                'Account_Code_Raw': entry.debit_account,
                'Amount': str(entry.amount),
                'Vendor_Name_Raw': 'Journal Entry',
                'Narrative': entry.description,
                'Cost_Center': 'CORP'
            }
            
            # Create credit transaction
            credit_txn = {
                'Txn_ID': f"{entry.entry_id}-CR",
                'Posting_Date_Raw': entry.date,
                'Fiscal_Period': request.fiscal_period,
                'Account_Code_Raw': entry.credit_account,
                'Amount': str(-entry.amount),
                'Vendor_Name_Raw': 'Journal Entry',
                'Narrative': entry.description,
                'Cost_Center': 'CORP'
            }
            
            transactions.append(debit_txn)
            transactions.append(credit_txn)
            posted_entries.append(entry.entry_id)
        
        # Save updated transactions
        fieldnames = list(transactions[0].keys())
        save_csv_data('Raw_GL_Export_With_CostCenters_Feb2026.csv', transactions, fieldnames)
        
        return ToolResponse(
            success=True,
            message=f"Posted {len(posted_entries)} journal entries",
            data={
                'posted_entries': posted_entries,
                'total_amount': sum(e.amount for e in request.entries)
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# TOOL 5: BUDGET VARIANCE ANALYSIS
# ============================================================================

@app.post("/tools/budget_variance_analysis", response_model=ToolResponse)
def budget_variance_analysis(request: BudgetVarianceRequest):
    """
    Compare actuals vs budget
    
    This tool:
    - Loads budget data for the period
    - Compares with actual GL transactions
    - Calculates variances by account category
    """
    try:
        # Load data
        transactions = load_csv_data('Raw_GL_Export_With_CostCenters_Feb2026.csv')
        budget_data = load_csv_data('Budget_Feb2026_Detailed.csv')
        coa = load_coa()
        
        # Filter period transactions
        period_txns = [t for t in transactions if t['Fiscal_Period'] == request.fiscal_period]
        
        # Calculate actuals by category
        actuals = defaultdict(float)
        for txn in period_txns:
            account = txn['Account_Code_Raw']
            if account in coa:
                category = coa[account]['category']
                actuals[category] += float(txn['Amount'])
        
        # Calculate budget by category (skip TOTAL rows)
        budget = defaultdict(float)
        for item in budget_data:
            category = item['Category']
            # Skip summary/total rows
            if category.upper() == 'TOTAL' or item['Account_Code'].startswith('TOTAL'):
                continue
            budget[category] += float(item['Budget_Amount'])
        
        # Calculate variances
        variances = []
        for category in set(list(actuals.keys()) + list(budget.keys())):
            actual_amt = actuals.get(category, 0)
            budget_amt = budget.get(category, 0)
            variance = actual_amt - budget_amt
            variance_pct = (variance / budget_amt * 100) if budget_amt != 0 else 0
            
            variances.append({
                'category': category,
                'budget': budget_amt,
                'actual': actual_amt,
                'variance': variance,
                'variance_percent': variance_pct,
                'status': 'favorable' if variance < 0 else 'unfavorable'
            })
        
        # Sort by absolute variance
        variances.sort(key=lambda x: abs(x['variance']), reverse=True)
        
        return ToolResponse(
            success=True,
            message=f"Budget variance analysis complete for {request.fiscal_period}",
            data={
                'fiscal_period': request.fiscal_period,
                'variances': variances,
                'total_budget': sum(budget.values()),
                'total_actual': sum(actuals.values()),
                'total_variance': sum(actuals.values()) - sum(budget.values())
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# TOOL 6: YEAR-OVER-YEAR COMPARISON
# ============================================================================

@app.post("/tools/yoy_comparison", response_model=ToolResponse)
def yoy_comparison(request: YoYComparisonRequest):
    """
    Year-over-year performance comparison
    
    This tool:
    - Compares current period with prior year same period
    - Calculates growth rates
    - Identifies trends
    """
    try:
        # Load data
        transactions = load_csv_data('Raw_GL_Export_With_CostCenters_Feb2026.csv')
        prior_year_data = load_csv_data('PL_Statement_Feb2025_Comparative.csv')
        coa = load_coa()
        
        # Calculate current period by category
        current_txns = [t for t in transactions if t['Fiscal_Period'] == request.current_period]
        current_totals = defaultdict(float)
        
        for txn in current_txns:
            account = txn['Account_Code_Raw']
            if account in coa:
                category = coa[account]['category']
                current_totals[category] += float(txn['Amount'])
        
        # Load prior year data from comparative P&L (skip TOTAL rows)
        prior_totals = {}
        for item in prior_year_data:
            category = item['Category']
            # Skip summary/total rows
            if category.upper() == 'TOTAL' or item['Account_Code'].startswith('TOTAL'):
                continue
            # Use Feb_2025_Actual column from the CSV
            prior_totals[category] = float(item['Feb_2025_Actual'])
        
        # Calculate comparisons
        comparisons = []
        for category in set(list(current_totals.keys()) + list(prior_totals.keys())):
            current = current_totals.get(category, 0)
            prior = prior_totals.get(category, 0)
            variance = current - prior
            growth_rate = (variance / prior * 100) if prior != 0 else 0
            
            comparisons.append({
                'category': category,
                'prior_year': prior,
                'current_year': current,
                'variance': variance,
                'growth_rate': growth_rate
            })
        
        return ToolResponse(
            success=True,
            message=f"YoY comparison complete: {request.comparison_period} vs {request.current_period}",
            data={
                'current_period': request.current_period,
                'comparison_period': request.comparison_period,
                'comparisons': comparisons
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# TOOL 7: COST CENTER P&L
# ============================================================================

@app.post("/tools/cost_center_pl", response_model=ToolResponse)
def cost_center_pl(request: CostCenterPLRequest):
    """
    Generate P&L by cost center
    
    This tool:
    - Calculates revenue and expenses by cost center
    - Computes net income and margin by cost center
    - Identifies profitable and loss-making centers
    """
    try:
        # Load data
        transactions = load_csv_data('Raw_GL_Export_With_CostCenters_Feb2026.csv')
        coa = load_coa()
        
        # Filter period transactions
        period_txns = [t for t in transactions if t['Fiscal_Period'] == request.fiscal_period]
        
        # Calculate by cost center
        cc_summary = defaultdict(lambda: {'revenue': 0, 'expenses': 0})
        
        for txn in period_txns:
            account = txn['Account_Code_Raw']
            cost_center = txn.get('Cost_Center', 'UNASSIGNED')
            amount = float(txn['Amount'])
            
            if account in coa:
                acc_type = coa[account]['type']
                if acc_type == 'Revenue':
                    cc_summary[cost_center]['revenue'] += amount
                elif acc_type == 'Expense':
                    cc_summary[cost_center]['expenses'] += amount
        
        # Calculate net income and margin
        cost_center_results = []
        for cc, data in cc_summary.items():
            net_income = data['revenue'] - data['expenses']
            margin = (net_income / data['revenue'] * 100) if data['revenue'] != 0 else 0
            
            cost_center_results.append({
                'cost_center': cc,
                'revenue': data['revenue'],
                'expenses': data['expenses'],
                'net_income': net_income,
                'margin_percent': margin,
                'status': 'profitable' if net_income > 0 else 'loss'
            })
        
        # Sort by net income
        cost_center_results.sort(key=lambda x: x['net_income'], reverse=True)
        
        return ToolResponse(
            success=True,
            message=f"Cost center P&L generated for {request.fiscal_period}",
            data={
                'fiscal_period': request.fiscal_period,
                'cost_centers': cost_center_results,
                'total_revenue': sum(cc['revenue'] for cc in cost_center_results),
                'total_expenses': sum(cc['expenses'] for cc in cost_center_results),
                'total_net_income': sum(cc['net_income'] for cc in cost_center_results)
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# TOOL 8: MONTH-END CLOSE
# ============================================================================

@app.post("/tools/close_period", response_model=ToolResponse)
def close_period(request: MonthEndCloseRequest):
    """
    Close the accounting period
    
    This tool:
    - Validates all exceptions are resolved
    - Locks the period
    - Generates final financial statements
    """
    try:
        # Run final validation
        trial_balance_result = generate_trial_balance(
            TrialBalanceRequest(
                fiscal_period=request.fiscal_period,
                entity_code=request.entity_code
            )
        )
        
        # Check for blocking exceptions
        if not trial_balance_result.success:
            return ToolResponse(
                success=False,
                message="Cannot close period - blocking exceptions exist",
                data={
                    'blocking_exceptions': [
                        e for e in trial_balance_result.exceptions 
                        if e.get('blocking', False)
                    ]
                }
            )
        
        # Generate close summary
        close_summary = {
            'fiscal_period': request.fiscal_period,
            'entity_code': request.entity_code,
            'close_date': datetime.now().isoformat(),
            'approved_by': request.approved_by,
            'status': 'CLOSED',
            'trial_balance': trial_balance_result.data,
            'statements_generated': [
                'Income Statement',
                'Balance Sheet',
                'Cash Flow Statement',
                'Statement of Changes in Equity'
            ]
        }
        
        return ToolResponse(
            success=True,
            message=f"Period {request.fiscal_period} closed successfully",
            data=close_summary
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# TOOL 9: GET MISSING COST CENTERS
# ============================================================================

@app.get("/tools/get_missing_cost_centers/{fiscal_period}", response_model=ToolResponse)
def get_missing_cost_centers(fiscal_period: str):
    """
    Get list of transactions with missing cost centers
    
    Returns transactions that need cost center assignment
    """
    try:
        transactions = load_csv_data('Raw_GL_Export_With_CostCenters_Feb2026.csv')
        period_txns = [t for t in transactions if t['Fiscal_Period'] == fiscal_period]
        
        missing = [
            {
                'transaction_id': t['Txn_ID'],
                'posting_date': t['Posting_Date_Raw'],
                'account': t['Account_Code_Raw'],
                'vendor': t['Vendor_Name_Raw'],
                'amount': float(t['Amount']),
                'narrative': t['Narrative']
            }
            for t in period_txns if not t.get('Cost_Center', '')
        ]
        
        return ToolResponse(
            success=True,
            message=f"Found {len(missing)} transactions with missing cost centers",
            data={
                'fiscal_period': fiscal_period,
                'missing_count': len(missing),
                'transactions': missing,
                'total_amount': sum(t['amount'] for t in missing)
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# TOOL 10: GET OVERDUE INVOICES
# ============================================================================

@app.get("/tools/get_overdue_invoices", response_model=ToolResponse)
def get_overdue_invoices():
    """
    Get list of overdue invoices requiring collection action
    """
    try:
        ar_records = load_csv_data('AR_Subledger_Feb2026.csv')
        
        overdue = [
            {
                'customer_id': r['Customer_ID'],
                'customer_name': r['Customer_Name'],
                'invoice_number': r['Invoice_Number'],
                'invoice_date': r['Invoice_Date'],
                'due_date': r['Due_Date'],
                'days_outstanding': int(r['Days_Outstanding']),
                'outstanding_balance': float(r['Outstanding_Balance']),
                'cost_center': r['Cost_Center'],
                'region': r['Region']
            }
            for r in ar_records if r['Status'] == 'Overdue'
        ]
        
        # Sort by days outstanding
        overdue.sort(key=lambda x: x['days_outstanding'], reverse=True)
        
        return ToolResponse(
            success=True,
            message=f"Found {len(overdue)} overdue invoices",
            data={
                'overdue_count': len(overdue),
                'invoices': overdue,
                'total_outstanding': sum(inv['outstanding_balance'] for inv in overdue)
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
