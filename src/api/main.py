import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np

from src.database import load_skus, init_db
from src.optimization.formulation import (
    solve_inventory_mip, 
    solve_inventory_lp_relaxation, 
    solve_inventory_scipy_lp,
    evaluate_reorder_policy
)
from src.optimization.sensitivity import run_sensitivity_analysis
from src.anomaly_detection.detector import detect_cost_anomalies
from src.tracking.tracker import OptimizationTracker

app = FastAPI(
    title="Inventory Cost Optimization & Negotiation flagging Service",
    description="Production-style FastAPI backend for retail supply chain optimization",
    version="1.0.0"
)

# Startup event to ensure database is initialized and seeded
@app.on_event("startup")
def startup_event():
    init_db()

# Pydantic schemas for request validation
class OptimizeRequest(BaseModel):
    budget_cap: float = Field(default=80000.0, gt=0, description="Total procurement budget limit ($)")
    capacity_cap: float = Field(default=150.0, gt=0, description="Total warehouse capacity limit (m³)")
    service_level_target: float = Field(default=0.95, ge=0.5, le=0.999, description="Target fill rate floor (0.5 to 0.999)")
    num_scenarios: int = Field(default=15, ge=5, le=100, description="Number of demand scenarios for SAA")
    use_scipy_lp: bool = Field(default=False, description="If true, uses SciPy LP relaxation instead of PuLP MIP")

class SKURecommendationResponse(BaseModel):
    sku_id: str
    category: str
    fabric: str
    style_group: str
    unit_cost: float
    current_inventory: float
    recommended_order_qty: float
    reorder_flag: int
    expected_shortfall: float
    expected_holding_cost: float
    expected_stockout_cost: float

class RunSummary(BaseModel):
    total_holding_cost: float
    total_stockout_cost: float
    total_cost: float
    baseline_holding_cost: float
    baseline_stockout_cost: float
    baseline_total_cost: float
    savings_dollar: float
    savings_percent: float
    service_level_achieved: float

class OptimizeResponse(BaseModel):
    run_id: str
    status: str
    objective_value: Optional[float] = None
    diagnostics: Dict[str, Any]
    summary: Optional[RunSummary] = None
    recommendations: Optional[List[SKURecommendationResponse]] = None

class AnomalyResponse(BaseModel):
    sku_id: str
    category: str
    fabric: str
    style_group: str
    supplier_id: str
    unit_cost: float
    median: float
    pct_deviation: float
    order_volume: float
    potential_savings: float
    explanation: str

class ScenarioRequest(BaseModel):
    base_budget: float = Field(default=80000.0, gt=0)
    base_capacity: float = Field(default=150.0, gt=0)
    service_level: float = Field(default=0.95, ge=0.5, le=0.999)
    num_scenarios: int = Field(default=15, ge=5, le=100)

class ScenarioResponse(BaseModel):
    status: str
    base_budget: float
    base_capacity: float
    base_objective: float
    budget_plus_10: Optional[float]
    budget_minus_10: Optional[float]
    capacity_plus_10: Optional[float]
    capacity_minus_10: Optional[float]
    empirical_budget_shadow_price: float
    empirical_capacity_shadow_price: float
    analytical_budget_shadow_price: float
    analytical_capacity_shadow_price: float
    budget_explanation: str
    capacity_explanation: str

@app.get("/health")
def health_check():
    """
    Service health check endpoint.
    """
    try:
        # Quick query check
        skus_df = load_skus()
        return {
            "status": "healthy",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "database": "connected",
            "skus_in_db": len(skus_df)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database connection error: {str(e)}")

@app.post("/optimize", response_model=OptimizeResponse)
def run_optimization(req: OptimizeRequest):
    """
    Solves the inventory cost optimization model under given constraints.
    Computes comparative metrics against a heuristic baseline (order-up-to mean demand).
    """
    # 1. Load data
    skus_df = load_skus()
    if skus_df.empty:
        raise HTTPException(status_code=500, detail="SKU database is empty.")
        
    # 2. Run solver
    if req.use_scipy_lp:
        status, obj, recs_df, diag = solve_inventory_scipy_lp(
            skus_df, req.budget_cap, req.capacity_cap, req.service_level_target, req.num_scenarios
        )
        solver_type = "SciPy_LP_Relaxation"
    else:
        status, obj, recs_df, diag = solve_inventory_mip(
            skus_df, req.budget_cap, req.capacity_cap, req.service_level_target, req.num_scenarios
        )
        solver_type = "PuLP_MIP"
        
    if status != "Optimal" or obj is None:
        run_id = OptimizationTracker.log_run(
            solver_type=solver_type,
            solver_status=status,
            budget_cap=req.budget_cap,
            capacity_cap=req.capacity_cap,
            service_level_target=req.service_level_target,
            total_holding_cost=0.0,
            total_stockout_penalty=0.0,
            total_cost=0.0,
            savings_vs_base=0.0,
            run_time_seconds=diag["runtime_seconds"],
            recommendations_df=pd.DataFrame(columns=["sku_id", "recommended_order_qty", "reorder_flag", "expected_shortfall", "expected_holding_cost", "expected_stockout_cost"])
        )
        return OptimizeResponse(
            run_id=run_id,
            status=status,
            objective_value=None,
            diagnostics=diag,
            summary=None,
            recommendations=None
        )
        
    # 3. Calculate heuristic baseline costs
    # Baseline policy: order what is needed to cover mean weekly demand: x_i = max(0, mean_demand - current_inv)
    baseline_qtys = {}
    for idx, row in skus_df.iterrows():
        sku_id = row["sku_id"]
        baseline_qtys[sku_id] = max(0.0, row["mean_weekly_demand"] - row["current_inventory"])
        
    base_h, base_s, base_total = evaluate_reorder_policy(
        skus_df, baseline_qtys, req.num_scenarios
    )
    
    # 4. Compile detailed results
    # Join recommendations with SKU master details
    res_merged = recs_df.merge(skus_df, on="sku_id")
    
    opt_h = res_merged["expected_holding_cost"].sum()
    opt_s = res_merged["expected_stockout_cost"].sum()
    opt_total = opt_h + opt_s
    
    savings_dollar = base_total - opt_total
    savings_pct = (savings_dollar / base_total) * 100.0 if base_total > 0 else 0.0
    
    # Calculate service level achieved (actual fill rate achieved)
    # Fill Rate = 1 - (expected shortfall / expected demand)
    total_shortfall = res_merged["expected_shortfall"].sum()
    total_demand = res_merged["mean_weekly_demand"].sum()
    fill_rate_achieved = 1.0 - (total_shortfall / total_demand) if total_demand > 0 else 1.0
    
    summary = RunSummary(
        total_holding_cost=round(opt_h, 2),
        total_stockout_cost=round(opt_s, 2),
        total_cost=round(opt_total, 2),
        baseline_holding_cost=round(base_h, 2),
        baseline_stockout_cost=round(base_s, 2),
        baseline_total_cost=round(base_total, 2),
        savings_dollar=round(savings_dollar, 2),
        savings_percent=round(savings_pct, 2),
        service_level_achieved=round(fill_rate_achieved, 4)
    )
    
    # Log run in sqlite tracker
    run_id = OptimizationTracker.log_run(
        solver_type=solver_type,
        solver_status=status,
        budget_cap=req.budget_cap,
        capacity_cap=req.capacity_cap,
        service_level_target=req.service_level_target,
        total_holding_cost=opt_h,
        total_stockout_penalty=opt_s,
        total_cost=opt_total,
        savings_vs_base=savings_dollar,
        run_time_seconds=diag["runtime_seconds"],
        recommendations_df=recs_df
    )
    
    # Build recommendation responses
    recs_list = []
    for _, row in res_merged.iterrows():
        recs_list.append(SKURecommendationResponse(
            sku_id=row["sku_id"],
            category=row["category"],
            fabric=row["fabric"],
            style_group=row["style_group"],
            unit_cost=row["unit_cost"],
            current_inventory=row["current_inventory"],
            recommended_order_qty=row["recommended_order_qty"],
            reorder_flag=int(row["reorder_flag"]),
            expected_shortfall=row["expected_shortfall"],
            expected_holding_cost=row["expected_holding_cost"],
            expected_stockout_cost=row["expected_stockout_cost"]
        ))
        
    return OptimizeResponse(
        run_id=run_id,
        status=status,
        objective_value=obj,
        diagnostics=diag,
        summary=summary,
        recommendations=recs_list
    )

@app.get("/anomalies", response_model=List[AnomalyResponse])
def get_anomalies():
    """
    Exposes flagged pricing outliers, linking them to the order recommendations of the most recent optimization run.
    """
    skus_df = load_skus()
    if skus_df.empty:
        raise HTTPException(status_code=500, detail="SKU database is empty.")
        
    # Get latest run recommendations if they exist
    history = OptimizationTracker.get_history()
    recs_df = None
    if not history.empty:
        latest_run_id = history.iloc[0]["run_id"]
        recs_df = OptimizationTracker.get_run_details(latest_run_id)
        
    anomalies_df = detect_cost_anomalies(skus_df, recs_df)
    
    result = []
    for _, row in anomalies_df.iterrows():
        result.append(AnomalyResponse(
            sku_id=row["sku_id"],
            category=row["category"],
            fabric=row["fabric"],
            style_group=row["style_group"],
            supplier_id=row["supplier_id"],
            unit_cost=row["unit_cost"],
            median=row["median"],
            pct_deviation=row["pct_deviation"],
            order_volume=row["order_volume"],
            potential_savings=row["potential_savings"],
            explanation=row["explanation"]
        ))
        
    return result

@app.post("/scenario", response_model=ScenarioResponse)
def run_scenario(req: ScenarioRequest):
    """
    Performs sensitivity analysis around the base constraints.
    Returns comparative objectives and shadow prices.
    """
    skus_df = load_skus()
    if skus_df.empty:
        raise HTTPException(status_code=500, detail="SKU database is empty.")
        
    result = run_sensitivity_analysis(
        skus_df=skus_df,
        base_budget=req.base_budget,
        base_capacity=req.base_capacity,
        service_level=req.service_level,
        num_scenarios=req.num_scenarios
    )
    
    if result["status"] == "Infeasible":
        raise HTTPException(status_code=400, detail=result["message"])
        
    return ScenarioResponse(
        status=result["status"],
        base_budget=result["base_budget"],
        base_capacity=result["base_capacity"],
        base_objective=result["base_objective"],
        budget_plus_10=result["budget_plus_10"],
        budget_minus_10=result["budget_minus_10"],
        capacity_plus_10=result["capacity_plus_10"],
        capacity_minus_10=result["capacity_minus_10"],
        empirical_budget_shadow_price=result["empirical_budget_shadow_price"],
        empirical_capacity_shadow_price=result["empirical_capacity_shadow_price"],
        analytical_budget_shadow_price=result["analytical_budget_shadow_price"],
        analytical_capacity_shadow_price=result["analytical_capacity_shadow_price"],
        budget_explanation=result["budget_explanation"],
        capacity_explanation=result["capacity_explanation"]
    )
