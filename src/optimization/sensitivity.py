from typing import Dict, Any, Tuple
import pandas as pd
from src.optimization.formulation import solve_inventory_mip, solve_inventory_lp_relaxation

def run_sensitivity_analysis(
    skus_df: pd.DataFrame,
    base_budget: float,
    base_capacity: float,
    service_level: float,
    num_scenarios: int = 15
) -> Dict[str, Any]:
    """
    Executes a sensitivity analysis by flexing budget and capacity by +/-10%.
    Computes both empirical shadow prices (from MIP resolves) and analytical shadow prices (from LP relaxation).
    """
    # 1. Base Run
    base_status, base_obj, _, _ = solve_inventory_mip(
        skus_df, base_budget, base_capacity, service_level, num_scenarios
    )
    
    if base_status != "Optimal" or base_obj is None:
        return {
            "status": "Infeasible",
            "message": "Base configuration is infeasible; sensitivity cannot be computed."
        }
        
    # 2. Flex Budget: +10% and -10%
    budget_plus = base_budget * 1.10
    budget_minus = base_budget * 0.90
    
    _, obj_b_plus, _, _ = solve_inventory_mip(
        skus_df, budget_plus, base_capacity, service_level, num_scenarios
    )
    _, obj_b_minus, _, _ = solve_inventory_mip(
        skus_df, budget_minus, base_capacity, service_level, num_scenarios
    )
    
    # Compute empirical budget shadow price
    # Note: Objective is minimized cost, so a higher budget usually decreases cost.
    # Shadow price = dObjective / dBudget. A negative value means increasing budget decreases cost (savings!).
    if obj_b_plus is not None and obj_b_minus is not None:
        emp_budget_sp = (obj_b_plus - obj_b_minus) / (budget_plus - budget_minus)
    elif obj_b_plus is not None:
        emp_budget_sp = (obj_b_plus - base_obj) / (budget_plus - base_budget)
    elif obj_b_minus is not None:
        emp_budget_sp = (base_obj - obj_b_minus) / (base_budget - budget_minus)
    else:
        emp_budget_sp = 0.0
        
    # 3. Flex Capacity: +10% and -10%
    cap_plus = base_capacity * 1.10
    cap_minus = base_capacity * 0.90
    
    _, obj_c_plus, _, _ = solve_inventory_mip(
        skus_df, base_budget, cap_plus, service_level, num_scenarios
    )
    _, obj_c_minus, _, _ = solve_inventory_mip(
        skus_df, base_budget, cap_minus, service_level, num_scenarios
    )
    
    # Compute empirical capacity shadow price
    if obj_c_plus is not None and obj_c_minus is not None:
        emp_cap_sp = (obj_c_plus - obj_c_minus) / (cap_plus - cap_minus)
    elif obj_c_plus is not None:
        emp_cap_sp = (obj_c_plus - base_obj) / (cap_plus - base_capacity)
    elif obj_c_minus is not None:
        emp_cap_sp = (base_obj - obj_c_minus) / (base_capacity - cap_minus)
    else:
        emp_cap_sp = 0.0
        
    # 4. Analytical Shadow Prices from LP Relaxation
    _, _, _, _, duals = solve_inventory_lp_relaxation(
        skus_df, base_budget, base_capacity, service_level, num_scenarios
    )
    
    lp_budget_sp = duals.get("budget_shadow_price", 0.0)
    lp_cap_sp = duals.get("capacity_shadow_price", 0.0)
    
    # Generate business interpretation
    # A negative shadow price means cost reduction
    budget_explanation = "Budget constraint is not binding; increasing budget provides no additional savings."
    if abs(emp_budget_sp) > 1e-4:
        savings_pct = abs(emp_budget_sp) * 100
        budget_explanation = f"Every additional $1.00 of budget saves approximately ${abs(emp_budget_sp):.2f} in supply chain costs ({savings_pct:.1f}% marginal return) by shifting order mix toward high-margin/high-penalty SKUs."
        
    cap_explanation = "Warehouse storage capacity is not binding; expanding warehouse volume will not reduce costs."
    if abs(emp_cap_sp) > 1e-4:
        cap_explanation = f"Every additional 1 m³ of warehouse capacity reduces weekly costs by ${abs(emp_cap_sp):.2f} by permitting larger batches and avoiding stockout backorders on bulky products."
        
    return {
        "status": "Success",
        "base_budget": base_budget,
        "base_capacity": base_capacity,
        "base_objective": round(base_obj, 2),
        "budget_plus_10": round(obj_b_plus, 2) if obj_b_plus else None,
        "budget_minus_10": round(obj_b_minus, 2) if obj_b_minus else None,
        "capacity_plus_10": round(obj_c_plus, 2) if obj_c_plus else None,
        "capacity_minus_10": round(obj_c_minus, 2) if obj_c_minus else None,
        "empirical_budget_shadow_price": round(emp_budget_sp, 4),
        "empirical_capacity_shadow_price": round(emp_cap_sp, 4),
        "analytical_budget_shadow_price": round(lp_budget_sp, 4),
        "analytical_capacity_shadow_price": round(lp_cap_sp, 4),
        "budget_explanation": budget_explanation,
        "capacity_explanation": cap_explanation
    }
