import pytest
from src.database import load_skus
from src.optimization.formulation import solve_inventory_mip

def test_default_configuration_feasibility():
    """
    Sanity check: Verifies that the default dashboard settings (10M budget,
    5,000 m3 warehouse capacity, 95% aggregate service level) are fully
    feasible and yield an Optimal solution on the full 1,000 SKU dataset.
    """
    skus_df = load_skus()
    assert len(skus_df) == 1000, "Database should contain 1,000 SKUs"
    
    # Run solver under dashboard default configuration
    status, obj, recs_df, diag = solve_inventory_mip(
        skus_df=skus_df,
        budget_cap=10000000.0,      # $10,000,000
        capacity_cap=5000.0,        # 5,000 m³
        service_level_target=0.95,   # 95% aggregate fill rate
        num_scenarios=15
    )
    
    assert status == "Optimal", f"Default settings should be optimal, but solver returned: {status}. Reason: {diag.get('error_message')}"
    assert obj is not None, "Objective value should be numeric"
    assert len(recs_df) == 1000, "Should generate recommendations for all 1,000 SKUs"
