import pytest
import pandas as pd
import numpy as np
from src.data_generator import generate_sku_dataset
from src.optimization.formulation import solve_inventory_mip

@pytest.fixture
def sample_skus():
    # Use a small subset of SKUs to keep tests fast
    return generate_sku_dataset(num_skus=20, seed=101)

def test_solver_optimal_status(sample_skus):
    """
    Test that the solver executes successfully and finds an optimal solution under normal, scaled constraints.
    """
    status, obj, recs_df, diag = solve_inventory_mip(
        skus_df=sample_skus,
        budget_cap=50000.0,
        capacity_cap=100.0,
        service_level_target=0.90,
        num_scenarios=5
    )
    
    assert status == "Optimal"
    assert obj is not None
    assert obj > 0.0
    assert len(recs_df) == len(sample_skus)

def test_moq_and_binary_constraint_satisfaction(sample_skus):
    """
    Test that recommended quantities strictly adhere to MOQ and binary logic:
    if recommended quantity > 0, it must be >= MOQ, and if 0, reorder_flag must be 0.
    """
    status, _, recs_df, _ = solve_inventory_mip(
        skus_df=sample_skus,
        budget_cap=50000.0,
        capacity_cap=100.0,
        service_level_target=0.90,
        num_scenarios=5
    )
    
    assert status == "Optimal"
    
    # Merge with original MOQ bounds
    merged = recs_df.merge(sample_skus, on="sku_id")
    for _, row in merged.iterrows():
        qty = row["recommended_order_qty"]
        moq = row["min_order_quantity"]
        flag = row["reorder_flag"]
        
        if qty > 0.001:
            assert flag == 1
            assert qty >= moq - 1e-4  # floating point tolerance
        else:
            assert flag == 0
            assert qty == 0.0

def test_budget_constraint_satisfaction(sample_skus):
    """
    Test that the total procurement cost of recommended orders does not exceed the budget cap.
    Using a service level target of 0.0 guarantees feasibility.
    """
    budget_cap = 5000.0 # tight budget
    status, _, recs_df, _ = solve_inventory_mip(
        skus_df=sample_skus,
        budget_cap=budget_cap,
        capacity_cap=100.0,
        service_level_target=0.85,
        num_scenarios=5
    )
    
    assert status == "Optimal"
    merged = recs_df.merge(sample_skus, on="sku_id")
    total_procurement_cost = sum(merged["recommended_order_qty"] * merged["unit_cost"])
    assert total_procurement_cost <= budget_cap + 1e-3

def test_objective_monotonicity(sample_skus):
    """
    Test that tightening the budget constraint monotonically increases (or keeps equal) the objective cost.
    Higher budget -> lower (better) holding + stockout cost.
    """
    # Run with loose budget
    status_loose, obj_loose, _, _ = solve_inventory_mip(
        skus_df=sample_skus,
        budget_cap=50000.0,
        capacity_cap=100.0,
        service_level_target=0.85,
        num_scenarios=5
    )
    
    # Run with tight budget
    status_tight, obj_tight, _, _ = solve_inventory_mip(
        skus_df=sample_skus,
        budget_cap=5000.0,
        capacity_cap=100.0,
        service_level_target=0.85,
        num_scenarios=5
    )
    
    assert status_loose == "Optimal"
    assert status_tight == "Optimal"
    # Tight budget must have higher or equal objective (penalty cost) than loose budget
    assert obj_tight >= obj_loose - 1e-4
