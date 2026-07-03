import pytest
import os
import pandas as pd
from unittest.mock import patch, MagicMock
from streamlit.testing.v1 import AppTest

@pytest.fixture
def mock_dashboard_environment():
    # Create mock SKUs DataFrame
    mock_skus = pd.DataFrame([
        {
            "sku_id": f"SKU-{i}",
            "category": "Structured Tops",
            "fabric": "Cotton",
            "style_group": "Cotton Structured Tops",
            "unit_cost": 25.0,
            "unit_volume": 0.02,
            "min_order_quantity": 50,
            "current_inventory": 100,
            "mean_weekly_demand": 150,
            "demand_variance": 400,
            "weekly_holding_cost": 0.5,
            "stockout_penalty": 5.0,
            "supplier_id": "SUPPLIER_A",
            "service_level_target": 0.95
        } for i in range(5)
    ])
    
    # Create mock recommendation DataFrame
    mock_recs = pd.DataFrame([
        {
            "sku_id": f"SKU-{i}",
            "recommended_order_qty": 50.0,
            "reorder_flag": 1,
            "expected_shortfall": 5.0,
            "expected_holding_cost": 25.0,
            "expected_stockout_cost": 25.0
        } for i in range(5)
    ])
    
    # Create mock runs history DataFrame
    mock_history = pd.DataFrame([
        {
            "run_id": "RUN-20260703-120000-abcdef",
            "timestamp": "2026-07-03 12:00:00",
            "solver_type": "PuLP_MIP",
            "solver_status": "Optimal",
            "budget_cap": 10000000.0,
            "capacity_cap": 5000.0,
            "service_level_target": 0.95,
            "total_cost": 50.0,
            "savings_vs_base": 100.0,
            "run_time_seconds": 0.5
        }
    ])
    
    # Setup mocks
    mocks = {
        "load_skus": patch("src.database.load_skus", return_value=mock_skus),
        "solve_mip": patch("src.optimization.formulation.solve_inventory_mip", return_value=("Optimal", 50.0, mock_recs, {})),
        "solve_scipy": patch("src.optimization.formulation.solve_inventory_scipy_lp", return_value=("Optimal", 50.0, mock_recs, {})),
        "evaluate_reorder": patch("src.optimization.formulation.evaluate_reorder_policy", return_value=(30.0, 120.0, 150.0)),
        "get_history": patch("src.tracking.tracker.OptimizationTracker.get_history", return_value=mock_history),
        "get_run_details": patch("src.tracking.tracker.OptimizationTracker.get_run_details", return_value=mock_recs),
        "log_run": patch("src.tracking.tracker.OptimizationTracker.log_run", return_value="RUN-MOCK"),
        "sensitivity": patch("src.optimization.sensitivity.run_sensitivity_analysis", return_value={
            "status": "Success",
            "base_budget": 10000000.0,
            "base_capacity": 5000.0,
            "base_objective": 50.0,
            "budget_plus_10": 48.0,
            "budget_minus_10": 52.0,
            "capacity_plus_10": 50.0,
            "capacity_minus_10": 50.0,
            "empirical_budget_shadow_price": 0.2,
            "empirical_capacity_shadow_price": 0.0,
            "analytical_budget_shadow_price": 0.15,
            "analytical_capacity_shadow_price": 0.0,
            "budget_explanation": "Marginal value is low.",
            "capacity_explanation": "Space is not binding."
        }),
        "anomalies": patch("src.anomaly_detection.detector.detect_cost_anomalies", return_value=pd.DataFrame([
            {
                "sku_id": "SKU-0",
                "category": "Structured Tops",
                "fabric": "Cotton",
                "style_group": "Cotton Structured Tops",
                "supplier_id": "SUPPLIER_A",
                "unit_cost": 85.0,
                "median": 25.0,
                "pct_deviation": 240.0,
                "order_volume": 50.0,
                "potential_savings": 3000.0,
                "explanation": "Cost outlier in Cotton Structured Tops group."
            }
        ]))
    }
    
    # Start all patches
    active_mocks = {name: p.start() for name, p in mocks.items()}
    yield active_mocks
    # Stop all patches
    for p in mocks.values():
        p.stop()

def test_dashboard_tabs_smoke(mock_dashboard_environment):
    """
    Smoke test that programmatically runs the Streamlit dashboard app and
    asserts that it compiles and runs without raising any unhandled exceptions.
    """
    app_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../dashboard/app.py"))
    assert os.path.exists(app_path), f"Dashboard script not found at: {app_path}"
    
    # Use Streamlit's AppTest to run the file
    at = AppTest.from_file(app_path)
    
    # Set run mode environment variable or mock health status if running in API mode
    # Here we mock requests.get to return healthy check status
    with patch("requests.get") as mock_get, patch("requests.post") as mock_post:
        # Mock health endpoint
        mock_health_res = MagicMock()
        mock_health_res.status_code = 200
        mock_health_res.json.return_value = {"status": "healthy", "skus_in_db": 5}
        
        # Mock history endpoint
        mock_history_res = MagicMock()
        mock_history_res.status_code = 200
        mock_history_res.json.return_value = [
            {
                "run_id": "RUN-20260703-120000-abcdef",
                "timestamp": "2026-07-03 12:00:00",
                "solver_type": "PuLP_MIP",
                "solver_status": "Optimal",
                "budget_cap": 10000000.0,
                "capacity_cap": 5000.0,
                "service_level_target": 0.95,
                "total_cost": 50.0,
                "savings_vs_base": 100.0,
                "run_time_seconds": 0.5
            }
        ]
        
        # Mock anomalies endpoint
        mock_anomalies_res = MagicMock()
        mock_anomalies_res.status_code = 200
        mock_anomalies_res.json.return_value = [
            {
                "sku_id": "SKU-0",
                "category": "Structured Tops",
                "fabric": "Cotton",
                "style_group": "Cotton Structured Tops",
                "supplier_id": "SUPPLIER_A",
                "unit_cost": 85.0,
                "median": 25.0,
                "pct_deviation": 240.0,
                "order_volume": 50.0,
                "potential_savings": 3000.0,
                "explanation": "Cost outlier in Cotton Structured Tops group."
            }
        ]
        
        # Determine requests route returns
        def get_route(url, *args, **kwargs):
            if "health" in url:
                return mock_health_res
            elif "history" in url:
                return mock_history_res
            elif "anomalies" in url:
                return mock_anomalies_res
            return mock_health_res
            
        mock_get.side_effect = get_route
        
        # Mock optimize post endpoint
        mock_opt_res = MagicMock()
        mock_opt_res.status_code = 200
        mock_opt_res.json.return_value = {
            "run_id": "RUN-MOCK",
            "status": "Optimal",
            "objective_value": 50.0,
            "diagnostics": {"runtime_seconds": 0.5},
            "summary": {
                "total_holding_cost": 25.0,
                "total_stockout_cost": 25.0,
                "total_cost": 50.0,
                "baseline_holding_cost": 30.0,
                "baseline_stockout_cost": 120.0,
                "baseline_total_cost": 150.0,
                "savings_dollar": 100.0,
                "savings_percent": 66.6,
                "service_level_achieved": 0.95
            },
            "recommendations": [
                {
                    "sku_id": f"SKU-{i}",
                    "category": "Structured Tops",
                    "fabric": "Cotton",
                    "style_group": "Cotton Structured Tops",
                    "unit_cost": 25.0,
                    "current_inventory": 100,
                    "recommended_order_qty": 50.0,
                    "reorder_flag": 1,
                    "expected_shortfall": 5.0,
                    "expected_holding_cost": 25.0,
                    "expected_stockout_cost": 25.0
                } for i in range(5)
            ]
        }
        
        # Mock sensitivity post endpoint
        mock_sens_res = MagicMock()
        mock_sens_res.status_code = 200
        mock_sens_res.json.return_value = {
            "status": "Success",
            "base_budget": 10000000.0,
            "base_capacity": 5000.0,
            "base_objective": 50.0,
            "budget_plus_10": 48.0,
            "budget_minus_10": 52.0,
            "capacity_plus_10": 50.0,
            "capacity_minus_10": 50.0,
            "empirical_budget_shadow_price": 0.2,
            "empirical_capacity_shadow_price": 0.0,
            "analytical_budget_shadow_price": 0.15,
            "analytical_capacity_shadow_price": 0.0,
            "budget_explanation": "Marginal value is low.",
            "capacity_explanation": "Space is not binding."
        }
        
        def post_route(url, *args, **kwargs):
            if "optimize" in url:
                return mock_opt_res
            elif "scenario" in url:
                return mock_sens_res
            return mock_opt_res
            
        mock_post.side_effect = post_route
        
        # Run the Streamlit app script execution loop
        at.run(timeout=10)
        
        # Check if any exception was raised during rendering
        assert not at.exception, f"Streamlit application raised an unhandled exception: {at.exception}"
