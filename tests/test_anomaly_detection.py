import pytest
import pandas as pd
import numpy as np
from src.anomaly_detection.detector import detect_cost_anomalies

@pytest.fixture
def clean_and_anomalous_skus():
    """
    Creates a small controlled dataset of SKUs within one style group
    with one explicitly injected cost anomaly.
    """
    style_group = "Leather Activewear"
    skus = [
        # Normal peer group SKUs (median cost should be $50.0)
        {"sku_id": "SKU-1", "style_group": style_group, "category": "Activewear", "fabric": "Leather", "supplier_id": "S1", "unit_cost": 49.0, "mean_weekly_demand": 100},
        {"sku_id": "SKU-2", "style_group": style_group, "category": "Activewear", "fabric": "Leather", "supplier_id": "S1", "unit_cost": 50.0, "mean_weekly_demand": 100},
        {"sku_id": "SKU-3", "style_group": style_group, "category": "Activewear", "fabric": "Leather", "supplier_id": "S1", "unit_cost": 51.0, "mean_weekly_demand": 100},
        {"sku_id": "SKU-4", "style_group": style_group, "category": "Activewear", "fabric": "Leather", "supplier_id": "S1", "unit_cost": 50.0, "mean_weekly_demand": 100},
        
        # Outlier cost anomaly (deviates significantly from median)
        {"sku_id": "SKU-ANOMALY", "style_group": style_group, "category": "Activewear", "fabric": "Leather", "supplier_id": "S2", "unit_cost": 95.0, "mean_weekly_demand": 200}
    ]
    return pd.DataFrame(skus)

def test_anomaly_detection_flagging(clean_and_anomalous_skus):
    """
    Test that the anomalous SKU is correctly flagged and normal SKUs are not.
    """
    anomalies_df = detect_cost_anomalies(clean_and_anomalous_skus)
    
    # Check that only SKU-ANOMALY is flagged
    assert len(anomalies_df) == 1
    assert anomalies_df.loc[0, "sku_id"] == "SKU-ANOMALY"
    
    # Median of [49, 50, 51, 50, 95] is 50.0
    assert anomalies_df.loc[0, "median"] == 50.0
    assert anomalies_df.loc[0, "pct_deviation"] == 90.0 # (95 - 50) / 50 * 100

def test_potential_savings_calculation(clean_and_anomalous_skus):
    """
    Test that potential savings are calculated correctly as (unit_cost - median) * volume.
    """
    # Test fallback mode (uses mean weekly demand as volume)
    anomalies_df = detect_cost_anomalies(clean_and_anomalous_skus)
    
    # Expected savings = (95.0 - 50.0) * 200 = 45.0 * 200 = 9000.0
    assert anomalies_df.loc[0, "potential_savings"] == 9000.0
    
    # Test recommendations override mode (uses recommended order qty)
    recs = pd.DataFrame([
        {"sku_id": "SKU-ANOMALY", "recommended_order_qty": 150.0}
    ])
    
    anomalies_df_rec = detect_cost_anomalies(clean_and_anomalous_skus, recommendations_df=recs)
    # Expected savings = (95.0 - 50.0) * 150 = 45.0 * 150 = 6750.0
    assert anomalies_df_rec.loc[0, "potential_savings"] == 6750.0

def test_explanation_formatting(clean_and_anomalous_skus):
    """
    Test that explanations contain unit cost, median cost, style group, and potential savings.
    """
    anomalies_df = detect_cost_anomalies(clean_and_anomalous_skus)
    explanation = anomalies_df.loc[0, "explanation"]
    
    assert "95.00" in explanation
    assert "50.00" in explanation
    assert "Leather Activewear" in explanation
    assert "9,000.00" in explanation
