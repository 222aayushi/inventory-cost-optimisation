import numpy as np
import pandas as pd

def generate_sku_dataset(num_skus: int = 1000, seed: int = 42) -> pd.DataFrame:
    """
    Generates a realistic synthetic SKU dataset for a retail supply chain
    with embedded cost anomalies for evaluation.
    """
    np.random.seed(seed)
    
    categories = ["Structured Tops", "Basics", "Loungewear", "Activewear"]
    fabrics = ["Cotton", "Polyester", "Wool", "Denim", "Leather"]
    
    # Base costs and volume parameters per category
    cat_params = {
        "Structured Tops": {"base_cost": 20.0, "unit_volume": 0.01, "min_demand": 300, "max_demand": 1000},
        "Basics": {"base_cost": 35.0, "unit_volume": 0.02, "min_demand": 200, "max_demand": 800},
        "Loungewear": {"base_cost": 75.0, "unit_volume": 0.05, "min_demand": 100, "max_demand": 400},
        "Activewear": {"base_cost": 60.0, "unit_volume": 0.03, "min_demand": 150, "max_demand": 600}
    }
    
    # Cost premiums per fabric type
    fabric_premiums = {
        "Leather": 40.0,
        "Wool": 20.0,
        "Denim": 15.0,
        "Polyester": 5.0,
        "Cotton": 0.0
    }
    
    skus = []
    
    for i in range(num_skus):
        sku_id = f"SKU-{100000 + i}"
        category = np.random.choice(categories)
        fabric = np.random.choice(fabrics)
        style_group = f"{fabric} {category}"
        
        # Calculate unit cost
        base = cat_params[category]["base_cost"]
        premium = fabric_premiums[fabric]
        # Add random variation
        variation = np.random.normal(0, base * 0.05)
        unit_cost = max(base + premium + variation, 5.0)
        
        # Determine if this SKU is a cost anomaly (~5% probability)
        is_anomaly = np.random.rand() < 0.05
        anomaly_factor = 1.0
        if is_anomaly:
            # Inject anomaly: cost is 1.5x to 2.2x higher than expected
            anomaly_factor = np.random.uniform(1.5, 2.2)
            unit_cost *= anomaly_factor
            
        # Volume/Warehouse space
        unit_volume = cat_params[category]["unit_volume"]
        
        # Supplier assignment (10 suppliers)
        supplier_id = f"SUPPLIER-{np.random.randint(1, 11)}"
        
        # Demand Parameters (mean weekly demand)
        d_min = cat_params[category]["min_demand"]
        d_max = cat_params[category]["max_demand"]
        mean_demand = float(np.random.randint(d_min, d_max + 1))
        # Demand variance is high enough to show stockout risk
        demand_cv = np.random.uniform(0.15, 0.35) # Coefficient of variation
        demand_std = mean_demand * demand_cv
        demand_variance = demand_std ** 2
        
        # Inventory parameters
        # Current inventory ranges from stockout state to healthy stock
        current_inventory = float(np.random.randint(0, int(mean_demand * 1.5)))
        
        # Reorder parameters
        min_order_quantity = float(np.random.choice([50, 100, 150, 200]))
        lead_time = int(np.random.choice([1, 2, 3, 4]))
        
        # Financial Rates
        holding_cost_rate = np.random.uniform(0.18, 0.28) # 18% to 28% annual holding cost rate
        # Weekly holding cost per unit
        holding_cost = (holding_cost_rate * unit_cost) / 52.0
        
        # Stockout penalty is a multiplier of unit cost (1.8x to 3.0x)
        stockout_penalty = unit_cost * np.random.uniform(1.8, 3.0)
        
        # Default service level target (95%)
        service_level_target = 0.95
        
        skus.append({
            "sku_id": sku_id,
            "category": category,
            "fabric": fabric,
            "style_group": style_group,
            "unit_cost": round(unit_cost, 2),
            "is_anomaly_injected": int(is_anomaly),
            "anomaly_factor": round(anomaly_factor, 2),
            "unit_volume": round(unit_volume, 3),
            "supplier_id": supplier_id,
            "mean_weekly_demand": mean_demand,
            "demand_variance": round(demand_variance, 2),
            "current_inventory": current_inventory,
            "min_order_quantity": min_order_quantity,
            "lead_time_weeks": lead_time,
            "holding_cost_rate": round(holding_cost_rate, 4),
            "weekly_holding_cost": round(holding_cost, 4),
            "stockout_penalty": round(stockout_penalty, 2),
            "service_level_target": service_level_target
        })
        
    df = pd.DataFrame(skus)
    return df

if __name__ == "__main__":
    df = generate_sku_dataset()
    print(df.head())
    print(f"Total SKUs: {len(df)}")
    print(f"Injected anomalies: {df['is_anomaly_injected'].sum()}")
