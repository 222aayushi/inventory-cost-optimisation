import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional

def detect_cost_anomalies(
    skus_df: pd.DataFrame, 
    recommendations_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Identifies cost outliers within style groups using robust statistics (Median + IQR).
    Ranks flagged anomalies by potential negotiation dollar savings.
    """
    df = skus_df.copy()
    
    # Calculate median and IQR per style group
    stats = df.groupby("style_group")["unit_cost"].agg(
        median="median",
        q25=lambda x: np.percentile(x, 25),
        q75=lambda x: np.percentile(x, 75)
    ).reset_index()
    
    stats["iqr"] = stats["q75"] - stats["q25"]
    
    # Merge stats back to master SKU list
    df = df.merge(stats, on="style_group", how="left")
    
    # Define threshold for anomalies. 
    # Fallback to 15% above median if IQR is very small to avoid dividing/flagging minor noise.
    df["anomaly_threshold"] = df["median"] + 1.5 * df["iqr"]
    df["anomaly_threshold"] = np.where(
        df["iqr"] < 0.05 * df["median"],
        df["median"] * 1.15,
        df["anomaly_threshold"]
    )
    
    # Flag anomalies (only positive deviations represent negotiation savings)
    df["is_anomaly"] = (df["unit_cost"] > df["anomaly_threshold"]).astype(int)
    
    # Merge order quantities if available, else fall back to mean weekly demand
    if recommendations_df is not None:
        rec_subset = recommendations_df[["sku_id", "recommended_order_qty"]]
        df = df.merge(rec_subset, on="sku_id", how="left")
        df["order_volume"] = df["recommended_order_qty"].fillna(0.0)
    else:
        df["order_volume"] = df["mean_weekly_demand"]
        
    # Calculate potential savings: deviation from median * order volume
    df["cost_deviation"] = df["unit_cost"] - df["median"]
    df["pct_deviation"] = (df["cost_deviation"] / df["median"]) * 100.0
    
    # Potential dollar savings
    df["potential_savings"] = df["cost_deviation"] * df["order_volume"]
    df["potential_savings"] = np.where(df["is_anomaly"] == 1, df["potential_savings"], 0.0)
    
    # Filter anomalies and sort by potential savings descending
    anomalies = df[df["is_anomaly"] == 1].copy()
    anomalies = anomalies.sort_values(by="potential_savings", ascending=False)
    
    # Generate explanations
    explanations = []
    for _, row in anomalies.iterrows():
        sku = row["sku_id"]
        cost = row["unit_cost"]
        med = row["median"]
        pct = row["pct_deviation"]
        vol = row["order_volume"]
        sav = row["potential_savings"]
        grp = row["style_group"]
        cat = row["category"]
        fab = row["fabric"]
        
        explanation = (
            f"Unit cost (${cost:.2f}) is {pct:.1f}% above the peer group median "
            f"(${med:.2f}) for '{grp}' (Category: {cat}, Fabric: {fab}). "
            f"With a recommended purchase volume of {int(vol)} units, renegotiating "
            f"this price down to the group median yields a potential savings of ${sav:,.2f}."
        )
        explanations.append(explanation)
        
    anomalies["explanation"] = explanations
    
    # Keep only relevant columns for output
    output_cols = [
        "sku_id", "category", "fabric", "style_group", "supplier_id",
        "unit_cost", "median", "pct_deviation", "order_volume", 
        "potential_savings", "explanation"
    ]
    
    return anomalies[output_cols].reset_index(drop=True)

if __name__ == "__main__":
    from src.database import load_skus
    skus_df = load_skus()
    anomalies_df = detect_cost_anomalies(skus_df)
    print(f"Detected {len(anomalies_df)} anomalies.")
    if len(anomalies_df) > 0:
        print("\nTop anomaly explanation:")
        print(anomalies_df.loc[0, "explanation"])
