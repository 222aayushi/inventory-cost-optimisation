import os
import duckdb
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime
from src.data_generator import generate_sku_dataset

DB_PATH = "data/inventory.db"

def get_db_connection() -> duckdb.DuckDBPyConnection:
    """
    Returns a DuckDB connection. Creates the data directory if it doesn't exist.
    """
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    return duckdb.connect(DB_PATH)

def init_db(force_recreate: bool = False) -> None:
    """
    Initializes the database and seeds it with synthetic SKU data if empty or forced.
    """
    conn = get_db_connection()
    try:
        # Check if sku_master exists
        tables = conn.execute("SHOW TABLES").fetchall()
        table_names = [t[0] for t in tables]
        
        if "sku_master" not in table_names or force_recreate:
            print("Initializing and seeding DuckDB tables...")
            df = generate_sku_dataset(num_skus=1000)
            
            # Write sku_master
            conn.execute("DROP TABLE IF EXISTS sku_master")
            conn.execute("""
                CREATE TABLE sku_master (
                    sku_id VARCHAR PRIMARY KEY,
                    category VARCHAR,
                    fabric VARCHAR,
                    style_group VARCHAR,
                    unit_cost DOUBLE,
                    is_anomaly_injected INTEGER,
                    anomaly_factor DOUBLE,
                    unit_volume DOUBLE,
                    supplier_id VARCHAR,
                    mean_weekly_demand DOUBLE,
                    demand_variance DOUBLE,
                    current_inventory DOUBLE,
                    min_order_quantity DOUBLE,
                    lead_time_weeks INTEGER,
                    holding_cost_rate DOUBLE,
                    weekly_holding_cost DOUBLE,
                    stockout_penalty DOUBLE,
                    service_level_target DOUBLE
                )
            """)
            conn.register('df_view', df)
            conn.execute("INSERT INTO sku_master SELECT * FROM df_view")
            conn.unregister('df_view')
            print("sku_master seeded successfully.")
            
        if "optimization_runs" not in table_names or force_recreate:
            conn.execute("DROP TABLE IF EXISTS optimization_runs")
            conn.execute("""
                CREATE TABLE optimization_runs (
                    run_id VARCHAR PRIMARY KEY,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    solver_type VARCHAR,
                    solver_status VARCHAR,
                    budget_cap DOUBLE,
                    capacity_cap DOUBLE,
                    service_level_target DOUBLE,
                    total_holding_cost DOUBLE,
                    total_stockout_penalty DOUBLE,
                    total_cost DOUBLE,
                    savings_vs_base DOUBLE,
                    run_time_seconds DOUBLE,
                    num_skus INTEGER
                )
            """)
            print("optimization_runs table initialized.")
            
        if "run_recommendations" not in table_names or force_recreate:
            conn.execute("DROP TABLE IF EXISTS run_recommendations")
            conn.execute("""
                CREATE TABLE run_recommendations (
                    run_id VARCHAR,
                    sku_id VARCHAR,
                    recommended_order_qty DOUBLE,
                    reorder_flag INTEGER,
                    expected_shortfall DOUBLE,
                    expected_holding_cost DOUBLE,
                    expected_stockout_cost DOUBLE,
                    PRIMARY KEY (run_id, sku_id)
                )
            """)
            print("run_recommendations table initialized.")
            
    finally:
        conn.close()

def load_skus() -> pd.DataFrame:
    """
    Loads all SKUs from the database as a pandas DataFrame.
    """
    init_db()
    conn = get_db_connection()
    try:
        df = conn.execute("SELECT * FROM sku_master").df()
        return df
    finally:
        conn.close()

def save_optimization_run(
    run_id: str,
    solver_type: str,
    solver_status: str,
    budget_cap: float,
    capacity_cap: float,
    service_level_target: float,
    total_holding_cost: float,
    total_stockout_penalty: float,
    total_cost: float,
    savings_vs_base: float,
    run_time_seconds: float,
    recommendations_df: pd.DataFrame
) -> None:
    """
    Saves run metadata and per-SKU recommendations to the database.
    """
    conn = get_db_connection()
    try:
        # Insert run metadata
        conn.execute("""
            INSERT INTO optimization_runs (
                run_id, solver_type, solver_status, budget_cap, capacity_cap, 
                service_level_target, total_holding_cost, total_stockout_penalty, 
                total_cost, savings_vs_base, run_time_seconds, num_skus
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            run_id, solver_type, solver_status, budget_cap, capacity_cap,
            service_level_target, total_holding_cost, total_stockout_penalty,
            total_cost, savings_vs_base, run_time_seconds, len(recommendations_df)
        ))
        
        # Insert recommendations
        # Make sure recommendations_df contains [sku_id, recommended_order_qty, reorder_flag, expected_shortfall, expected_holding_cost, expected_stockout_cost]
        recommendations_df['run_id'] = run_id
        conn.register('rec_view', recommendations_df)
        conn.execute("""
            INSERT INTO run_recommendations (
                run_id, sku_id, recommended_order_qty, reorder_flag, 
                expected_shortfall, expected_holding_cost, expected_stockout_cost
            ) SELECT run_id, sku_id, recommended_order_qty, reorder_flag, 
                     expected_shortfall, expected_holding_cost, expected_stockout_cost 
              FROM rec_view
        """)
        conn.unregister('rec_view')
    finally:
        conn.close()

def get_runs_history() -> pd.DataFrame:
    """
    Retrieves the execution history of optimization runs.
    """
    conn = get_db_connection()
    try:
        df = conn.execute("SELECT * FROM optimization_runs ORDER BY timestamp DESC").df()
        return df
    finally:
        conn.close()

def get_run_recommendations(run_id: str) -> pd.DataFrame:
    """
    Retrieves recommended order quantities for a specific run.
    """
    conn = get_db_connection()
    try:
        df = conn.execute("""
            SELECT r.*, s.category, s.fabric, s.style_group, s.unit_cost, s.current_inventory
            FROM run_recommendations r
            JOIN sku_master s ON r.sku_id = s.sku_id
            WHERE r.run_id = ?
        """, (run_id,)).df()
        return df
    finally:
        conn.close()

if __name__ == "__main__":
    init_db(force_recreate=True)
    skus_df = load_skus()
    print(f"Loaded {len(skus_df)} SKUs from database.")
