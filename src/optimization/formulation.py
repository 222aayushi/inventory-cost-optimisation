import time
import uuid
import numpy as np
import pandas as pd
import pulp
from scipy.optimize import linprog
from typing import Dict, Any, Tuple, Optional

def generate_demand_scenarios(
    skus_df: pd.DataFrame, 
    num_scenarios: int = 15, 
    seed: int = 42
) -> np.ndarray:
    """
    Generates demand scenarios for each SKU using normal distribution clipped at 0.
    Returns an array of shape (num_skus, num_scenarios).
    """
    np.random.seed(seed)
    num_skus = len(skus_df)
    scenarios = np.zeros((num_skus, num_scenarios))
    
    for idx, row in skus_df.iterrows():
        mean = row["mean_weekly_demand"]
        std = np.sqrt(row["demand_variance"])
        samples = np.random.normal(mean, std, num_scenarios)
        scenarios[idx, :] = np.clip(samples, 0, None)
        
    return scenarios

def solve_inventory_mip(
    skus_df: pd.DataFrame,
    budget_cap: float,
    capacity_cap: float,
    service_level_target: float,
    num_scenarios: int = 15,
    solver_name: str = "PULP_CBC_CMD"
) -> Tuple[str, float, pd.DataFrame, Dict[str, Any]]:
    """
    Solves the Mixed-Integer Programming model using PuLP.
    """
    start_time = time.time()
    
    sku_ids = skus_df["sku_id"].tolist()
    num_skus = len(sku_ids)
    
    # Generate scenarios
    demand_scenarios = generate_demand_scenarios(skus_df, num_scenarios)
    
    # Create the PuLP Problem
    prob = pulp.LpProblem("Inventory_Cost_Optimization_MIP", pulp.LpMinimize)
    
    # Decision variables
    x = pulp.LpVariable.dicts("order_qty", sku_ids, lowBound=0, cat=pulp.LpContinuous)
    y = pulp.LpVariable.dicts("reorder", sku_ids, cat=pulp.LpBinary)
    
    # Scenario-specific ending inventory and shortfall variables
    I_vars = {}
    V_vars = {}
    for i in range(num_skus):
        sku_id = sku_ids[i]
        for s in range(num_scenarios):
            I_vars[(sku_id, s)] = pulp.LpVariable(f"I_{sku_id}_{s}", lowBound=0, cat=pulp.LpContinuous)
            V_vars[(sku_id, s)] = pulp.LpVariable(f"V_{sku_id}_{s}", lowBound=0, cat=pulp.LpContinuous)
            
    # Objective function: Minimize expected holding cost + expected stockout penalty
    holding_costs = []
    stockout_costs = []
    
    for i in range(num_skus):
        sku_id = sku_ids[i]
        h_cost = skus_df.loc[i, "weekly_holding_cost"]
        p_cost = skus_df.loc[i, "stockout_penalty"]
        
        for s in range(num_scenarios):
            holding_costs.append(h_cost * I_vars[(sku_id, s)])
            stockout_costs.append(p_cost * V_vars[(sku_id, s)])
            
    prob += (pulp.lpSum(holding_costs) + pulp.lpSum(stockout_costs)) / num_scenarios
    
    # Constraints
    # 1. Budget Constraint
    prob += pulp.lpSum(skus_df.loc[i, "unit_cost"] * x[sku_ids[i]] for i in range(num_skus)) <= budget_cap, "Budget_Cap"
    
    # 2. Warehouse capacity constraint (on incoming order volume)
    prob += pulp.lpSum(skus_df.loc[i, "unit_volume"] * x[sku_ids[i]] for i in range(num_skus)) <= capacity_cap, "Warehouse_Capacity"
    
    # 3. Supplier Capacity Constraints (assume 80,000 units max per supplier)
    supplier_capacity = 80000.0
    suppliers = skus_df["supplier_id"].unique()
    for supplier in suppliers:
        supplier_indices = skus_df[skus_df["supplier_id"] == supplier].index
        prob += pulp.lpSum(x[sku_ids[idx]] for idx in supplier_indices) <= supplier_capacity, f"Supplier_Cap_{supplier}"
        
    # 4. SKU-level constraints: MOQ, Inventory balance, service level
    for i in range(num_skus):
        sku_id = sku_ids[i]
        moq = skus_df.loc[i, "min_order_quantity"]
        current_stock = skus_df.loc[i, "current_inventory"]
        mean_dem = skus_df.loc[i, "mean_weekly_demand"]
        
        # Big-M constraint setting: tight bounds are crucial for CBC performance
        # We cap order size by either budget or warehouse space
        m_budget = budget_cap / skus_df.loc[i, "unit_cost"]
        m_vol = capacity_cap / skus_df.loc[i, "unit_volume"]
        M = min(5 * mean_dem, m_budget, m_vol)
        M = max(M, moq + 1)
        
        # MOQ bounds
        prob += x[sku_id] >= moq * y[sku_id], f"MOQ_Lower_{sku_id}"
        prob += x[sku_id] <= M * y[sku_id], f"MOQ_Upper_{sku_id}"
        
        # Inventory balance per scenario
        for s in range(num_scenarios):
            dem = demand_scenarios[i, s]
            prob += current_stock + x[sku_id] - dem == I_vars[(sku_id, s)] - V_vars[(sku_id, s)], f"Inv_Bal_{sku_id}_{s}"
            
        # Service level target (Fill Rate / expected shortfall cap)
        # Expected shortfall <= (1 - target) * mean demand
        prob += (pulp.lpSum(V_vars[(sku_id, s)] for s in range(num_scenarios)) / num_scenarios) <= (1.0 - service_level_target) * mean_dem, f"Service_Level_{sku_id}"
        
    # Solve
    if solver_name == "PULP_CBC_CMD":
        solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=60)
    else:
        solver = pulp.PULP_CBC_CMD(msg=False)
        
    status_code = prob.solve(solver)
    status = pulp.LpStatus[status_code]
    
    elapsed_time = time.time() - start_time
    objective_val = pulp.value(prob.objective) if status == "Optimal" else None
    
    # Compile outputs
    recs = []
    for i in range(num_skus):
        sku_id = sku_ids[i]
        rec_qty = x[sku_id].varValue if status == "Optimal" else 0.0
        reorder = y[sku_id].varValue if status == "Optimal" else 0.0
        
        # Calculate expected shortfall and ending inventory from solver variables
        if status == "Optimal":
            avg_shortfall = sum(V_vars[(sku_id, s)].varValue for s in range(num_scenarios)) / num_scenarios
            avg_ending_inv = sum(I_vars[(sku_id, s)].varValue for s in range(num_scenarios)) / num_scenarios
        else:
            avg_shortfall = max(0.0, skus_df.loc[i, "mean_weekly_demand"] - skus_df.loc[i, "current_inventory"])
            avg_ending_inv = max(0.0, skus_df.loc[i, "current_inventory"] - skus_df.loc[i, "mean_weekly_demand"])
            
        holding_c = avg_ending_inv * skus_df.loc[i, "weekly_holding_cost"]
        stockout_c = avg_shortfall * skus_df.loc[i, "stockout_penalty"]
        
        recs.append({
            "sku_id": sku_id,
            "recommended_order_qty": round(rec_qty, 2) if rec_qty is not None else 0.0,
            "reorder_flag": int(round(reorder)) if reorder is not None else 0,
            "expected_shortfall": round(avg_shortfall, 2),
            "expected_holding_cost": round(holding_c, 2),
            "expected_stockout_cost": round(stockout_c, 2)
        })
        
    recs_df = pd.DataFrame(recs)
    
    diagnostics = {
        "status": status,
        "objective_value": objective_val,
        "runtime_seconds": elapsed_time,
        "num_variables": len(prob.variables()),
        "num_constraints": len(prob.constraints),
        "gap": 0.0 # CBC doesn't expose gap easily without log parsing
    }
    

    return status, objective_val, recs_df, diagnostics

def solve_inventory_lp_relaxation(
    skus_df: pd.DataFrame,
    budget_cap: float,
    capacity_cap: float,
    service_level_target: float,
    num_scenarios: int = 15
) -> Tuple[str, float, pd.DataFrame, Dict[str, Any], Dict[str, float]]:
    """
    Solves the continuous LP relaxation using PuLP to extract shadow prices.
    Returns: status, objective, recommendations_df, diagnostics, duals
    """
    start_time = time.time()
    sku_ids = skus_df["sku_id"].tolist()
    num_skus = len(sku_ids)
    demand_scenarios = generate_demand_scenarios(skus_df, num_scenarios)
    
    prob = pulp.LpProblem("Inventory_LP_Relaxation", pulp.LpMinimize)
    
    # Decision variables (y is relaxed to continuous in [0, 1])
    x = pulp.LpVariable.dicts("order_qty", sku_ids, lowBound=0, cat=pulp.LpContinuous)
    y = pulp.LpVariable.dicts("reorder", sku_ids, lowBound=0, upBound=1, cat=pulp.LpContinuous)
    
    I_vars = {}
    V_vars = {}
    for i in range(num_skus):
        sku_id = sku_ids[i]
        for s in range(num_scenarios):
            I_vars[(sku_id, s)] = pulp.LpVariable(f"I_{sku_id}_{s}", lowBound=0, cat=pulp.LpContinuous)
            V_vars[(sku_id, s)] = pulp.LpVariable(f"V_{sku_id}_{s}", lowBound=0, cat=pulp.LpContinuous)
            
    # Objective
    holding_costs = []
    stockout_costs = []
    for i in range(num_skus):
        sku_id = sku_ids[i]
        h_cost = skus_df.loc[i, "weekly_holding_cost"]
        p_cost = skus_df.loc[i, "stockout_penalty"]
        for s in range(num_scenarios):
            holding_costs.append(h_cost * I_vars[(sku_id, s)])
            stockout_costs.append(p_cost * V_vars[(sku_id, s)])
            
    prob += (pulp.lpSum(holding_costs) + pulp.lpSum(stockout_costs)) / num_scenarios
    
    # Constraints
    # Store constraint objects to query their duals
    budget_con = pulp.lpSum(skus_df.loc[i, "unit_cost"] * x[sku_ids[i]] for i in range(num_skus)) <= budget_cap
    capacity_con = pulp.lpSum(skus_df.loc[i, "unit_volume"] * x[sku_ids[i]] for i in range(num_skus)) <= capacity_cap
    
    prob += budget_con, "Budget_Cap"
    prob += capacity_con, "Warehouse_Capacity"
    
    supplier_capacity = 80000.0
    suppliers = skus_df["supplier_id"].unique()
    for supplier in suppliers:
        supplier_indices = skus_df[skus_df["supplier_id"] == supplier].index
        prob += pulp.lpSum(x[sku_ids[idx]] for idx in supplier_indices) <= supplier_capacity, f"Supplier_Cap_{supplier}"
        
    for i in range(num_skus):
        sku_id = sku_ids[i]
        moq = skus_df.loc[i, "min_order_quantity"]
        current_stock = skus_df.loc[i, "current_inventory"]
        mean_dem = skus_df.loc[i, "mean_weekly_demand"]
        
        m_budget = budget_cap / skus_df.loc[i, "unit_cost"]
        m_vol = capacity_cap / skus_df.loc[i, "unit_volume"]
        M = min(5 * mean_dem, m_budget, m_vol)
        M = max(M, moq + 1)
        
        prob += x[sku_id] >= moq * y[sku_id], f"MOQ_Lower_{sku_id}"
        prob += x[sku_id] <= M * y[sku_id], f"MOQ_Upper_{sku_id}"
        
        for s in range(num_scenarios):
            dem = demand_scenarios[i, s]
            prob += current_stock + x[sku_id] - dem == I_vars[(sku_id, s)] - V_vars[(sku_id, s)], f"Inv_Bal_{sku_id}_{s}"
            
        # Service level target (Fill Rate / expected shortfall cap)
        # Expected shortfall <= (1 - target) * mean demand
        prob += (pulp.lpSum(V_vars[(sku_id, s)] for s in range(num_scenarios)) / num_scenarios) <= (1.0 - service_level_target) * mean_dem, f"Service_Level_{sku_id}"
        
    solver = pulp.PULP_CBC_CMD(msg=False)
    status_code = prob.solve(solver)
    status = pulp.LpStatus[status_code]
    
    elapsed_time = time.time() - start_time
    objective_val = pulp.value(prob.objective) if status == "Optimal" else None
    
    # Retrieve dual variables (shadow prices)
    duals = {}
    if status == "Optimal":
        # Dual value is stored in .pi attribute in PuLP
        duals["budget_shadow_price"] = budget_con.pi if budget_con.pi is not None else 0.0
        duals["capacity_shadow_price"] = capacity_con.pi if capacity_con.pi is not None else 0.0
    else:
        duals["budget_shadow_price"] = 0.0
        duals["capacity_shadow_price"] = 0.0
        
    recs = []
    for i in range(num_skus):
        sku_id = sku_ids[i]
        rec_qty = x[sku_id].varValue if status == "Optimal" else 0.0
        reorder = y[sku_id].varValue if status == "Optimal" else 0.0
        
        if status == "Optimal":
            avg_shortfall = sum(V_vars[(sku_id, s)].varValue for s in range(num_scenarios)) / num_scenarios
            avg_ending_inv = sum(I_vars[(sku_id, s)].varValue for s in range(num_scenarios)) / num_scenarios
        else:
            avg_shortfall = max(0.0, skus_df.loc[i, "mean_weekly_demand"] - skus_df.loc[i, "current_inventory"])
            avg_ending_inv = max(0.0, skus_df.loc[i, "current_inventory"] - skus_df.loc[i, "mean_weekly_demand"])
            
        holding_c = avg_ending_inv * skus_df.loc[i, "weekly_holding_cost"]
        stockout_c = avg_shortfall * skus_df.loc[i, "stockout_penalty"]
        
        recs.append({
            "sku_id": sku_id,
            "recommended_order_qty": round(rec_qty, 2) if rec_qty is not None else 0.0,
            "reorder_flag": float(reorder) if reorder is not None else 0.0,
            "expected_shortfall": round(avg_shortfall, 2),
            "expected_holding_cost": round(holding_c, 2),
            "expected_stockout_cost": round(stockout_c, 2)
        })
        
    recs_df = pd.DataFrame(recs)
    
    diagnostics = {
        "status": status,
        "objective_value": objective_val,
        "runtime_seconds": elapsed_time,
        "num_variables": len(prob.variables()),
        "num_constraints": len(prob.constraints)
    }
    

    return status, objective_val, recs_df, diagnostics, duals

def solve_inventory_scipy_lp(
    skus_df: pd.DataFrame,
    budget_cap: float,
    capacity_cap: float,
    service_level_target: float,
    num_scenarios: int = 15
) -> Tuple[str, float, pd.DataFrame, Dict[str, Any]]:
    """
    Solves the continuous LP relaxation using SciPy's linprog.
    Uses: vector z = [x_0..x_N-1, y_0..y_N-1, I_0,0..I_N-1,S-1, V_0,0..V_N-1,S-1]
    Total variables: N + N + N*S + N*S = 2N + 2NS.
    """
    start_time = time.time()
    sku_ids = skus_df["sku_id"].tolist()
    num_skus = len(sku_ids)
    demand_scenarios = generate_demand_scenarios(skus_df, num_scenarios)
    
    N = num_skus
    S = num_scenarios
    num_vars = 2 * N + 2 * N * S
    
    # Indices helper:
    # x_idx: 0 to N-1
    # y_idx: N to 2N-1
    # I_idx(i, s): 2N + i*S + s
    # V_idx(i, s): 2N + N*S + i*S + s
    
    # 1. Objective function vector c
    c = np.zeros(num_vars)
    for i in range(N):
        h_cost = skus_df.loc[i, "weekly_holding_cost"]
        p_cost = skus_df.loc[i, "stockout_penalty"]
        for s in range(S):
            I_idx = 2 * N + i * S + s
            V_idx = 2 * N + N * S + i * S + s
            c[I_idx] = h_cost / S
            c[V_idx] = p_cost / S
            
    # Bounds for each variable
    bounds = []
    # x bounds: [0, inf]
    for i in range(N):
        bounds.append((0, None))
    # y bounds: [0, 1] for LP relaxation
    for i in range(N):
        bounds.append((0, 1))
    # I bounds: [0, inf]
    for idx in range(N * S):
        bounds.append((0, None))
    # V bounds: [0, inf]
    for idx in range(N * S):
        bounds.append((0, None))
        
    # Linear constraints list
    A_ub = []
    b_ub = []
    A_eq = []
    b_eq = []
    
    # 1. Budget Constraint: sum(unit_cost * x) <= budget_cap
    row = np.zeros(num_vars)
    for i in range(N):
        row[i] = skus_df.loc[i, "unit_cost"]
    A_ub.append(row)
    b_ub.append(budget_cap)
    
    # 2. Warehouse Space Constraint: sum(volume * x) <= capacity_cap
    row = np.zeros(num_vars)
    for i in range(N):
        row[i] = skus_df.loc[i, "unit_volume"]
    A_ub.append(row)
    b_ub.append(capacity_cap)
    
    # 3. Supplier constraints (assume 80,000 max per supplier)
    supplier_capacity = 80000.0
    suppliers = skus_df["supplier_id"].unique()
    for supplier in suppliers:
        supplier_indices = skus_df[skus_df["supplier_id"] == supplier].index
        row = np.zeros(num_vars)
        for idx in supplier_indices:
            row[idx] = 1.0
        A_ub.append(row)
        b_ub.append(supplier_capacity)
        
    # 4. MOQ Constraints:
    # x_i >= moq_i * y_i => moq_i * y_i - x_i <= 0
    # x_i <= M_i * y_i => x_i - M_i * y_i <= 0
    for i in range(N):
        moq = skus_df.loc[i, "min_order_quantity"]
        mean_dem = skus_df.loc[i, "mean_weekly_demand"]
        m_budget = budget_cap / skus_df.loc[i, "unit_cost"]
        m_vol = capacity_cap / skus_df.loc[i, "unit_volume"]
        M = min(5 * mean_dem, m_budget, m_vol)
        M = max(M, moq + 1)
        
        # MOQ Lower: moq_i * y_i - x_i <= 0
        row_l = np.zeros(num_vars)
        row_l[i] = -1.0       # x_i
        row_l[N + i] = moq   # y_i
        A_ub.append(row_l)
        b_ub.append(0.0)
        
        # MOQ Upper: x_i - M_i * y_i <= 0
        row_u = np.zeros(num_vars)
        row_u[i] = 1.0       # x_i
        row_u[N + i] = -M    # y_i
        A_ub.append(row_u)
        b_ub.append(0.0)
        
    # 5. Service level constraint per SKU:
    # sum_s(V_i,s) / S <= (1 - target) * mean_demand
    for i in range(N):
        target = skus_df.loc[i, "service_level_target"] # default 0.95
        mean_dem = skus_df.loc[i, "mean_weekly_demand"]
        
        row = np.zeros(num_vars)
        for s in range(S):
            V_idx = 2 * N + N * S + i * S + s
            row[V_idx] = 1.0 / S
        A_ub.append(row)
        b_ub.append((1.0 - service_level_target) * mean_dem)
        
    # 6. Inventory Balance (Equality constraints)
    # current_inventory + x_i - d_i,s = I_i,s - V_i,s
    # => x_i - I_i,s + V_i,s = d_i,s - current_inventory
    for i in range(N):
        current_stock = skus_df.loc[i, "current_inventory"]
        for s in range(S):
            dem = demand_scenarios[i, s]
            
            row = np.zeros(num_vars)
            row[i] = 1.0                              # x_i
            I_idx = 2 * N + i * S + s
            V_idx = 2 * N + N * S + i * S + s
            row[I_idx] = -1.0                         # -I_i,s
            row[V_idx] = 1.0                          # V_i,s
            
            A_eq.append(row)
            b_eq.append(dem - current_stock)
            
    # Convert lists to numpy arrays
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    A_eq = np.array(A_eq)
    b_eq = np.array(b_eq)
    
    # Solve using SciPy linprog with Highs solver
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    
    elapsed_time = time.time() - start_time
    status = "Optimal" if res.success else "Infeasible"
    
    recs = []
    for i in range(N):
        sku_id = sku_ids[i]
        rec_qty = res.x[i] if res.success else 0.0
        reorder = res.x[N + i] if res.success else 0.0
        
        if res.success:
            avg_shortfall = sum(res.x[2 * N + N * S + i * S + s] for s in range(S)) / S
            avg_ending_inv = sum(res.x[2 * N + i * S + s] for s in range(S)) / S
        else:
            avg_shortfall = max(0.0, skus_df.loc[i, "mean_weekly_demand"] - skus_df.loc[i, "current_inventory"])
            avg_ending_inv = max(0.0, skus_df.loc[i, "current_inventory"] - skus_df.loc[i, "mean_weekly_demand"])
            
        holding_c = avg_ending_inv * skus_df.loc[i, "weekly_holding_cost"]
        stockout_c = avg_shortfall * skus_df.loc[i, "stockout_penalty"]
        
        recs.append({
            "sku_id": sku_id,
            "recommended_order_qty": round(rec_qty, 2),
            "reorder_flag": float(reorder),
            "expected_shortfall": round(avg_shortfall, 2),
            "expected_holding_cost": round(holding_c, 2),
            "expected_stockout_cost": round(stockout_c, 2)
        })
        
    recs_df = pd.DataFrame(recs)
    
    diagnostics = {
        "status": status,
        "objective_value": res.fun if res.success else None,
        "runtime_seconds": elapsed_time,
        "num_variables": num_vars,
        "num_constraints": len(A_ub) + len(A_eq)
    }
    

    return status, res.fun if res.success else None, recs_df, diagnostics

def evaluate_reorder_policy(
    skus_df: pd.DataFrame,
    order_qtys: Dict[str, float],
    num_scenarios: int = 15,
    seed: int = 42
) -> Tuple[float, float, float]:
    """
    Evaluates the holding and stockout costs for a given order quantity policy
    across generated scenarios. Returns (expected_holding_cost, expected_stockout_cost, total_cost).
    """
    demand_scenarios = generate_demand_scenarios(skus_df, num_scenarios, seed)
    sku_ids = skus_df["sku_id"].tolist()
    num_skus = len(sku_ids)
    
    total_holding = 0.0
    total_stockout = 0.0
    
    for i in range(num_skus):
        sku_id = sku_ids[i]
        x_qty = order_qtys.get(sku_id, 0.0)
        current_stock = skus_df.loc[i, "current_inventory"]
        h_cost = skus_df.loc[i, "weekly_holding_cost"]
        p_cost = skus_df.loc[i, "stockout_penalty"]
        
        for s in range(num_scenarios):
            dem = demand_scenarios[i, s]
            ending_inv = max(0.0, current_stock + x_qty - dem)
            shortfall = max(0.0, dem - current_stock - x_qty)
            
            total_holding += h_cost * ending_inv
            total_stockout += p_cost * shortfall
            
    avg_holding = total_holding / num_scenarios
    avg_stockout = total_stockout / num_scenarios
    total_cost = avg_holding + avg_stockout
    
    return avg_holding, avg_stockout, total_cost

if __name__ == "__main__":
    from src.database import load_skus
    skus_df = load_skus()
    print("Running a sample MIP solve...")
    status, obj, recs_df, diag = solve_inventory_mip(
        skus_df.head(20), # Small test
        budget_cap=250000.0,
        capacity_cap=200.0,
        service_level_target=0.90,
        num_scenarios=5
    )
    print(f"MIP Status: {status}, Obj: {obj}")
    print(diag)
    print(recs_df.head())
    
    print("\nRunning a sample SciPy LP solve...")
    status, obj, recs_df, diag = solve_inventory_scipy_lp(
        skus_df.head(20),
        budget_cap=250000.0,
        capacity_cap=200.0,
        service_level_target=0.90,
        num_scenarios=5
    )
    print(f"SciPy LP Status: {status}, Obj: {obj}")
    print(diag)


