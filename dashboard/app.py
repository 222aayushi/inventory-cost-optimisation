import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import time
import os

# Try importing backend modules directly for local fallback
try:
    from src.database import load_skus, init_db
    from src.optimization.formulation import solve_inventory_mip, solve_inventory_scipy_lp, evaluate_reorder_policy
    from src.optimization.sensitivity import run_sensitivity_analysis
    from src.anomaly_detection.detector import detect_cost_anomalies
    from src.tracking.tracker import OptimizationTracker
    BACKEND_AVAILABLE = True
except ImportError:
    BACKEND_AVAILABLE = False

# API Base URL
API_URL = os.environ.get("API_URL", "http://localhost:8000")

# Setup page layout
st.set_page_config(
    page_title="Luxury Retail | Inventory Cost Optimization",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Injection for Luxury Brand Aesthetic
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:ital,wght@0,400;0,600;0,700;1,400&display=swap');

    /* Global page styling */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: #0B0B0C !important;
        color: #F3F4F6 !important;
        font-family: 'Inter', sans-serif !important;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #121214 !important;
        border-right: 1px solid #232326 !important;
    }
    
    /* Typography */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Playfair Display', serif !important;
        font-weight: 600 !important;
        color: #F8F9FA !important;
        letter-spacing: 0.5px;
    }
    h1 {
        border-bottom: 1px solid #232326;
        padding-bottom: 15px;
        margin-bottom: 25px;
    }
    
    /* Custom KPI Cards */
    .kpi-container {
        display: flex;
        justify-content: space-between;
        gap: 15px;
        margin-bottom: 25px;
    }
    .kpi-card {
        background-color: #141416;
        border: 1px solid #28282B;
        border-radius: 8px;
        padding: 20px;
        flex: 1;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
        transition: transform 0.2s, border-color 0.2s;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        border-color: #F9D3D8;
    }
    .kpi-value {
        font-size: 2rem;
        font-weight: 700;
        color: #F9D3D8; /* Blush Pink */
        font-family: 'Playfair Display', serif;
        margin-bottom: 5px;
    }
    .kpi-label {
        font-size: 0.75rem;
        color: #D4AF37; /* Champagne Gold */
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }
    
    /* Metric subtexts */
    .kpi-subtext {
        font-size: 0.75rem;
        color: #8C8C96;
        margin-top: 5px;
    }
    
    /* Elegant input elements */
    .stSlider, .stSelectbox, .stTextInput, .stButton button {
        background-color: #141416 !important;
        color: #F3F4F6 !important;
    }
    
    /* Custom tab indicators */
    .stTabs [data-baseweb="tab-list"] {
        gap: 15px;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'Playfair Display', serif !important;
        font-size: 1.1rem !important;
        color: #8C8C96 !important;
        background-color: transparent !important;
        border-bottom: 2px solid transparent !important;
        padding: 8px 16px !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #F9D3D8 !important;
    }
    .stTabs [aria-selected="true"] {
        color: #F9D3D8 !important;
        border-bottom-color: #F9D3D8 !important;
    }
    
    /* Warning/Info alert overrides */
    .stAlert {
        background-color: #1B1B1E !important;
        border: 1px solid #34343D !important;
        color: #F3F4F6 !important;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 20px;
        font-size: 0.8rem;
        color: #5C5C64;
        border-top: 1px solid #1C1C1E;
        margin-top: 50px;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to check API status and set up mode
def check_api_status():
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        if response.status_code == 200:
            return "API", response.json()
    except requests.exceptions.RequestException:
        pass
    
    if BACKEND_AVAILABLE:
        return "LOCAL", None
    else:
        return "UNAVAILABLE", None

mode, health_data = check_api_status()

# Sidebar Setup
st.sidebar.markdown(
    "<h3 style='color: #F9D3D8; font-family: Playfair Display; margin-bottom: 0px;'>LUXURY RETAIL BRAND</h3>"
    "<p style='font-size: 0.8rem; color: #8C8C96; letter-spacing: 1px; text-transform: uppercase;'>Supply Chain Engine</p>",
    unsafe_allow_html=True
)

st.sidebar.markdown("---")

# Initialize session state for default calibration values
if "budget_val" not in st.session_state:
    st.session_state.budget_val = 10000000.0
if "capacity_val" not in st.session_state:
    st.session_state.capacity_val = 5000.0
if "service_val" not in st.session_state:
    st.session_state.service_val = 95.0
if "scenarios_val" not in st.session_state:
    st.session_state.scenarios_val = 15
if "scipy_val" not in st.session_state:
    st.session_state.scipy_val = False

# Solver Configuration parameters in Sidebar
st.sidebar.subheader("System Configuration")
budget_slider = st.sidebar.slider("Procurement Budget ($)", 1000000.0, 20000000.0, key="budget_val", step=500000.0, format="$%d")
capacity_slider = st.sidebar.slider("Warehouse Space (m³)", 500.0, 10000.0, key="capacity_val", step=250.0)
service_slider = st.sidebar.slider("Service level target (%)", 80.0, 99.0, key="service_val", step=1.0) / 100.0
scenarios_slider = st.sidebar.slider("Stochastic Scenarios (SAA)", 5, 50, key="scenarios_val", step=5)
use_scipy = st.sidebar.checkbox("Use LP relaxation (SciPy linprog)", key="scipy_val")

# Reset Button
if st.sidebar.button("Reset to Recommended Defaults"):
    st.session_state.budget_val = 10000000.0
    st.session_state.capacity_val = 5000.0
    st.session_state.service_val = 95.0
    st.session_state.scenarios_val = 15
    st.session_state.scipy_val = False
    st.rerun()

st.sidebar.markdown("---")

# Show Connection Status
if mode == "API":
    st.sidebar.success(f"FastAPI Server Connected (SKUs in db: {health_data.get('skus_in_db')})")
elif mode == "LOCAL":
    st.sidebar.info("Running in Direct Python Mode (API Offline)")
else:
    st.sidebar.error("Error: Backend formulation files not found, and FastAPI server is offline.")
    st.stop()

# Title Banner
st.markdown(
    "<div style='margin-bottom: 20px;'>"
    "<h1 style='font-family: Playfair Display; font-size: 2.8rem; margin-bottom: 5px; color:#F8F9FA;'>INVENTORY COST OPTIMIZATION</h1>"
    "<p style='color: #8C8C96; font-size: 1.0rem; letter-spacing: 0.5px;'>Mixed-Integer Programming Formulation & Robust Cost Anomaly Flags</p>"
    "</div>",
    unsafe_allow_html=True
)

# Run Optimization Solver
@st.cache_data(ttl=60, show_spinner=False)
def get_optimization_results(budget, capacity, service, num_scenarios, scipy_flag):
    if mode == "API":
        payload = {
            "budget_cap": budget,
            "capacity_cap": capacity,
            "service_level_target": service,
            "num_scenarios": num_scenarios,
            "use_scipy_lp": scipy_flag
        }
        res = requests.post(f"{API_URL}/optimize", json=payload)
        if res.status_code == 200:
            data = res.json()
            recs_df = pd.DataFrame(data["recommendations"]) if data["status"] == "Optimal" else None
            return data["status"], data["summary"], recs_df, data["diagnostics"]
        else:
            err = res.json().get("detail", "Solver failed.")
            return "Infeasible", None, None, {"error_message": err}
    else:
        # Fallback local computation
        start_t = time.time()
        skus_df = load_skus()
        
        if scipy_flag:
            status, obj, recs_df, diag = solve_inventory_scipy_lp(skus_df, budget, capacity, service, num_scenarios)
            sol_type = "SciPy_LP_Relaxation"
        else:
            status, obj, recs_df, diag = solve_inventory_mip(skus_df, budget, capacity, service, num_scenarios)
            sol_type = "PuLP_MIP"
            
        if status != "Optimal" or obj is None:
            return "Infeasible", None, None, diag
            
        # Baseline calculations
        baseline_qtys = {row["sku_id"]: max(0.0, row["mean_weekly_demand"] - row["current_inventory"]) for _, row in skus_df.iterrows()}
        base_h, base_s, base_total = evaluate_reorder_policy(skus_df, baseline_qtys, num_scenarios)
        
        res_merged = recs_df.merge(skus_df, on="sku_id")
        opt_h = res_merged["expected_holding_cost"].sum()
        opt_s = res_merged["expected_stockout_cost"].sum()
        opt_total = opt_h + opt_s
        savings_dollar = base_total - opt_total
        savings_pct = (savings_dollar / base_total) * 100.0 if base_total > 0 else 0.0
        
        total_shortfall = res_merged["expected_shortfall"].sum()
        total_demand = res_merged["mean_weekly_demand"].sum()
        fill_rate_achieved = 1.0 - (total_shortfall / total_demand) if total_demand > 0 else 1.0
        
        summary = {
            "total_holding_cost": opt_h,
            "total_stockout_cost": opt_s,
            "total_cost": opt_total,
            "baseline_holding_cost": base_h,
            "baseline_stockout_cost": base_s,
            "baseline_total_cost": base_total,
            "savings_dollar": savings_dollar,
            "savings_percent": savings_pct,
            "service_level_achieved": fill_rate_achieved
        }
        
        # Log run in tracker
        OptimizationTracker.log_run(
            solver_type=sol_type,
            solver_status=status,
            budget_cap=budget,
            capacity_cap=capacity,
            service_level_target=service,
            total_holding_cost=opt_h,
            total_stockout_penalty=opt_s,
            total_cost=opt_total,
            savings_vs_base=savings_dollar,
            run_time_seconds=diag["runtime_seconds"],
            recommendations_df=recs_df
        )
        
        return status, summary, res_merged, diag

# Fetch Anomaly Results
@st.cache_data(ttl=60, show_spinner=False)
def get_anomalies_results():
    if mode == "API":
        res = requests.get(f"{API_URL}/anomalies")
        if res.status_code == 200:
            return pd.DataFrame(res.json())
        return pd.DataFrame()
    else:
        skus_df = load_skus()
        history = OptimizationTracker.get_history()
        recs_df = None
        if not history.empty:
            latest_run_id = history.iloc[0]["run_id"]
            recs_df = OptimizationTracker.get_run_details(latest_run_id)
        return detect_cost_anomalies(skus_df, recs_df)

with st.spinner("Invoking Mixed-Integer Solver..."):
    status, summary, recs_df, diagnostics = get_optimization_results(
        budget_slider, capacity_slider, service_slider, scenarios_slider, use_scipy
    )

# Handle Infeasibility
if status == "Infeasible":
    st.error("⚠️ Solver Infeasibility Alert")
    reason = diagnostics.get("error_message", "No detailed diagnostic reason available.")
    st.markdown(
        f"> [!IMPORTANT]\n"
        f"> **Diagnostic Reason:**\n"
        f"> {reason}\n\n"
        "**Recommended Actions:**\n"
        "- Increase the Procurement Budget slider in the sidebar.\n"
        "- Lower the Service Level Target floor in the sidebar.\n"
        "- Increase the Warehouse Space capacity limit."
    )
    st.stop()

# Layout Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Executive Summary", 
    "📈 Optimization Room", 
    "🔍 Negotiation / Anomaly Room", 
    "⚖️ What-If Sensitivity"
])

# ----------------- TAB 1: EXECUTIVE SUMMARY -----------------
with tab1:
    # KPI Row
    st.markdown(f"""
    <div class="kpi-container">
        <div class="kpi-card">
            <div class="kpi-value">${summary['savings_dollar']:,.2f}</div>
            <div class="kpi-label">Expected Cost Savings</div>
            <div class="kpi-subtext">vs. Mean Reorder Policy</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-value">{summary['savings_percent']:.1f}%</div>
            <div class="kpi-label">Cost Reduction</div>
            <div class="kpi-subtext">Percentage saved on holding+stockout</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-value">{summary['service_level_achieved']*100:.1f}%</div>
            <div class="kpi-label">Achieved Service Level</div>
            <div class="kpi-subtext">Fill rate across all SKUs</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-value">${summary['total_cost']:,.2f}</div>
            <div class="kpi-label">Optimized Total Cost</div>
            <div class="kpi-subtext">Holding: ${summary['total_holding_cost']:,.0f} | Stockout: ${summary['total_stockout_cost']:,.0f}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.subheader("Holdings & Stockout Cost Profile: Baseline vs. Optimized")
        
        # Build bar chart for comparison
        cost_comparison = pd.DataFrame({
            "Policy": ["Baseline", "Baseline", "Optimized", "Optimized"],
            "Cost Component": ["Holding Cost", "Stockout Penalty", "Holding Cost", "Stockout Penalty"],
            "Cost ($)": [
                summary["baseline_holding_cost"], 
                summary["baseline_stockout_cost"], 
                summary["total_holding_cost"], 
                summary["total_stockout_cost"]
            ]
        })
        
        fig = px.bar(
            cost_comparison, 
            x="Policy", 
            y="Cost ($)", 
            color="Cost Component",
            color_discrete_map={"Holding Cost": "#E6C17A", "Stockout Penalty": "#F9D3D8"},
            barmode="stack",
            height=400,
            text_auto=',.0f'
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#F8F9FA',
            xaxis_title="",
            yaxis=dict(gridcolor='#1C1C1E')
        )
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("Category Allocation & Volume Profile")
        
        category_summary = recs_df.groupby("category").agg(
            ordered_units=("recommended_order_qty", "sum"),
            ordered_value=("recommended_order_qty", lambda x: sum(x * recs_df.loc[x.index, "unit_cost"]))
        ).reset_index()
        
        fig_donut = px.pie(
            category_summary,
            values="ordered_value",
            names="category",
            color_discrete_sequence=["#F9D3D8", "#E6C17A", "#E08595", "#B8860B"],
            hole=0.4,
            height=400
        )
        fig_donut.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#F8F9FA'
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    # Historical solver run status table
    st.markdown("### 📜 Solver Execution History (MLOps Log)")
    if mode == "API" or BACKEND_AVAILABLE:
        runs_history = OptimizationTracker.get_history()
        if not runs_history.empty:
            runs_history["timestamp"] = pd.to_datetime(runs_history["timestamp"])
            runs_history_disp = runs_history.rename(columns={
                "run_id": "Run ID",
                "timestamp": "Timestamp (UTC)",
                "solver_type": "Solver Type",
                "solver_status": "Status",
                "budget_cap": "Budget Cap ($)",
                "capacity_cap": "Capacity (m³)",
                "service_level_target": "SL Target",
                "total_cost": "Total Cost ($)",
                "savings_vs_base": "Savings ($)",
                "run_time_seconds": "Solve Time (s)"
            })
            st.dataframe(
                runs_history_disp[["Run ID", "Timestamp (UTC)", "Solver Type", "Status", "Budget Cap ($)", "Capacity (m³)", "SL Target", "Total Cost ($)", "Savings ($)", "Solve Time (s)"]].head(8),
                use_container_width=True
            )
        else:
            st.write("No historical runs found.")

# ----------------- TAB 2: OPTIMIZATION ROOM -----------------
with tab2:
    st.subheader("SKU-Level Recommended Order Quantities")
    
    # Filter controls
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        cat_filter = st.selectbox("Category Filter", ["All"] + list(recs_df["category"].unique()))
    with col_f2:
        fab_filter = st.selectbox("Fabric Filter", ["All"] + list(recs_df["fabric"].unique()))
    with col_f3:
        status_filter = st.selectbox("Order Status Filter", ["All", "Reorder Recommended", "No Action"])
        
    filtered_df = recs_df.copy()
    if cat_filter != "All":
        filtered_df = filtered_df[filtered_df["category"] == cat_filter]
    if fab_filter != "All":
        filtered_df = filtered_df[filtered_df["fabric"] == fab_filter]
    if status_filter == "Reorder Recommended":
        filtered_df = filtered_df[filtered_df["reorder_flag"] == 1]
    elif status_filter == "No Action":
        filtered_df = filtered_df[filtered_df["reorder_flag"] == 0]
        
    # Order quantity details display
    display_cols = [
        "sku_id", "category", "fabric", "style_group", "unit_cost", 
        "current_inventory", "recommended_order_qty", "reorder_flag", 
        "expected_shortfall", "expected_holding_cost", "expected_stockout_cost"
    ]
    st.dataframe(filtered_df[display_cols].reset_index(drop=True), use_container_width=True)
    
    # Solver Diagnostics / Model structure details
    with st.expander("🛠️ Technical Solver Diagnostics"):
        col_diag1, col_diag2 = st.columns(2)
        with col_diag1:
            st.write(f"**Solver Engine**: {'SciPy linprog' if use_scipy else 'PuLP (CBC solver)'}")
            st.write(f"**Solver Status**: {diagnostics.get('status', 'N/A')}")
            st.write(f"**Execution Runtime**: {diagnostics.get('runtime_seconds', 0.0):.4f} seconds")
        with col_diag2:
            st.write(f"**Decision Variables**: {diagnostics.get('num_variables', 0)}")
            st.write(f"**Linear Constraints**: {diagnostics.get('num_constraints', 0)}")
            if not use_scipy:
                st.write(f"**Optimality Gap**: {diagnostics.get('gap', 0.0)*100:.2f}%")

# ----------------- TAB 3: ANOMALY ROOM -----------------
with tab3:
    st.subheader("Supplier Cost Anomalies & Negotiation Flagging")
    st.markdown(
        "> [!TIP]\n"
        "> Cost anomalies represent procurement negotiation opportunities. The system groups SKUs by comparable styles, "
        "computes expected peer prices using robust median statistics, and flags SKUs costing more than 1.5x IQR above the median. "
        "They are ranked by total savings potential (price premium multiplied by recommended purchase volume)."
    )
    
    anomalies_df = get_anomalies_results()
    
    if anomalies_df.empty:
        st.success("🎉 No cost anomalies detected in the current catalog!")
    else:
        # Summary metrics
        total_anomalies = len(anomalies_df)
        total_savings_potential = anomalies_df["potential_savings"].sum()
        
        st.markdown(f"""
        <div style="display:flex; gap:15px; margin-bottom:20px;">
            <div style="flex:1; background-color:#1E1416; border:1px solid #4A1A1E; padding:15px; border-radius:8px; text-align:center;">
                <h4 style="margin:0; color:#FFA4A4; font-family:Inter; font-size:0.9rem; text-transform:uppercase; letter-spacing:1px;">Outliers Flagged</h4>
                <div style="font-size:2rem; font-weight:700; color:#FFA4A4; font-family:Playfair Display;">{total_anomalies} SKUs</div>
            </div>
            <div style="flex:1; background-color:#141A16; border:1px solid #1A4A24; padding:15px; border-radius:8px; text-align:center;">
                <h4 style="margin:0; color:#A4FFA4; font-family:Inter; font-size:0.9rem; text-transform:uppercase; letter-spacing:1px;">Total Negotiation Value</h4>
                <div style="font-size:2rem; font-weight:700; color:#A4FFA4; font-family:Playfair Display;">${total_savings_potential:,.2f}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Display table
        st.dataframe(
            anomalies_df[["sku_id", "category", "fabric", "style_group", "supplier_id", "unit_cost", "median", "pct_deviation", "order_volume", "potential_savings"]].rename(columns={
                "sku_id": "SKU ID",
                "category": "Category",
                "fabric": "Fabric",
                "style_group": "Style Group",
                "supplier_id": "Supplier",
                "unit_cost": "Unit Cost ($)",
                "median": "Peer Median ($)",
                "pct_deviation": "Premium %",
                "order_volume": "Order Volume",
                "potential_savings": "Negotiation Savings ($)"
            }).style.format({
                "Unit Cost ($)": "${:.2f}",
                "Peer Median ($)": "${:.2f}",
                "Premium %": "{:.1f}%",
                "Negotiation Savings ($)": "${:,.2f}",
                "Order Volume": "{:,.0f}"
            }),
            use_container_width=True
        )
        
        # Selectbox to explore anomaly
        st.markdown("### 🔎 Negotiation Room - SKU Deep Dive")
        selected_sku = st.selectbox("Select flagged SKU to view peer comparison & script", anomalies_df["sku_id"])
        
        sku_detail = anomalies_df[anomalies_df["sku_id"] == selected_sku].iloc[0]
        
        col_d1, col_d2 = st.columns([1, 1])
        
        with col_d1:
            st.markdown(f"#### SKU {selected_sku} Peer Cost Comparison")
            
            # Fetch peer group details
            skus_df = load_skus()
            peer_group = skus_df[skus_df["style_group"] == sku_detail["style_group"]]
            
            # Create distribution chart
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Bar(
                x=peer_group["sku_id"],
                y=peer_group["unit_cost"],
                name="Comparable SKU Unit Cost",
                marker_color="#8C8C96",
                opacity=0.6
            ))
            # Highlight selected SKU
            fig_hist.add_trace(go.Bar(
                x=[selected_sku],
                y=[sku_detail["unit_cost"]],
                name="Anomalous SKU",
                marker_color="#F9D3D8"
            ))
            # Add line for peer group median
            fig_hist.add_hline(
                y=sku_detail["median"], 
                line_dash="dash", 
                line_color="#E6C17A",
                annotation_text=f"Peer Median (${sku_detail['median']:.2f})",
                annotation_position="bottom right"
            )
            fig_hist.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='#F8F9FA',
                yaxis_title="Unit Cost ($)",
                xaxis_title="SKU list in Style Group",
                yaxis=dict(gridcolor='#1C1C1E'),
                height=350,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_hist, use_container_width=True)
            
        with col_d2:
            st.markdown("#### 💬 Buyer Negotiation Explanation")
            st.info(sku_detail["explanation"])
            st.markdown(
                f"**Procurement Negotiation Pitch**:\n\n"
                f"> *\"We noticed that the contracted unit cost of **{selected_sku}** is currently set to **${sku_detail['unit_cost']:.2f}**, "
                f"which is **{sku_detail['pct_deviation']:.1f}% higher** than the average contract price of **${sku_detail['median']:.2f}** "
                f"for comparable products in the **{sku_detail['style_group']}** family. As we are placing a large-volume order of **{int(sku_detail['order_volume'])} units**, "
                f"we expect supplier price alignment. Re-aligning this unit cost saves **${sku_detail['potential_savings']:,.2f}** this period alone, "
                f"strengthening our joint volume-commitment.\"*"
            )

# ----------------- TAB 4: WHAT-IF SENSITIVITY -----------------
with tab4:
    st.subheader("Constraint Sensitivity & Shadow Prices")
    st.markdown(
        "> [!NOTE]\n"
        "> Sensitivity analysis (marginal cost analysis) reveals how relaxing or tightening constraints impact total supply chain costs. "
        "In optimization, the **Shadow Price** is the marginal value of relaxing a constraint by 1 unit. "
        "Here, we resolve the model with $\pm 10\%$ changes to examine empirical shadow prices, and compare them with the "
        "analytical shadow prices derived from the LP relaxation dual variables."
    )
    
    @st.cache_data(ttl=60, show_spinner=False)
    def get_sensitivity_results(budget, capacity, service, num_scenarios):
        if mode == "API":
            payload = {
                "base_budget": budget,
                "base_capacity": capacity,
                "service_level": service,
                "num_scenarios": num_scenarios
            }
            res = requests.post(f"{API_URL}/scenario", json=payload)
            if res.status_code == 200:
                return res.json()
            return None
        else:
            skus_df = load_skus()
            return run_sensitivity_analysis(skus_df, budget, capacity, service, num_scenarios)
            
    with st.spinner("Calculating sensitivity matrices..."):
        sens = get_sensitivity_results(budget_slider, capacity_slider, service_slider, scenarios_slider)
        
    if sens is None or sens.get("status") == "Infeasible":
        st.error("Could not run sensitivity: constraints are too tight.")
    else:
        col_s1, col_s2 = st.columns(2)
        
        with col_s1:
            st.markdown("### Budget Sensitivity (MIP Resolved)")
            st.write(sens["budget_explanation"])
            
            # Draw plot
            b_x = ["-10% Budget", "Base Budget", "+10% Budget"]
            b_y = [sens["budget_minus_10"], sens["base_objective"], sens["budget_plus_10"]]
            
            # Remove None values
            b_x_f = [x for x, y in zip(b_x, b_y) if y is not None]
            b_y_f = [y for y in b_y if y is not None]
            
            fig_b = px.line(
                x=b_x_f, 
                y=b_y_f, 
                markers=True,
                height=300
            )
            fig_b.update_traces(line_color="#F9D3D8", marker=dict(size=8, color="#E6C17A"))
            fig_b.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='#F8F9FA',
                xaxis_title="",
                yaxis_title="Total Supply Chain Cost ($)",
                yaxis=dict(gridcolor='#1C1C1E')
            )
            st.plotly_chart(fig_b, use_container_width=True)
            
        with col_s2:
            st.markdown("### Warehouse Space Sensitivity (MIP Resolved)")
            st.write(sens["capacity_explanation"])
            
            # Draw plot
            c_x = ["-10% Capacity", "Base Capacity", "+10% Capacity"]
            c_y = [sens["capacity_minus_10"], sens["base_objective"], sens["capacity_plus_10"]]
            
            # Remove None values
            c_x_f = [x for x, y in zip(c_x, c_y) if y is not None]
            c_y_f = [y for y in c_y if y is not None]
            
            fig_c = px.line(
                x=c_x_f, 
                y=c_y_f, 
                markers=True,
                height=300
            )
            fig_c.update_traces(line_color="#E6C17A", marker=dict(size=8, color="#F9D3D8"))
            fig_c.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='#F8F9FA',
                xaxis_title="",
                yaxis_title="Total Supply Chain Cost ($)",
                yaxis=dict(gridcolor='#1C1C1E')
            )
            st.plotly_chart(fig_c, use_container_width=True)
            
        # Summary details table comparing dual and resolved shadow prices
        st.markdown("### 🧠 Shadow Price Comparison")
        st.markdown(
            "Here we compare the **Empirical Shadow Price** (the cost delta by physically resolving the MIP) "
            "with the **LP Relaxation Analytical Shadow Price** (the mathematical dual variable from relaxing binary variables). "
            "LP relaxation shadow prices are extremely fast to compute and offer excellent directional guidance, but they assume continuous order decisions (no MOQs)."
        )
        
        shadow_df = pd.DataFrame({
            "Constraint": ["Procurement Budget ($)", "Warehouse Capacity (m³)"],
            "Empirical Shadow Price (MIP)": [
                f"${abs(sens['empirical_budget_shadow_price']):.4f} saved per $1 spent",
                f"${abs(sens['empirical_capacity_shadow_price']):.4f} saved per 1 m³ added"
            ],
            "Analytical Shadow Price (LP Dual)": [
                f"${abs(sens['analytical_budget_shadow_price']):.4f} saved per $1 spent",
                f"${abs(sens['analytical_capacity_shadow_price']):.4f} saved per 1 m³ added"
            ]
        })
        st.table(shadow_df)

# Footer
st.markdown(
    "<div class='footer'>"
    "Retail Inventory Planning & Optimization Dashboard. Developed for DS Technical Interview. "
    "All data and outcomes are simulated on synthetic SKU profiles."
    "</div>",
    unsafe_allow_html=True
)
