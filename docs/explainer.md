# Inventory Cost Optimization & Negotiation flagging: Technical Explainer

This document serves as an internal guide and interview prep explainer for the Inventory Cost Optimization system. It explains the core mathematical optimization, pricing anomaly models, and how this design scales to real-world enterprise production systems.

---

## 1. What is Linear Programming (LP) and Mixed-Integer Programming (MIP)?

At a high level, mathematical optimization is the process of finding the "best" decision from a set of feasible options. We define three things:
1. **Decision Variables**: The choices we need to make (e.g., *"How many units of SKU-A should we order?"*).
2. **Objective Function**: The business goal we want to maximize or minimize (e.g., *"Minimize total holding + stockout costs"*).
3. **Constraints**: The boundaries we must operate within (e.g., *"Don't exceed our $80,000 procurement budget"* and *"Ensure the storage volume of our orders doesn't exceed our 150 m³ warehouse capacity"*).

### Linear Programming (LP)
In a pure LP, all decision variables are **continuous** (can be any decimal number like $120.45$). The relationship between variables in the objective and constraints must be strictly **linear** (no multiplication of variables, no exponents). LPs are mathematically simple and can be solved extremely quickly (millions of variables in seconds) using algorithms like the Simplex or Barrier methods.

### Mixed-Integer Programming (MIP)
In retail, many decisions are discrete:
- **Minimum Order Quantities (MOQ)**: You cannot order 1.5 units; you must either order 0 or at least the supplier's MOQ (e.g., 100 units).
- **Setup Costs / Fixed Fees**: If you place an order, you pay a flat shipping fee; if you don't order, you pay nothing.

To model this, we introduce **binary variables** ($y_i \in \{0, 1\}$) that act as on/off switches:
- $y_i = 1$ if we place an order for SKU $i$.
- $y_i = 0$ if we do not.

This transforms the model into a **Mixed-Integer Program (MIP)**, which combines continuous decision variables (order quantities $x_i$) with integer/binary variables ($y_i$). MIPs are mathematically NP-hard and solved using **Branch-and-Bound** and **Branch-and-Cut** algorithms, which systematically search a tree of LP relaxations.

---

## 2. Why Choose PuLP & Fallback to SciPy?

For this implementation, we selected **PuLP** as the primary framework and provided a **SciPy** fallback:

- **PuLP**:
  - **Strengths**: A highly expressive, algebraic modeling language in Python. It lets us write equations like `prob += x[i] >= moq * y[i]` directly in code. It separates model formulation from solver execution, meaning we can swap out the backend solver (CBC, Gurobi, CPLEX, COPT) with a single line of code.
  - **Production Ready**: Ships with a pre-compiled, robust open-source solver (CBC) that handles MIPs out of the box, making it perfect for containerized deployment (no external binary setup required).
  
- **SciPy (`scipy.optimize.linprog`)**:
  - **LP Relaxation fallback**: SciPy is a staple of the scientific python stack. We use it to solve the **LP relaxation** (treating binary variables as continuous fractions in $[0, 1]$).
  - **Dual Values / Shadow Prices**: Solving the LP relaxation allows us to extract the **dual variables** (also known as shadow prices). The shadow price of a constraint tells us the marginal savings we would gain by relaxing that constraint by one unit (e.g., *\"What is the value of adding $1 of budget?\"*). This is extremely useful for strategic "what-if" planning.

---

## 3. Cost Anomaly Detection: Bridging Data Science and Procurement

In a large retail catalog with thousands of active styles, colors, and fabrics, buyers negotiate contract prices with multiple suppliers. Due to fragmented vendor negotiations, data-entry errors, or legacy contracts, comparable items often have different prices.

### How the Anomaly Detector Works:
1. **Comparable Peer Grouping**: We group SKUs by `style_group` (material + category combination, e.g., *"Leather Activewear"*, *"Wool Loungewear"*). This ensures we compare "apples to apples" (material type is the primary cost driver in apparel).
2. **Robust Statistics (Median & IQR)**: We compute the peer group median and Interquartile Range (IQR). We avoid mean and standard deviation because they are heavily influenced by the outliers we are trying to find.
3. **The Threshold (Tukey's Fence)**:
   $$\text{Threshold} = \text{Median} + 1.5 \times \text{IQR}$$
   If a SKU's cost exceeds this threshold, it is flagged. If the group has very low variance (e.g., IQR is nearly 0), we apply a fallback threshold of $15\%$ above the median.
4. **Business Impact Prioritization**: A statistical outlier is only a business priority if it represents significant spend. We rank anomalies by:
   $$\text{Negotiation Savings} = (\text{Unit Cost} - \text{Peer Median}) \times \text{Order Volume}$$
   By linking the saving potential to the **recommended order quantity** from our optimization engine, we immediately identify the highest-value contract renegotiation opportunities.

---

## 4. Scaling to Production: What Changes with Real Data?

If we were to deploy this system in a real-world enterprise environment, we would expand the synthetic model to address real-world operational complexities:

### A. Stochastic Programming & Demand Uncertainty
Our SAA (Sample Average Approximation) model uses 15–50 scenarios to approximate weekly demand. In a full production system:
- We would feed the optimizer a full probability distribution of demand generated directly by a downstream SKU-level demand forecasting model (e.g., DeepAR or Temporal Fusion Transformers).
- Instead of static scenarios, we would use **stochastic programming** or **robust optimization** to shield against extreme stockout penalties on high-margin items.

### B. Lead-Time Uncertainty & Multi-Period Rolling Horizons
Apparel supply chains are subject to global shipping delays:
- **Lead-time is stochastic**: The model must optimize safety stock based on both demand variance and lead-time variance.
- **Multi-period Planning**: A weekly single-period model is expanded to a multi-period rolling horizon (e.g., 12 weeks), modeling orders, inventory carryover, and seasonal trends across periods:
  $$Inventory_{t} = Inventory_{t-1} + Order_{t-LeadTime} - Demand_{t}$$

### C. Joint Replenishment & Multi-Echelon Constraints
Inventory doesn't sit in a vacuum; it flows through a network:
- **Multi-Echelon Optimization**: We must coordinate inventory between Central Distribution Centers (DCs) and individual retail stores (e.g., routing stock to stores with the highest fill rate return).
- **Joint Replenishment**: Ordering multiple SKUs from the same supplier simultaneously triggers shipping discounts or container-load optimization.

### D. Advanced Optimization Solvers
For a 100,000+ SKU catalog across a 12-week horizon under multi-echelon constraints, open-source solvers like CBC will encounter performance ceilings. Production deployments typically transition to commercial solvers:
- **Gurobi** or **CPLEX**: Solve massive MIPs up to 100x faster than CBC using advanced presolve heuristics, cutting planes, and parallelized branch-and-bound.
- **Decomposition Techniques**: Using Benders Decomposition or Column Generation to split the master optimization problem into smaller, parallelizable sub-problems.
