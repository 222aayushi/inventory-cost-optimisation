import uuid
import pandas as pd
from datetime import datetime
from src.database import save_optimization_run, get_runs_history, get_run_recommendations

class OptimizationTracker:
    """
    Lightweight MLOps tracker for logging optimization run configurations,
    metadata, objectives, runtimes, and resulting SKU recommendations.
    """
    
    @staticmethod
    def log_run(
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
    ) -> str:
        """
        Logs a run to DuckDB database and returns the generated run_id.
        """
        run_id = f"RUN-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{str(uuid.uuid4())[:8]}"
        
        save_optimization_run(
            run_id=run_id,
            solver_type=solver_type,
            solver_status=solver_status,
            budget_cap=budget_cap,
            capacity_cap=capacity_cap,
            service_level_target=service_level_target,
            total_holding_cost=total_holding_cost,
            total_stockout_penalty=total_stockout_penalty,
            total_cost=total_cost,
            savings_vs_base=savings_vs_base,
            run_time_seconds=run_time_seconds,
            recommendations_df=recommendations_df
        )
        return run_id

    @staticmethod
    def get_history() -> pd.DataFrame:
        """
        Returns a DataFrame of all past logged optimization runs.
        """
        return get_runs_history()

    @staticmethod
    def get_run_details(run_id: str) -> pd.DataFrame:
        """
        Returns a DataFrame of SKU recommended quantities for a specific run.
        """
        return get_run_recommendations(run_id)
