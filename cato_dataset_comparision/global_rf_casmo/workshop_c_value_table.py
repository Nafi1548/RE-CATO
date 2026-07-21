import pandas as pd
import numpy as np

def get_pareto_front(costs):
    """
    Identifies the Pareto frontier for a given set of evaluations.
    Assumes all objectives are being MINIMIZED.
    """
    # Drop duplicates to prevent identical values from skewing metrics
    costs = np.unique(costs, axis=0) 
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    
    for i, c in enumerate(costs):
        if is_efficient[i]:
            # Keep any point with a lower cost
            is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1)
            is_efficient[i] = True
            
    return costs[is_efficient]

def c_metric(front_a, front_b):
    """
    Calculates C(A, B): the fraction of points in front_B 
    weakly dominated by at least one point in front_A.
    """
    if len(front_b) == 0:
        return 0.0
        
    dominated_count = 0
    for b in front_b:
        # Check if ANY 'a' in front_a weakly dominates 'b' (a <= b for all objectives)
        if np.any(np.all(front_a <= b, axis=1)):
            dominated_count += 1
            
    return dominated_count / len(front_b)

def evaluate_pairs(pairs, objective_cols):
    results = []
    
    for pair in pairs:
        print(f"Evaluating: {pair['name']}")
        
        # Load datasets
        df_base = pd.read_csv(pair['baseline'])[0:300]
        df_app = pd.read_csv(pair['approach'])[0:300]
        
        # Extract objectives as numpy arrays
        costs_base = df_base[objective_cols].dropna().values
        costs_app = df_app[objective_cols].dropna().values
        
        # Get Pareto Fronts
        front_base = get_pareto_front(costs_base)
        front_app = get_pareto_front(costs_app)
        
        # Calculate C-Metrics
        c_base_app = c_metric(front_base, front_app)
        c_app_base = c_metric(front_app, front_base)
        
        print(f"  C(Baseline, Approach): {c_base_app:.4f}")
        print(f"  C(Approach, Baseline): {c_app_base:.4f}\n")

if __name__ == "__main__":
    # Define the 3 pairs mapping baselines to their respective approaches
    evaluation_pairs = [
        {
            "name": "Rubayet: Baseline vs RF Mean GP Var Approach 1",
            "baseline": "rubayet_baseline.csv",
            "approach": "rubayet_global_gp_var.csv"
        },
        {
            "name": "Server: Baseline vs Global RF Local GP Approach 2",
            "baseline": "server_baseline.csv",
            "approach": "global_rf_local_gp_approach2.csv"
        },
        {
            "name": "Server: Baseline vs Hybrid Global RF Mixed",
            "baseline": "server_baseline.csv",
            "approach": "hybrid_global_rf_mixed.csv"
        }
    ]
    
    # Specify the target objectives to minimize
    target_objectives = ['neg_f1_score', 'compute_cost'] 
    
    evaluate_pairs(evaluation_pairs, target_objectives)