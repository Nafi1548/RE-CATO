import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from botorch.utils.multi_objective.pareto import is_non_dominated

# ==========================================
# Configuration
# ==========================================
BASE_DIR = "./dataset_comparision"  # Use "." if running directly in the folder
HM_OBJ_COLUMNS = ['neg_f1_score', 'compute_cost']

def get_pareto_points(Y_min: torch.Tensor):
    """
    Extracts the non-dominated Pareto front points.
    Y_min: Tensor where we assume MINIMIZATION for all objectives.
    """
    # BoTorch's is_non_dominated assumes MAXIMIZATION.
    # We multiply by -1 to flip our minimization space into maximization for the check.
    Y_max = -Y_min
    pareto_mask = is_non_dominated(Y_max)
    pareto_front = Y_min[pareto_mask]
    
    # Sort by the first objective (neg_f1_score) so the plot line draws neatly left-to-right
    sorted_indices = torch.argsort(pareto_front[:, 0])
    return pareto_front[sorted_indices]

def analyze_folder(folder_path):
    print(f"\nAnalyzing folder: {os.path.abspath(folder_path)}")
    
    pt_path = os.path.join(folder_path, "bo_results.pt")
    csv_path = os.path.join(folder_path, "post_output_samples.csv")
    
    if not os.path.exists(pt_path) or not os.path.exists(csv_path):
        print(f"  Missing files. Skipping.")
        return
    
    # 1. Load Data
    data_pt = torch.load(pt_path, map_location=torch.device('cpu'))
    Y_ours_raw = data_pt['Y_history'].float()
    
    df_hm = pd.read_csv(csv_path)
    Y_hm_raw = torch.tensor(df_hm[HM_OBJ_COLUMNS].values, dtype=torch.float32)
    
    # 2. Fix the Sign Mismatch (Revert to Minimization space)
    Y_hm_min = Y_hm_raw.clone()
    Y_ours_min = Y_ours_raw.clone()
    
    if Y_ours_raw[:, 1].mean() < 0 and Y_hm_raw[:, 1].mean() > 0:
        print("  -> Detected MORBO internal maximization. Reverting to raw minimization values.")
        Y_ours_min = Y_ours_min * -1.0
        
    # 3. Extract Pareto Frontiers
    pareto_ours = get_pareto_points(Y_ours_min)
    pareto_hm = get_pareto_points(Y_hm_min)
    
    # 4. Plotting
    plt.figure(figsize=(10, 6))
    
    # Plot ONLY the actual Pareto frontiers (no background scatter)
    plt.plot(pareto_ours[:, 0].numpy(), pareto_ours[:, 1].numpy(), marker='o', markersize=6, color='blue', linewidth=2, label='MORBO+Casmopolitan')
    plt.plot(pareto_hm[:, 0].numpy(), pareto_hm[:, 1].numpy(), marker='s', markersize=6, color='orange', linewidth=2, label='Hypermapper')
    
    # Add labels and formatting
    plt.xlabel('Negative F1 Score (Lower is Better)')
    plt.ylabel('Compute Cost (Lower is Better)')
    plt.title('Pareto Frontier Comparison')
    
    # Optional: If the plot is STILL stretched by one absurdly high cost point on the frontier, 
    # uncomment the line below to hard-cap the Y-axis to a reasonable number (e.g., 5000)
    # plt.ylim(bottom=0, top=5000) 

    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plot_path = os.path.join(folder_path, "pareto_comparison_clean.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  -> Successfully saved clean Pareto plot to {plot_path}")
    
if __name__ == "__main__":
    if BASE_DIR == ".":
        analyze_folder(".")
    else:
        folders = [f.path for f in os.scandir(BASE_DIR) if f.is_dir()]
        for folder in folders:
            analyze_folder(folder)