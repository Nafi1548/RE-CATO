import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from botorch.utils.multi_objective.box_decompositions.dominated import DominatedPartitioning
from botorch.utils.multi_objective.pareto import is_non_dominated

# ==========================================
# Configuration
# ==========================================
BASE_DIR = "./dataset_comparision"  # Use "." if running directly in the folder
HM_OBJ_COLUMNS = ['neg_f1_score', 'compute_cost']

def compute_hv_trajectory(Y_norm: torch.Tensor, ref_point: torch.Tensor):
    """Computes the cumulative hypervolume for each iteration in [0, 1] space."""
    hvs = []
    for i in range(1, len(Y_norm) + 1):
        Y_step = Y_norm[:i]
        # Only keep points strictly better than the reference point
        better_than_ref = (Y_step > ref_point).all(dim=-1)
        Y_feasible = Y_step[better_than_ref]
        
        if len(Y_feasible) == 0:
            hvs.append(0.0)
        else:
            partitioning = DominatedPartitioning(ref_point=ref_point, Y=Y_feasible)
            hv = partitioning.compute_hypervolume().item()
            hvs.append(hv)
    return hvs

def get_pareto_front(Y_min: torch.Tensor):
    """Extracts the non-dominated Pareto front points."""
    # BoTorch assumes maximization for is_non_dominated
    mask = is_non_dominated(-Y_min)
    return Y_min[mask]

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
    
    # 2. Fix the Sign Mismatch
    Y_hm_min = Y_hm_raw.clone()
    Y_ours_min = Y_ours_raw.clone()
    
    if Y_ours_raw[:, 1].mean() < 0 and Y_hm_raw[:, 1].mean() > 0:
        print("  -> Detected MORBO internal maximization. Reverting to raw minimization values.")
        Y_ours_min = Y_ours_min * -1.0
        
    # 3. Normalize Data using ONLY the Pareto Fronts
    # Extract only the best points to define our bounding box
    pf_ours = get_pareto_front(Y_ours_min)
    pf_hm = get_pareto_front(Y_hm_min)
    
    Y_pf_all = torch.cat([pf_ours, pf_hm], dim=0)
    
    nadir = Y_pf_all.max(dim=0).values  # Worst values ON the Pareto front
    ideal = Y_pf_all.min(dim=0).values  # Best values ON the Pareto front
    
    # Add a tiny 1% margin to nadir so the absolute worst frontier points don't get 0.0 volume
    margin = (nadir - ideal) * 0.01
    nadir = nadir + margin
    
    print(f"  -> Realistic Best (Ideal): {ideal.tolist()}")
    print(f"  -> Realistic Worst (Nadir): {nadir.tolist()}")
    
    # Transform: (Worst - Current) / (Worst - Best)
    Y_ours_norm = (nadir - Y_ours_min) / (nadir - ideal)
    Y_hm_norm = (nadir - Y_hm_min) / (nadir - ideal)
    
    # 4. Compute Hypervolume 
    # Because of the margin above, points slightly worse than the Pareto front 
    # will be negative and properly ignored by the ref_point cut-off.
    ref_point = torch.tensor([0.0, 0.0])
    
    hv_ours = compute_hv_trajectory(Y_ours_norm, ref_point)
    hv_hm = compute_hv_trajectory(Y_hm_norm, ref_point)
    
    # 5. Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(hv_ours) + 1), hv_ours, label='MORBO+Casmopolitan', linewidth=2, color='blue')
    plt.plot(range(1, len(hv_hm) + 1), hv_hm, label='Hypermapper', linewidth=2, color='orange')
    
    plt.xlabel('Number of Evaluations (Samples)')
    plt.ylabel('Normalized Hypervolume (Relative to Pareto Front)')
    plt.title('Realistic Hypervolume Improvement')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plot_path = os.path.join(folder_path, "hv_comparison_realistic.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  -> Successfully saved plot to {plot_path}")

if __name__ == "__main__":
    if BASE_DIR == ".":
        analyze_folder(".")
    else:
        folders = [f.path for f in os.scandir(BASE_DIR) if f.is_dir()]
        for folder in folders:
            analyze_folder(folder)