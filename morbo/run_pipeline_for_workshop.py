import sys
import os
import time
import argparse
import pandas as pd
import torch

# Ensure paths are set correctly for your environment
sys.path.append("/home/cato/cato_nafi/morbo") 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from helper import consts2
from helper import utils
from helper import prior_injection

# Import the core runners from your existing scripts
# Ensure these files are in the same directory and accessible
from cato_optimize import hm_run
from run_cato_with_merged import morbo_casmo_run 

# Filter out warnings for a clean workshop output
import warnings
warnings.filterwarnings("ignore")

def execute_combined_pipeline(args):
    candidate_features = consts2.candidate_features
    mi = prior_injection.compute_mi_scores(candidate_features, pkt_depth="all")
    candidate_features = [k for k, v in mi.items() if v > 0]

    for trial in range(args.num_trials):
        print(f"\n{'='*50}")
        print(f"=== STARTING WORKSHOP PIPELINE - TRIAL {trial+1} ===")
        print(f"{'='*50}")
        
        for max_pkt_depth in args.max_pkt_depth.split(","):
            max_pkt = int(max_pkt_depth)
            
            # --- PHASE 1: BASELINE CATO ---
            print("\n[PHASE 1] Running Baseline CATO (HyperMapper) Initialization...")
            
            # We run hm_run for the initial budget (e.g., 60 total evaluations)
            cato_initial_budget = 10 
            cato_bo_iters = args.cato_iters - cato_initial_budget
            
            # Run CATO and get the output directory where the CSV is saved
            # Note: You may need to modify hm_run in cato_optimize.py slightly to return the output_dir 
            # alongside the scenario_file, or we can reconstruct the path dynamically here.
            scenario_file = hm_run(
                candidate_features=candidate_features, 
                max_pkt_depth=max_pkt, 
                num_init=cato_initial_budget, 
                num_iter=cato_bo_iters, 
                include_priors=args.priors, 
                damping_factor=args.damping_factor, 
                experiment_dir=args.experiment_dir
            )
            
            # Reconstruct the exact CSV path HyperMapper just generated
            cato_output_dir = os.path.dirname(scenario_file)
            cato_csv_path = os.path.join(cato_output_dir, "post_output_samples.csv")
            
            if not os.path.exists(cato_csv_path):
                raise FileNotFoundError(f"Failed to locate CATO output CSV at {cato_csv_path}")
                
            print(f"[INFO] CATO Phase complete. Data saved to: {cato_csv_path}")

            # --- PHASE 2: MORBO+CASMOPOLITAN ---
            print("\n[PHASE 2] Handing off to MORBO+Casmopolitan Framework...")
            
            # We pass the CATO generated CSV directly into our warm-start runner
            morbo_casmo_run(
                candidate_features=candidate_features, 
                max_pkt_depth=max_pkt, 
                num_init=10,                  # <--- ADD THIS LINE
                csv_init_path=cato_csv_path,  # Automated handoff!
                num_iter=args.morbo_iters,    # Remaining budget
                include_priors=args.priors,
                damping_factor=float(args.damping_factor), 
                experiment_dir=args.experiment_dir,
                use_log_warp=False, 
                use_unified_kernel=False, 
                use_mixture_kernel=True, 
                use_casmo_mixed_kernel=False,
                # csv_init_path=None
            )
            
            print(f"\n=== PIPELINE COMPLETE FOR MAX DEPTH {max_pkt} ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified CATO to MORBO Pipeline")
    parser.add_argument("max_pkt_depth", type=str, help="Comma separated list of Maximum packet depth")
    parser.add_argument("cato_iters", type=int, help="Number of initial baseline CATO iterations to run")
    parser.add_argument("morbo_iters", type=int, help="Number of MORBO+Casmopolitan iterations to run after handoff")
    parser.add_argument("damping_factor", type=str, help="Dampen feature priors")
    parser.add_argument("experiment_dir", type=str, help="Path to experiment output dir")
    parser.add_argument("--num_trials", type=int, default=1, help="Number of trials")
    parser.add_argument("--priors", action="store_true", help="Include priors")
    parser.set_defaults(priors=True)
    
    args = parser.parse_args()
    execute_combined_pipeline(args)