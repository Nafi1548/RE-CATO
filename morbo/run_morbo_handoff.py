import sys
import os
import argparse
import warnings

# Filter out warnings for a clean output
warnings.filterwarnings("ignore")

# Ensure paths are set correctly for your environment
sys.path.append("/home/cato/cato_nafi/morbo") 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from helper import consts
from helper import prior_injection

# Import ONLY the MORBO runner
from run_cato_with_merged import morbo_casmo_run 

def execute_csv_pipeline(args):
    # 1. Verify the CSV exists before doing anything
    if not os.path.exists(args.csv_path):
        raise FileNotFoundError(f"Could not find the provided CSV at: {args.csv_path}")

    # 2. Feature Setup & Dimensonality Reduction
    candidate_features = consts.candidate_features
    mi = prior_injection.compute_mi_scores(candidate_features, pkt_depth="all")
    candidate_features = [k for k, v in mi.items() if v > 0]

    for trial in range(args.num_trials):
        print(f"\n{'='*60}")
        print(f"=== STARTING MORBO FROM CSV - TRIAL {trial+1} ===")
        print(f"{'='*60}")
        
        for max_pkt_depth in args.max_pkt_depth.split(","):
            max_pkt = int(max_pkt_depth.strip())
            
            print(f"\n[INFO] Loading initial data from: {args.csv_path}")
            print(f"[INFO] Handing off to MORBO+Casmopolitan for depth {max_pkt}...")
            
            # Run MORBO directly using the existing CSV
            morbo_casmo_run(
                candidate_features=candidate_features, 
                max_pkt_depth=max_pkt, 
                num_init=args.num_init,       # The number of samples in your CSV (e.g., 60)
                num_iter=args.morbo_iters,    # The additional BO budget you want MORBO to run
                include_priors=args.priors,
                damping_factor=float(args.damping_factor), 
                experiment_dir=args.experiment_dir,
                use_log_warp=False,            # Ensure this is True for huge bounds
                use_unified_kernel=False, 
                use_mixture_kernel=True, 
                use_casmo_mixed_kernel=True,
                csv_init_path=args.csv_path   # Inject the existing CSV!
            )
            
            print(f"\n=== MORBO PIPELINE COMPLETE FOR MAX DEPTH {max_pkt} ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MORBO directly from a pre-existing CATO CSV")
    parser.add_argument("csv_path", type=str, help="Absolute or relative path to the existing HyperMapper CSV")
    parser.add_argument("max_pkt_depth", type=str, help="Comma separated list of Maximum packet depths")
    parser.add_argument("num_init", type=int, help="The number of initial samples contained in the CSV (e.g., 60)")
    parser.add_argument("morbo_iters", type=int, help="Number of ADDITIONAL BO iterations for MORBO to run")
    parser.add_argument("damping_factor", type=str, help="Dampen feature priors")
    parser.add_argument("experiment_dir", type=str, help="Path to experiment output dir")
    
    parser.add_argument("--num_trials", type=int, default=1, help="Number of trials")
    parser.add_argument("--priors", action="store_true", help="Include priors")
    parser.set_defaults(priors=True)
    
    args = parser.parse_args()
    execute_csv_pipeline(args)