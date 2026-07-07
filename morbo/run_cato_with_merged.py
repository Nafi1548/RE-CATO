import sys
import os

# 1. Point to the root directory containing the 'morbo' folder
# Update this absolute path to wherever your CATO root actually is
sys.path.append("/home/cato/cato_nafi/morbo") 

# 2. Point to your helper/measure directory (as it was in your original code)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 3. Now Python can find the morbo module
from morbo.state import TRBOState
from morbo.trust_region import TurboHParams
# from morbo.gen import TS_select_batch_MORBO
from morbo.discrete_ts_batch import discrete_ts_batch_select
from botorch.utils.sampling import draw_sobol_samples
import shutil
import warnings
import time
import datetime
import numpy as np
import argparse
import torch

from pprint import pprint

# Assuming your merged framework is accessible via these imports
from morbo.state import TRBOState
from morbo.trust_region import TurboHParams
# from morbo.gen import TS_select_batch_MORBO
from botorch.utils.sampling import draw_sobol_samples
from morbo.discrete_ts_batch import discrete_ts_batch_select

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from helper import consts2
from helper import utils
from helper import prior_injection
from measure import measure_compute
from measure import measure_inference

# Filter out warnings
warnings.filterwarnings("ignore")

candidate_features = consts2.candidate_features

# dimensionality reduction
mi = prior_injection.compute_mi_scores(candidate_features, pkt_depth="all")
candidate_features = [k for k,v in mi.items() if v > 0]

def evaluate_cato_batch(X_tensor):
    """
    Evaluates a batch of candidates.
    Maps the mixed-space tensor back to the CATO domain hparameters.
    MORBO/BoTorch assumes MAXIMIZATION by default. 
    We maximize F1 Score and maximize Negative Compute Cost.
    """
    Y = []
    for x in X_tensor:
        # 1. Decode categorical features (first N dimensions)
        feature_set = []
        for i, feat in enumerate(candidate_features):
            # Casmopolitan keeps these as ordinal/integer values internally
            if int(round(x[i].item())) == 1:
                feature_set.append(feat)
                
        # 2. Decode continuous/integer feature (last dimension)
        # pkt_depth = int(round(x[-1].item()))
        # AFTER — shifts internal {0..max_pkt_depth-1} to real {1..max_pkt_depth}
        pkt_depth = int(round(x[-1].item())) + 1
        print(utils.CYAN, feature_set, pkt_depth, utils.RESET)
        import math
        # # 3. Evaluate
        # y_f1 = measure_inference.get_f1_score(feature_set, pkt_depth)  
        # y_compute = measure_compute.get_compute_cost(feature_set, pkt_depth)
        
        # Evaluate physically
        y_f1 = measure_inference.get_f1_score(feature_set, pkt_depth)  
        raw_compute = measure_compute.get_compute_cost(feature_set, pkt_depth)
        
        # Log-transform the compute cost before negating it for maximization
        # This perfectly aligns the target surface with the log-warped kernel
        y_compute = math.log1p(raw_compute)

        # 4. Cleanup: Delete the generated features_* directory
        feature_decimal = utils.feature_decimal(feature_set)
        dataset_dir = os.path.join(consts2.dataset_dir, f"pkts_{pkt_depth}")
        model_dir = os.path.join(dataset_dir, f'features_{feature_decimal}')

        if os.path.exists(model_dir):
            try:
                shutil.rmtree(model_dir)
                print(utils.YELLOW + f"Deleted directory: {model_dir}" + utils.RESET)
            except OSError as e:
                print(utils.RED + f"Error deleting directory {model_dir}: {e}" + utils.RESET)

        # Append [Objective 1 (Maximize), Objective 2 (Maximize)]
        Y.append([y_f1, -y_compute]) 
        
    return torch.tensor(Y, dtype=X_tensor.dtype, device=X_tensor.device)

def morbo_casmo_run(candidate_features, max_pkt_depth, num_init, num_iter, include_priors, damping_factor, experiment_dir="",use_log_warp=False, use_unified_kernel = True, use_mixture_kernel = False, use_casmo_mixed_kernel = False):
    """
    Run the custom morbo+casmopolitan optimization and save matching CATO's structure.
    """
    # --- DIRECTORY SETUP (Matching Original CATO) ---
    candidate_decimal = utils.feature_decimal(candidate_features)
    # Using 'merged_' prefix to distinguish from original 'hmp_' (HyperMapper) runs
    output_dir = os.path.join(consts2.results_dir, f"merged_{candidate_decimal}")
    os.makedirs(output_dir, exist_ok=True)
    
    if not include_priors:
        exp_dir = experiment_dir + "_np"
    else:
        exp_dir = experiment_dir
        
    merged_dir = os.path.join(output_dir, exp_dir)
    os.makedirs(merged_dir, exist_ok=True)

    # create new timestamped output directory
    dt = datetime.datetime.fromtimestamp(time.time())
    ts = dt.strftime('%Y-%m-%d-%H-%M-%S')
    run_output_dir = os.path.join(merged_dir, f"max{max_pkt_depth}_init{num_init}_iter{num_iter}_damp{damping_factor}_{ts}")
    os.makedirs(run_output_dir, exist_ok=True)
    # ------------------------------------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.double
    
    # # --- 1. Define the Mixed Search Space ---
    # num_categorical = len(candidate_features)
    # dim = num_categorical + 1 # +1 for pkt_depth
    
    # cat_dims = list(range(num_categorical))
    # cont_dims = [num_categorical]
    # config = [2] * num_categorical # 2 states for each feature (0 or 1)
    
    # bounds = torch.zeros(2, dim, dtype=dtype, device=device)
    # bounds[1, cat_dims] = 1.0 
    # bounds[0, cont_dims[0]] = 1.0
    # bounds[1, cont_dims[0]] = float(max_pkt_depth)

    # --- 1. Define the Mixed Search Space (FIXED TO PURE DISCRETE) ---
    num_categorical = len(candidate_features)
    dim = num_categorical + 1 # +1 for pkt_depth
    if use_casmo_mixed_kernel:
        # THE ORIGINAL CASMOPOLITAN PATH
        cat_dims = list(range(num_categorical))
        cont_dims = [num_categorical]
        binary_dims = list(range(num_categorical))
        ordinal_dims = []
        ordinal_config = []
        config = [2] * num_categorical + [0] # 0 denotes a continuous dimension
        
        bounds = torch.zeros(2, dim, dtype=dtype, device=device)
        bounds[1, :num_categorical] = 1.0
        bounds[0, num_categorical] = 0.0
        bounds[1, num_categorical] = float(max_pkt_depth - 1)
    else:
        # THE PURE DISCRETE PATH (Your custom architectures)
        cat_dims = list(range(dim)) 
        cont_dims = [] 
        binary_dims = list(range(num_categorical))
        ordinal_dims = [num_categorical]
        ordinal_config = [max_pkt_depth]
        config = [2] * num_categorical + [max_pkt_depth]
        
        bounds = torch.zeros(2, dim, dtype=dtype, device=device)
        bounds[1, :num_categorical] = 1.0
        bounds[0, num_categorical] = 0.0                        
        bounds[1, num_categorical] = float(max_pkt_depth - 1)

    # max_ref_point = [0.0, -1000.0] 
    max_ref_point = [0.0, -20.0] 
    

    tr_hparams = TurboHParams(
        restart_hv_scalarizations=True,
        n_initial_points=num_init,
        batch_size=1, 
        cat_dims=cat_dims,
        cont_dims=cont_dims,
        config=config,
        max_reference_point=max_ref_point,
        hypervolume=True,
        use_ard=True,
        n_trust_regions=2,
        # --- Add these two lines! ---
        failure_streak=max(dim // 3, 10),
        success_streak=3,
        # ---------------------------- 
        # Memory constraints for Hugepages
        min_tr_size=max(2, num_init // 2),
        max_tr_size=500,               
        max_cholesky_size=1000,        
        raw_samples=1024,              
        pred_batch_limit=128,          
        length_init_discrete=max(1, num_categorical // 2),
        length_max_discrete=num_categorical,

        # # ADD THESE THREE LINES:
        # binary_dims=list(range(num_categorical)),
        # ordinal_dims=[num_categorical],
        # ordinal_config=[max_pkt_depth],
        # --- THE FIX: Pass the dynamically computed locals ---
        binary_dims=binary_dims,
        ordinal_dims=ordinal_dims,
        ordinal_config=ordinal_config,
        # Ensure your other ablation flags are also passed down from the runner args!
        use_log_warp=use_log_warp,
        use_unified_kernel=use_unified_kernel,
        use_mixture_kernel=use_mixture_kernel,
        use_casmo_mixed_kernel=use_casmo_mixed_kernel
    )

    trbo_state = TRBOState(
        dim=dim,
        num_outputs=2,
        num_objectives=2,
        bounds=bounds,
        max_evals=num_init + num_iter,
        tr_hparams=tr_hparams,
    ).to(device=device, dtype=dtype)

    # --- 3. Initial Design (DoE) ---
    print(f"Generating {num_init} initial samples...")
    # X_init = draw_sobol_samples(bounds=bounds, n=num_init, q=1).squeeze(1)
    # X_init[:, cat_dims] = torch.round(X_init[:, cat_dims])
    # X_init[:, cont_dims] = torch.round(X_init[:, cont_dims])
    # Y_init = evaluate_cato_batch(X_init)
    # 1. Generate standard linear Sobol samples [0, 1]

    import math as _math
    # Sobol bounds must use the same dtype as trbo_state (torch.double) to avoid
    # silent float32/float64 mismatches when X_init is passed to trbo_state.update().
    unit_bounds = torch.stack([
        torch.zeros(dim, dtype=dtype, device=device),
        torch.ones(dim, dtype=dtype, device=device),
    ])

    # raw_sobol = draw_sobol_samples(bounds=torch.stack([torch.zeros(dim), torch.ones(dim)]).to(device), n=num_init, q=1).squeeze(1)
    raw_sobol = draw_sobol_samples(bounds=unit_bounds, n=num_init, q=1).squeeze(1)
    
    # X_init = torch.zeros_like(raw_sobol)
    X_init = torch.zeros(num_init, dim, dtype=dtype, device=device)
    
    # 2. Map binary variables normally (threshold at 0.5)
    X_init[:, :num_categorical] = torch.round(raw_sobol[:, :num_categorical])
    
    # # 3. Log-warp the ordinal packet depth variable
    # # This transforms the [0, 1] draw into an exponentially biased draw towards 0
    # max_val = max_pkt_depth - 1
    # X_init[:, num_categorical] = torch.round(torch.exp(raw_sobol[:, num_categorical] * _math.log(max_val + 1)) - 1)

    # 2. STANDARD LINEAR MAPPING (CATO Baseline)
    # This replaces the log-warp so the initial points are uniformly distributed 
    # across the entire 0 to 349,999 range.
    max_val = max_pkt_depth - 1
    # X_init[:, num_categorical] = torch.round(raw_sobol[:, num_categorical] * float(max_val))
    
    # Ensure strict bounds
    # X_init[:, num_categorical] = torch.clamp(X_init[:, num_categorical], 0.0, float(max_val))

    # 2. Ordinal Packet Depth Toggle
    max_val = max_pkt_depth - 1
    if tr_hparams.use_log_warp:
        # Log-warped initialization (Biased towards 0)
        import math as _math
        X_init[:, num_categorical] = torch.round(
            torch.exp(raw_sobol[:, num_categorical] * _math.log(max_val + 1)) - 1
        )
    else:
        # Standard Linear Initialization (Uniform over 0 to 3.5 lacs)
        X_init[:, num_categorical] = torch.round(raw_sobol[:, num_categorical] * float(max_val))
    
    X_init[:, num_categorical] = torch.clamp(X_init[:, num_categorical], 0.0, float(max_val))
    Y_init = evaluate_cato_batch(X_init)
    
    trbo_state.update(X=X_init, Y=Y_init, new_ind=torch.full((X_init.shape[0],), 0, dtype=torch.long, device=device))
    trbo_state.log_restart_points(X=X_init, Y=Y_init)

    for i in range(tr_hparams.n_trust_regions):
        trbo_state.initialize_standard(tr_idx=i, restart=False, switch_strategy=False, X_init=X_init, Y_init=Y_init)

    trbo_state.update_data_across_trs()
    trbo_state.TR_index_history.fill_(-2)

    # Track start time for BO loop
    start_ts = time.time()

    # --- 4. Bayesian Optimization Loop ---
    print("Starting BO Loop...")
    while trbo_state.n_evals < trbo_state.max_evals:
        print(f"\n--- Iteration {trbo_state.n_evals}/{trbo_state.max_evals} ---")
        
        selection_output = discrete_ts_batch_select(trbo_state=trbo_state)
        X_cand = selection_output.X_cand
        tr_indices = selection_output.tr_indices
        
        X_cand[:, cat_dims] = torch.round(X_cand[:, cat_dims])
        X_cand[:, cont_dims] = torch.round(X_cand[:, cont_dims])

        trbo_state.tabu_set.log_iteration()
        Y_cand = evaluate_cato_batch(X_cand)

        trbo_state.update(X=X_cand, Y=Y_cand, new_ind=tr_indices)
        should_restart_trs = trbo_state.update_trust_regions_and_log(
            X_cand=X_cand, Y_cand=Y_cand, tr_indices=tr_indices, batch_size=tr_hparams.batch_size, verbose=True
        )

        switch_strategy = trbo_state.check_switch_strategy()
        if switch_strategy:
            should_restart_trs = [True for _ in should_restart_trs]
            
        if any(should_restart_trs):
            for i in range(tr_hparams.n_trust_regions):
                if should_restart_trs[i]:
                    print(f"Restarting trust region {i}")
                    trbo_state.TR_index_history[trbo_state.TR_index_history == i] = -1
                    init_kwargs = {}
                    if tr_hparams.restart_hv_scalarizations:
                        X_center = trbo_state.gen_new_restart_design()
                        X_center[:, cat_dims] = torch.round(X_center[:, cat_dims])
                        X_center[:, cont_dims] = torch.round(X_center[:, cont_dims])
                        Y_center = evaluate_cato_batch(X_center)
                        init_kwargs.update({"X_init": X_center, "Y_init": Y_center, "X_center": X_center})
                        trbo_state.update(X=X_center, Y=Y_center, new_ind=torch.tensor([i], dtype=torch.long, device=device))
                        trbo_state.log_restart_points(X=X_center, Y=Y_center)

                    trbo_state.initialize_standard(tr_idx=i, restart=True, switch_strategy=switch_strategy, **init_kwargs)
                    if tr_hparams.restart_hv_scalarizations:
                        trbo_state.update_data_across_trs()

        trbo_state.update_data_across_trs()
        
        if trbo_state.hv is not None:
            print(f"Current Hypervolume: {trbo_state.hv:.3f}")

    end_ts = time.time()
    elapsed = end_ts - start_ts
    print(f"Optimization Complete. BO elapsed: {elapsed:.2f}s")

    # Invert the -log1p transform for the second objective (cost)
    Y_hist_cpu = trbo_state.Y_history.cpu()
    Y_hist_raw = Y_hist_cpu.clone()
    Y_hist_raw[:, 1] = torch.expm1(-Y_hist_cpu[:, 1])  # exp(-y) - 1

    if trbo_state.pareto_Y is not None:
        pareto_Y_cpu = trbo_state.pareto_Y.cpu()
        pareto_Y_raw = pareto_Y_cpu.clone()
        pareto_Y_raw[:, 1] = torch.expm1(-pareto_Y_cpu[:, 1])
    else:
        pareto_Y_cpu = None
        pareto_Y_raw = None
    # --- SAVE RESULTS ---
    # output_data = {
    #     "X_history": trbo_state.X_history.cpu(),
    #     "Y_history": trbo_state.Y_history.cpu(),
    #     "pareto_X": trbo_state.pareto_X.cpu() if trbo_state.pareto_X is not None else None,
    #     "pareto_Y": trbo_state.pareto_Y.cpu() if trbo_state.pareto_Y is not None else None,
    #     "final_hypervolume": trbo_state.hv,
    #     "elapsed_time_sec": elapsed
    # }
    # --- SAVE RESULTS ---
    output_data = {
        "X_history": trbo_state.X_history.cpu(),
        "Y_history_transformed": Y_hist_cpu,  # Keep for debugging the BO space
        "Y_history_raw": Y_hist_raw,          # Use this for publication plots
        "pareto_X": trbo_state.pareto_X.cpu() if trbo_state.pareto_X is not None else None,
        "pareto_Y_transformed": pareto_Y_cpu,
        "pareto_Y_raw": pareto_Y_raw,
        "final_hypervolume": trbo_state.hv,
        "elapsed_time_sec": elapsed
    }
    save_path = os.path.join(run_output_dir, "bo_results.pt")
    torch.save(output_data, save_path)
    print(utils.GREEN + f"Results successfully saved to {save_path}" + utils.RESET)

    return trbo_state

def main(args):
    for i in range(args.num_trials):
        print(f"\n================ Trial {i+1} ================")
        for max_pkt_depth in args.max_pkt_depth.split(","):
            for num_init in args.num_init.split(","):
                for num_iter in args.num_iter.split(","):
                    for damping_factor in args.damping_factor.split(","):
                        print(f"Configs: max_pkt={max_pkt_depth}, init={num_init}, iter={num_iter}, priors={args.priors}, damp={damping_factor}")
                        
                        morbo_casmo_run(
                            candidate_features, 
                            int(max_pkt_depth), 
                            int(num_init), 
                            int(num_iter) - int(num_init), 
                            args.priors,
                            float(damping_factor), 
                            experiment_dir=args.experiment_dir,
                            use_log_warp=False, 
                            use_unified_kernel = True, 
                            use_mixture_kernel = False, 
                            use_casmo_mixed_kernel = False
                        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CATO with MORBO+Casmopolitan")
    parser.add_argument("max_pkt_depth", type=str, help="Comma separated list of Maximum packet depth")
    parser.add_argument("num_init", type=str, help="Comma separated list of Number of initial samples to query")
    parser.add_argument("num_iter", type=str, help="Comma separated list of Number of BO iterations")
    parser.add_argument("damping_factor", type=str, help="Comma separated list of Dampen feature priors. 0 = no damping, 1 = no prior")
    parser.add_argument("--mixture_kernel", action="store_true", help="Use lambda-weighted mixture kernel")
    parser.add_argument("experiment_dir", type=str, help="Path to experiment output dir")
    parser.add_argument("--unified_kernel", action="store_true", help="Use a single kernel for binary and ordinal dims")
    parser.add_argument("--num_trials", type=int, default=1, help="Number of trials")
    parser.add_argument("--priors", action="store_true", help="Include priors")

    parser.set_defaults(priors=True)
    main(parser.parse_args())