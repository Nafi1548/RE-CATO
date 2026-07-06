import argparse
import time
import os
import numpy as np
import pandas as pd
import torch
from botorch.acquisition.objective import IdentityMCObjective
from botorch.utils.sampling import draw_sobol_samples

# Import your merged framework components
from morbo.state import TRBOState
from morbo.trust_region import TurboHParams
# from morbo.gen import TS_select_batch_MORBO
from morbo.discrete_ts_batch import discrete_ts_batch_select
# Corrected import for the test function
from mixed_test_func.synthetic import Ackley53

def main():
    parser = argparse.ArgumentParser(description='Run Ackley53 on merged MORBO+Casmopolitan framework')
    parser.add_argument('--max_iters', type=int, default=400, help='Maximum number of BO iterations.')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for BO.')
    parser.add_argument('--n_init', type=int, default=30, help='Number of initialising random points')
    parser.add_argument('--lamda', type=float, default=1e-6, help='Noise injection')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--save_path', type=str, default='./output/', help='Save directory')
    args = parser.parse_args()

    # 1. Setup Device & Seed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tkwargs = {"dtype": torch.float64, "device": device}
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    os.makedirs(args.save_path, exist_ok=True)

    # 2. Instantiate Objective
    f_ackley = Ackley53(lamda=args.lamda)
    cat_dims = list(f_ackley.categorical_dims)
    cont_dims = list(f_ackley.continuous_dims)
    config = list(f_ackley.config)
    dim = f_ackley.dim

    # Wrapper to make Ackley53 compatible with MORBO (MORBO maximizes, Ackley minimizes)
    def eval_objective(X: torch.Tensor) -> torch.Tensor:
        X_np = X.cpu().detach().numpy()
        y_np = f_ackley.compute(X_np)
        y_tensor = torch.tensor(y_np, **tkwargs)
        return -y_tensor.view(-1, 1)  # Negate because MORBO maximizes

    # 3. Define Bounds
    bounds = torch.zeros(2, dim, **tkwargs)
    bounds[0, cont_dims] = torch.tensor(f_ackley.lb, **tkwargs)
    bounds[1, cont_dims] = torch.tensor(f_ackley.ub, **tkwargs)
    for i, c in enumerate(cat_dims):
        bounds[0, c] = 0
        bounds[1, c] = config[i] - 1  # Categorical bounds: 0 to (n_vertices - 1)

    # 4. Configure TuRBO/MORBO Hyperparameters mapping to Casmopolitan Kwargs
    tr_hparams = TurboHParams(
        cat_dims=cat_dims,
        cont_dims=cont_dims,
        config=config,
        length_init_discrete=30,  
        length_max_discrete=50,   
        length_min_discrete=1,
        length_init=0.8,
        length_min=0.01,
        length_max=1.6,
        success_streak=3,                             # <--- ADD THIS
        failure_streak=max(dim // 3, 10),             # <--- ADD THIS (or just set to 40)
        batch_size=args.batch_size,
        n_initial_points=args.n_init,
        n_trust_regions=1,        
        max_tr_size=2000,
        min_tr_size=10,
        hypervolume=False,        
        use_ard=True,
        track_history=True,
        verbose=True
    )

    # 5. Initialize TRBO State
    trbo_state = TRBOState(
        dim=dim,
        max_evals=args.max_iters * args.batch_size + args.n_init,
        num_outputs=1,
        num_objectives=1,
        bounds=bounds,
        tr_hparams=tr_hparams,
        objective=IdentityMCObjective()
    ).to(**tkwargs)

    # 6. Generate Initial Sobol Points
    print(f"Generating {args.n_init} initial points...")
    X_init = draw_sobol_samples(bounds=bounds, n=args.n_init, q=1).squeeze(1)
    X_init[:, cat_dims] = torch.round(X_init[:, cat_dims]) # Ensure categorical are integers
    Y_init = eval_objective(X_init)

    trbo_state.update(
        X=X_init,
        Y=Y_init,
        new_ind=torch.full((X_init.shape[0],), 0, dtype=torch.long, device=device),
    )
    trbo_state.log_restart_points(X=X_init, Y=Y_init)

    # Initialize standard Trust Region
    for i in range(tr_hparams.n_trust_regions):
        trbo_state.initialize_standard(
            tr_idx=i, restart=False, switch_strategy=False, X_init=X_init, Y_init=Y_init
        )
    trbo_state.update_data_across_trs()
    trbo_state.TR_index_history.fill_(-2)

    # 7. Tracking Setup (Mirroring Casmopolitan's DataFrame)
    res = pd.DataFrame(np.nan, index=np.arange(args.max_iters * args.batch_size),
                       columns=['Index', 'LastValue', 'BestValue', 'Time'])

    # 8. Main Optimization Loop
    print(f"----- Starting BO Loop for {args.max_iters} iterations -----")
    for i in range(args.max_iters):
        start = time.time()
        
        # Select candidates using merged Thompson Sampling + Interleaved Search
        selection_output = discrete_ts_batch_select(trbo_state=trbo_state)
        X_next = selection_output.X_cand
        tr_indices = selection_output.tr_indices
        trbo_state.tabu_set.log_iteration()
        
        # Evaluate
        Y_next = eval_objective(X_next)
        end = time.time()
        
        # Update State
        trbo_state.update(X=X_next, Y=Y_next, new_ind=tr_indices)
        should_restart_trs = trbo_state.update_trust_regions_and_log(
            X_cand=X_next, Y_cand=Y_next, tr_indices=tr_indices, 
            batch_size=args.batch_size, verbose=True
        )

        # Handle Restarts
        if any(should_restart_trs):
            for tr_idx in range(tr_hparams.n_trust_regions):
                if should_restart_trs[tr_idx]:
                    print(f"Restarting trust region {tr_idx}")
                    trbo_state.TR_index_history[trbo_state.TR_index_history == tr_idx] = -1
                    trbo_state.initialize_standard(
                        tr_idx=tr_idx, restart=True, switch_strategy=False
                    )

        # Logging (reversing the objective negation back to minimization for display)
        current_y = -Y_next[-1].item()
        best_y = -trbo_state.Y_history.max().item()
        
        for idx in range(args.batch_size):
            global_idx = i * args.batch_size + idx
            res.iloc[global_idx, :] = [global_idx, current_y, best_y, end - start]

        print(f"Iter {i:03d} | Last f(X): {current_y:.4f} | Best f(X): {best_y:.4f} | TR Length [Cont/Disc]: {trbo_state.trust_regions[0].length.item():.2f}/{trbo_state.trust_regions[0].length_discrete.item()}")

    # 9. Save Results
    save_file = os.path.join(args.save_path, f"morbo_casmo_ackley53_seed{args.seed}.csv")
    res.to_csv(save_file, index=False)
    print(f"Optimization complete. Results saved to {save_file}")

if __name__ == '__main__':
    main()