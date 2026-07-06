import torch
import numpy as np
import time
from botorch.acquisition.objective import IdentityMCObjective
from botorch.utils.sampling import draw_sobol_samples

# Casmopolitan imports
# from mixed_test_func import PestControl
from test_funcs import PestControl
from test_funcs.random_seed_config import generate_random_seed_pestcontrol

# MORBO imports
from morbo.state import TRBOState
from morbo.trust_region import TurboHParams
# from morbo.gen import TS_select_batch_MORBO
from morbo.discrete_ts_batch import discrete_ts_batch_select
class PestControlWrapper:
    """Wraps Casmopolitan's PestControl to work with MORBO's PyTorch logic."""
    def __init__(self, f_obj, dtype, device):
        self.f_obj = f_obj
        self.dtype = dtype
        self.device = device

    def __call__(self, X: torch.Tensor):
        X_np = X.detach().cpu().numpy()
        
        # Get normalized values from Casmopolitan
        Y_np = self.f_obj.compute(X_np, normalize=self.f_obj.normalize)
        
        # NEGATE for MORBO so it maximizes the right direction
        Y_np = -Y_np 
        
        return torch.as_tensor(Y_np, dtype=self.dtype, device=self.device).view(-1, 1)

def run_pestcontrol_merged(
    seed=0,
    max_evals=400,
    batch_size=1,
    n_init=20,
    n_trust_regions=1 # Using 1 TR is common for single-objective local BO
):
    # 1. Setup device and seed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.double
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 2. Instantiate the PestControl problem from Casmopolitan
    # The default value of 20 is provided also in COMBO
    random_seed_ = sorted(generate_random_seed_pestcontrol())[20] 
    f_casmo = PestControl(random_seed=random_seed_)
    
    # Wrap it for PyTorch
    f = PestControlWrapper(f_casmo, dtype=dtype, device=device)

   # 3. Extract Space Configurations (Purely Categorical)
    # PestControl might use 'config' or 'n_vertices' depending on the Casmopolitan version
    config = f_casmo.config if hasattr(f_casmo, 'config') else f_casmo.n_vertices
    dim = len(config)
    
    cat_dims = list(range(dim))  # All dimensions are categorical
    cont_dims = []               # No continuous dimensions!

    # Construct overall bounds tensor for MORBO
    bounds = torch.zeros(2, dim, dtype=dtype, device=device)
    # Categorical bounds: [0, n_categories - 1]
    for i, n_cat in enumerate(config):
        bounds[0, i] = 0.0
        bounds[1, i] = float(n_cat - 1)

    # Continuous bounds
    for i, c_dim in enumerate(cont_dims):
        bounds[0, c_dim] = f_casmo.lb[i]
        bounds[1, c_dim] = f_casmo.ub[i]

   # 4. Set up Mixed-Space TurboHParams
    tr_hparams = TurboHParams(
        length_init=0.8,
        length_min=0.5**7,
        length_max=1.6,
        length_max_discrete=25, 
        length_init_discrete=20, 
        length_min_discrete=1,
        cat_dims=cat_dims,      
        cont_dims=cont_dims,    
        config=config,          
        batch_size=batch_size,
        n_initial_points=n_init,
        min_tr_size=n_init,     # <-- THIS IS THE NEW FIX
        n_trust_regions=n_trust_regions,
        hypervolume=False,      
        verbose=True,
        use_simple_rff=True,
        success_streak=3,       # <--- NEW: Casmopolitan's default success tolerance
        failure_streak=4       # <--- NEW: Casmopolitan's default failure tolerance
    )

    # 5. Initialize TRBO State
    # Note: For single objective, num_outputs=1, num_objectives=1
    trbo_state = TRBOState(
        dim=dim,
        num_outputs=1,
        num_objectives=1,
        bounds=bounds,
        max_evals=max_evals,
        tr_hparams=tr_hparams,
        objective=IdentityMCObjective(), #
    )

    # 6. Generate Initial Points
    print("Generating initial points...")
    
    # Draw Sobol samples strictly for continuous dimensions (if any exist)
    X_init = draw_sobol_samples(bounds=bounds, n=n_init, q=1).squeeze(1)
    
    # Overwrite categorical dimensions with EXACT uniform integer sampling
    for i, c in enumerate(cat_dims):
        n_cat = config[i]
        X_init[:, c] = torch.randint(0, n_cat, (n_init,), dtype=dtype, device=device)
        
    Y_init = f(X_init)

    trbo_state.update(
        X=X_init,
        Y=Y_init,
        new_ind=torch.full((n_init,), 0, dtype=torch.long, device=device),
    )

    # Initialize standard Trust Region
    for i in range(n_trust_regions):
        trbo_state.initialize_standard(
            tr_idx=i,
            restart=False,
            switch_strategy=False,
            X_init=X_init,
            Y_init=Y_init,
        )

    # 7. Main Optimization Loop
    print(f"----- Starting Optimization (Max Evals: {max_evals}) -----")
    all_tr_indices = [-1] * n_init
    
    while trbo_state.n_evals < max_evals:
        # Generate candidates using merged TS logic
        selection_output = discrete_ts_batch_select(trbo_state=trbo_state)
        X_cand = selection_output.X_cand
        tr_indices = selection_output.tr_indices
        
        # Evaluate
        Y_cand = f(X_cand)
        
        # Update State
        trbo_state.update(X=X_cand, Y=Y_cand, new_ind=tr_indices)
        should_restart = trbo_state.update_trust_regions_and_log(
            X_cand=X_cand,
            Y_cand=Y_cand,
            tr_indices=tr_indices,
            batch_size=batch_size,
            verbose=True
        )

        # Handle Restarts if TR collapsed
        for i in range(n_trust_regions):
            if should_restart[i]:
                print(f"Restarting Trust Region {i}...")
                trbo_state.TR_index_history[trbo_state.TR_index_history == i] = -1
                trbo_state.initialize_standard(
                    tr_idx=i,
                    restart=True,
                    switch_strategy=False
                )
                
        # Optional: Print best observed value so far
        best_val = trbo_state.Y_history.max().item()
        print(f"Iteration {trbo_state.n_evals}/{max_evals} | Best Y: {best_val:.4f}")

    # ... [End of your while loop] ...
    print("Optimization Complete!")
    
    n_evals_array = list(range(1, len(trbo_state.Y_history) + 1))
    
    # 1. Re-negate the objective
    true_normalized_Y = -trbo_state.Y_history
    
    # 2. Un-normalize
    f_mean = torch.tensor(f_casmo.mean, dtype=dtype, device=device)
    f_std = torch.tensor(f_casmo.std, dtype=dtype, device=device)
    true_unnormalized_Y = (true_normalized_Y * f_std) + f_mean
    
    # 3. Track the MINIMUM score (using cummin!)
    best_y_history = torch.cummin(true_unnormalized_Y, dim=0)[0].cpu().numpy().flatten()
    
    output = {
        "n_evals": n_evals_array,
        "best_y_history": best_y_history,
        "all_y": true_unnormalized_Y.cpu().numpy().flatten()
    }
    
    torch.save(output, "merged_pestcontrol_results.pt")

if __name__ == "__main__":
    run_pestcontrol_merged()
