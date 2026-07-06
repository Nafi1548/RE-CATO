"""
morbo/discrete_ts_batch.py

Correct MORBO sequential greedy batch selection for pure discrete spaces.

Architecture per batch step i in {1, ..., q}:
  1. Each TR generates a deterministic local TS landscape using a Z-Cache.
  2. Each TR navigates this landscape using a discrete local search to maximize Marginal HVI.
  3. Global HVI competition selects the winner across all TRs.
  4. Kriging Believer: condition the winning TR's model on
     (x_winner, mu(x_winner)) before the next step.
"""

from __future__ import annotations

import logging
from typing import List, NamedTuple, Optional

import numpy as np
import gpytorch
import torch
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)
from botorch.utils.multi_objective.pareto import is_non_dominated
from torch import Tensor

from botorch.utils.sampling import sample_simplex

from morbo.state import TRBOState
from morbo.localbo_utils import interleaved_search

logger = logging.getLogger(__name__)


class CandidateSelectionOutput(NamedTuple):
    X_cand: Tensor       # (batch_size, d)  final selected batch
    tr_indices: Tensor   # (batch_size,)    which TR each point came from


# ---------------------------------------------------------------------------
# HVI of a single point over a fixed Pareto baseline
# ---------------------------------------------------------------------------

def _marginal_hvi_scores(
    f_ts: Tensor,           # (N, M) — TS objective values for a BATCH of candidates
    pareto_Y: Tensor,       # (n_pareto, M) — current Pareto front
    ref_point: Tensor,      # (M,)
    baseline_hv: float      # <--- NEW: Passed in to prevent O(N) recalculations
) -> Tensor:
    """
    Compute marginal HVI for a batch of candidates in f_ts.
    Returns an (N,) tensor of scores. Points that do not dominate ref_point receive 0.
    """
    device, dtype = f_ts.device, f_ts.dtype
    N_pool = f_ts.shape[0]
    scores = torch.zeros(N_pool, device=device, dtype=dtype)

    # Fast rejection: candidate must strictly dominate the reference point
    valid_mask = (f_ts > ref_point).all(dim=-1)

    for i in range(N_pool):
        # Skip expensive partitioning if the point doesn't even beat the reference
        if not valid_mask[i]:
            continue

        fk = f_ts[i:i+1]  # (1, M)

        try:
            if pareto_Y.numel() == 0:
                p = FastNondominatedPartitioning(ref_point=ref_point, Y=fk)
                scores[i] = p.compute_hypervolume().item()
            else:
                combined = torch.cat([pareto_Y, fk], dim=0)
                p = FastNondominatedPartitioning(ref_point=ref_point, Y=combined)
                scores[i] = max(0.0, p.compute_hypervolume().item() - baseline_hv)
        # except Exception:
        #     scores[i] = 0.0
        # Catch standard mathematical failures, let structural bugs crash the script
        except (ValueError, RuntimeError): 
            scores[i] = 0.0

    return scores

logger = logging.getLogger(__name__)

class ChebyshevUCBAcquisition:
    def __init__(self, model, ref_point, tkwargs, beta=2.0):
        """
        Phase 1: Deterministic, spatially correlated scalarization.
        Draws a random simplex weight vector for this specific restart.
        """
        self.model = model
        self.ref_point = ref_point.to(**tkwargs)
        self.tkwargs = tkwargs
        self.beta = beta
        
        # M is the number of objectives
        self.M = ref_point.shape[-1]
        
        # Draw a random weight vector lambda ~ Uniform(Simplex)
        # We add a small epsilon to avoid zero-weights ignoring an objective entirely
        raw_weights = sample_simplex(self.M, **tkwargs).squeeze() + 1e-3
        self.weights = raw_weights / raw_weights.sum()

    def __call__(self, x) -> float:
        """Evaluates the Chebyshev UCB score for a given point x."""
        # Normalize input format
        if isinstance(x, np.ndarray):
            x_t = torch.tensor(x.flatten(), **self.tkwargs).unsqueeze(0)
        else:
            x_t = x.view(1, -1)
            
        with torch.no_grad():
            # BoTorch automatically untransforms the posterior to the original objective space.
            # No manual outcome_transform inversion is needed.
            post = self.model.posterior(x_t)
                
            mean = post.mean.squeeze(0)
            std = post.variance.clamp_min(1e-12).sqrt().squeeze(0)
            
        # Upper Confidence Bound
        ucb = mean + self.beta * std
        
        # Chebyshev Scalarization (Maximization formulation)
        # min_m [ lambda_m * (UCB_m - ref_point_m) ]
        chebyshev_score = torch.min(self.weights * (ucb - self.ref_point), dim=-1)[0]
        
        return chebyshev_score

# def generate_chebyshev_pool(trbo_state, tr, model, n_restarts=10, step=50, interval=1):
#     """Runs independent local searches to build a high-quality candidate pool."""
#     tkwargs = {"device": trbo_state.bounds.device, "dtype": trbo_state.bounds.dtype}
    
#     # Extract explicitly separated dimensions required by interleaved_search
#     binary_dims = getattr(trbo_state.tr_hparams, "binary_dims", None) or []
#     ordinal_dims = getattr(trbo_state.tr_hparams, "ordinal_dims", None) or []
#     ordinal_config = getattr(trbo_state.tr_hparams, "ordinal_config", None) or []
#     cont_dims = getattr(trbo_state.tr_hparams, "cont_dims", None) or []
#     config = getattr(trbo_state.tr_hparams, "config", None) or []
    
#     x_center = tr.X_center_normalized.cpu().numpy().flatten()
#     length_discrete = int(tr.length_discrete.item())
    
#     # Handle continuous bounds (note: interleaved_search wants ub BEFORE lb)
#     if len(cont_dims) > 0:
#         lb = tr.bounds[0, cont_dims].cpu().numpy()
#         ub = tr.bounds[1, cont_dims].cpu().numpy()
#     else:
#         lb = np.array([])
#         ub = np.array([])

#     pool_X = []
    
#     from morbo.localbo_utils import interleaved_search

#     for r in range(n_restarts):
#         # 1. Instantiate the stationary acquisition function with a fresh random weight
#         acq_fn = ChebyshevUCBAcquisition(model, trbo_state.ref_point, tkwargs)
        
#         # 2. Run CASMOPOLITAN local search with the CORRECT signature
#         best_x_np, _ = interleaved_search(
#             x_center=x_center,
#             f=acq_fn,
#             binary_dims=binary_dims,
#             ordinal_dims=ordinal_dims,
#             ordinal_config=ordinal_config,
#             cont_dims=cont_dims, 
#             config=config,
#             ub=ub,              # MUST BE BEFORE lb
#             lb=lb,
#             max_hamming_dist=length_discrete,
#             n_restart=1,        # One restart per lambda
#             batch_size=1,
#             interval=interval,
#             step=step
#         )
#         pool_X.append(best_x_np[0])
        
#     # Deduplicate the pool
#     pool_X_np = np.unique(np.stack(pool_X, axis=0), axis=0)
#     return torch.tensor(pool_X_np, **tkwargs)
from morbo.utils import sample_tr_pure_discrete_subset
import torch
import numpy as np

def generate_chebyshev_pool(trbo_state, tr, model, n_restarts=5, step=150, interval=1):
    """Builds a hybrid candidate pool using dense random sampling and guided local search."""
    tkwargs = {"device": trbo_state.bounds.device, "dtype": trbo_state.bounds.dtype}
    
    # Extract configuration
    binary_dims = getattr(trbo_state.tr_hparams, "binary_dims", None) or []
    ordinal_dims = getattr(trbo_state.tr_hparams, "ordinal_dims", None) or []
    ordinal_config = getattr(trbo_state.tr_hparams, "ordinal_config", None) or []
    cont_dims = getattr(trbo_state.tr_hparams, "cont_dims", None) or []
    config = getattr(trbo_state.tr_hparams, "config", None) or []
    
    # Fetch the raw_samples hyperparameter (defaulting to 4096)
    raw_samples = getattr(trbo_state.tr_hparams, "raw_samples", 4096)
    
    x_center = tr.X_center_normalized.cpu().numpy().flatten()
    length_discrete = int(tr.length_discrete.item())
    
    # ---------------------------------------------------------
    # 1. MORBO EXPLORATION: Dense Random Sampling inside TR
    # ---------------------------------------------------------
    # For CATO, the space is discrete, so we generate a massive pool of neighbors
    X_random = sample_tr_pure_discrete_subset(
        best_X=tr.X_center_normalized.unsqueeze(0),
        n_discrete_points=raw_samples,
        length_discrete=length_discrete,
        binary_dims=binary_dims,
        ordinal_dims=ordinal_dims,
        ordinal_config=ordinal_config
    ).to(**tkwargs)

    # ---------------------------------------------------------
    # 2. CASMOPOLITAN EXPLOITATION: Guided Chebyshev Search
    # ---------------------------------------------------------
    if len(cont_dims) > 0:
        lb = tr.bounds[0, cont_dims].cpu().numpy()
        ub = tr.bounds[1, cont_dims].cpu().numpy()
    else:
        lb = np.array([])
        ub = np.array([])

    pool_X_cheb = []
    from morbo.localbo_utils import interleaved_search

    for r in range(n_restarts):
        acq_fn = ChebyshevUCBAcquisition(model, trbo_state.ref_point, tkwargs)
        
        best_x_np, _ = interleaved_search(
            x_center=x_center,
            f=acq_fn,
            binary_dims=binary_dims,
            ordinal_dims=ordinal_dims,
            ordinal_config=ordinal_config,
            cont_dims=cont_dims, 
            config=config,
            ub=ub,
            lb=lb,
            max_hamming_dist=length_discrete,
            n_restart=1, 
            batch_size=1,
            interval=interval,
            step=step
        )
        pool_X_cheb.append(best_x_np[0])
        
    X_cheb = torch.tensor(np.stack(pool_X_cheb, axis=0), **tkwargs)

    # ---------------------------------------------------------
    # 3. COMBINE AND DEDUPLICATE
    # ---------------------------------------------------------
    combined_pool = torch.cat([X_random, X_cheb], dim=0)
    
    # Remove duplicates efficiently on GPU to avoid redundant GP evaluations
    final_pool = torch.unique(combined_pool, dim=0)

    return final_pool

# ---------------------------------------------------------------------------
# PHASE 2: Main Batch Selection Loop
# ---------------------------------------------------------------------------
def discrete_ts_batch_select(trbo_state: TRBOState) -> CandidateSelectionOutput:
    tkwargs = {"device": trbo_state.bounds.device, "dtype": trbo_state.bounds.dtype}
    batch_size = trbo_state.tr_hparams.batch_size
    n_trs = len(trbo_state.trust_regions)
    objective = trbo_state.trust_regions[0].objective
    ref_point = trbo_state.ref_point.clone().to(**tkwargs)
    
    fantasy_models = [trbo_state.models[j] for j in range(n_trs)]
    current_pareto_Y = objective(trbo_state.pareto_Y_better_than_ref.clone().to(**tkwargs))

    X_next = torch.empty(0, trbo_state.dim, **tkwargs)
    tr_indices_selected = torch.zeros(batch_size, device=tkwargs["device"], dtype=torch.long)

    # 1. GENERATE CANDIDATE POOLS FOR ALL TRs (Done once per batch step)
    tr_pools = []
    for j in range(n_trs):
        tr = trbo_state.trust_regions[j]
        model_j = fantasy_models[j]
        
        logger.info(f"Generating Phase 1 Chebyshev Pool for TR {j}...")
        pool_X = generate_chebyshev_pool(
            trbo_state=trbo_state, 
            tr=tr, 
            model=model_j, 
            n_restarts=5, 
            step=150
        )
        tr_pools.append(pool_X)

    # 2. BATCH SELECTION (Joint TS + HVI)
    for i in range(batch_size):
        best_cand = None
        best_hvi = float("-inf")
        best_tr_idx = 0
        # --- FIX ISSUE 10: Compute Baseline HV exactly ONCE per batch step ---
        if current_pareto_Y.numel() > 0:
            base_part = FastNondominatedPartitioning(ref_point=ref_point, Y=current_pareto_Y)
            baseline_hv = base_part.compute_hypervolume().item()
        else:
            baseline_hv = 0.0
        # ---------------------------------------------------------------------
        for j in range(n_trs):
            pool_X = tr_pools[j]
            model_j = fantasy_models[j]
            
            # Draw an EXACT Joint Thompson Sample over the finite pool
            # with torch.no_grad(), gpytorch.settings.fast_pred_var():
            #     posterior = model_j.posterior(pool_X)
            #     ts_sample = posterior.sample().squeeze(0)  # Shape: (N_pool, M)
            
            with torch.no_grad(), gpytorch.settings.max_cholesky_size(float("inf")):
                posterior = model_j.posterior(pool_X)
                ts_sample = posterior.sample(torch.Size([1])).squeeze(0)   # (N_pool, M)

            ts_obj = objective(ts_sample)


            # --- ONE SHOT CALL: Score the entire pool instantly ---
            scores = _marginal_hvi_scores(ts_obj, current_pareto_Y, ref_point, baseline_hv)
            
            # Find the best candidate in this TR's pool
            best_local_idx = scores.argmax()
            best_local_score = scores[best_local_idx].item()
            
            if best_local_score > best_hvi:
                best_hvi = best_local_score
                best_cand = pool_X[best_local_idx].unsqueeze(0)
                best_tr_idx = j

            # # Evaluate the Marginal HVI for every point in the pool
            # for idx in range(pool_X.shape[0]):
            #     f_ts_single = ts_obj[idx].unsqueeze(0)
            #     score = _marginal_hvi_scores(f_ts_single, current_pareto_Y, ref_point).item()
                
            #     if score > best_hvi:
            #         best_hvi = score
            #         best_cand = pool_X[idx].unsqueeze(0)
            #         best_tr_idx = j

        # Fallback if no improvement
        if best_cand is None or best_hvi <= 0.0:
            j_fallback = int(torch.randint(n_trs, (1,)).item())
            best_cand = trbo_state.trust_regions[j_fallback].X_center_normalized
            best_tr_idx = j_fallback

        tr_indices_selected[i] = best_tr_idx
        X_next = torch.cat([X_next, best_cand], dim=0)
        
        # ---------------------------------------------------------
        # 3. KRIGING BELIEVER UPDATE (ONLY IF BATCH SIZE > 1)
        # ---------------------------------------------------------
        # For q=1, hallucinating points is mathematically useless and 
        # induces severe matrix singularities in discrete spaces.
        if batch_size > 1 and i < batch_size - 1:
            with torch.no_grad():
                # EXACT FIX: Use the deterministic posterior mean to prevent stochastic drift
                fantasy_y_raw = fantasy_models[best_tr_idx].posterior(best_cand).mean

            fantasy_f_obj = objective(fantasy_y_raw)

            above_ref = (fantasy_f_obj > ref_point).all(dim=-1)
            if above_ref.any():
                current_pareto_Y = torch.cat([current_pareto_Y, fantasy_f_obj[above_ref]], dim=0)
                current_pareto_Y = current_pareto_Y[is_non_dominated(current_pareto_Y)]

            try:
                fantasy_models[best_tr_idx] = fantasy_models[best_tr_idx].condition_on_observations(
                    X=best_cand,
                    Y=fantasy_y_raw,
                )

                # 4. SYNCHRONIZE POOL (Regenerate only the winning TR's pool for the next batch step)
                if i < batch_size - 1:
                    tr_pools[best_tr_idx] = generate_chebyshev_pool(
                        trbo_state=trbo_state, 
                        tr=trbo_state.trust_regions[best_tr_idx], 
                        model=fantasy_models[best_tr_idx], 
                        n_restarts=5, 
                        step=150
                    )

            except Exception as e:
                logger.warning(f"KB Failed TR {best_tr_idx}: {e}")

    return CandidateSelectionOutput(X_cand=X_next, tr_indices=tr_indices_selected)