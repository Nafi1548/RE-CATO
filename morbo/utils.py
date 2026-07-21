#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# import gpytorch
from math import ceil, log
from typing import Any, Callable, Dict, List, Optional
import numpy as np
# Add this to the top of utils.py imports
from morbo.kernels import MixtureKernel
import torch
from botorch.exceptions.errors import BotorchTensorDimensionError
# from botorch.fit import fit_gpytorch_model
from botorch.fit import fit_gpytorch_mll
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.model import Model
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
# from botorch.optim.fit import fit_gpytorch_torch
from botorch.utils.sampling import draw_sobol_samples
from gpytorch import settings as gpytorch_settings
from gpytorch.constraints import GreaterThan, Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood, SumMarginalLogLikelihood
from gpytorch.priors.torch_priors import GammaPrior
from torch import Tensor
from torch.distributions import Normal
# from morbo.kernels import UnifiedDiscreteKernel
from morbo.kernels import UnifiedL1DiscreteKernel,DiscreteMixtureKernel

def sample_tr_discrete_points(
    X_center: Tensor, length: float, n_discrete_points: int, qmc: bool = False
) -> Tensor:
    r"""Sample points around `X_center` for use in discrete Thompson sampling.

    Sample perturbed points around `X_center` such that the added perturbations
        are sampled from N(0, (length/4)^2) and truncated to be within
        [-length/2, -length/2].

    Args:
        X_center: a `1 x d`-dim tensor containing the center of trust region. `X_center`
            must be normalized to be within `[0, 1]^d`.
        length: edge length of the trust region's hypercube.
        n_discrete_points: number of points to sample for use in discrete TS.
        qmc: boolean indicating whether to use qmc

    Returns:
        Tensor: a `n_discrete_points x d`-dim tensor containing the sampled points.
    """
    d = X_center.shape[1]
    # sample points from N(X_center, (length/4)^2), truncated to be within
    # [X_center-length/2, X_center+length/2].
    # To do this, we sample perturbations from N(0, (length/4)^2) truncated to be
    # within [max(-X_center, -L/2), min(1-X_center, L/2) using the inverse transform
    # and then add these perturbations to X_center.
    sigma = length / 4.0
    if qmc:
        bounds = torch.stack(
            [torch.zeros_like(X_center[0]), torch.ones_like(X_center[0])], dim=0
        )
        u = draw_sobol_samples(bounds=bounds, n=n_discrete_points, q=1).squeeze(1)
    else:
        u = torch.rand(
            (n_discrete_points, d), dtype=X_center.dtype, device=X_center.device
        )
    # compute bounds to sample from
    a = (-X_center).clamp_min(-length / 2.0)
    b = (1 - X_center).clamp_max(length / 2.0)
    # compute z-score of bounds
    alpha = a / sigma
    beta = b / sigma
    normal = Normal(0, 1)
    cdf_alpha = normal.cdf(alpha)
    perturbation = normal.icdf(cdf_alpha + u * (normal.cdf(beta) - cdf_alpha)) * sigma
    X_discrete = X_center + perturbation

    # Clip points that are still outside
    return X_discrete.clamp(0.0, 1.0)


# def sample_tr_discrete_points_subset_d(
#     best_X: Tensor,
#     normalized_tr_bounds: Tensor,
#     n_discrete_points: int,
#     length: float,
#     qmc: bool = False,
#     trunc_normal_perturb: bool = False,
#     prob_perturb: float = None,
# ) -> Tensor:
#     r"""Sample discrete for TS by perturbing ~20 dims of `best_X`.

#     If `trunc_normal_perturb=True`, the perturbed samples are truncated normal
#     around `best_X`. Otherwise, these are uniformly distributed in
#     `normalize_tr_bounds`.
#     """
#     assert normalized_tr_bounds.ndim == 2
#     d = normalized_tr_bounds.shape[-1]
#     if prob_perturb is None:
#         # Only perturb a subset of the features
#         prob_perturb = min(20.0 / d, 1.0)

#     if best_X.shape[0] == 1:
#         X_cand = best_X.repeat(n_discrete_points, 1)
#     else:
#         rand_indices = torch.randint(
#             best_X.shape[0], (n_discrete_points,), device=best_X.device
#         )
#         X_cand = best_X[rand_indices]

#     if trunc_normal_perturb:
#         pert = sample_tr_discrete_points(
#             X_center=X_cand, length=length, n_discrete_points=n_discrete_points, qmc=qmc
#         )
#         # make sure perturbations are in bounds
#         # if X_cand contains pareto points, the perturbed points might not be in the TR
#         # TODO: refactor to this into a `project_on_box` helper function T65690436
#         pert = torch.min(
#             torch.max(pert, normalized_tr_bounds[0]), normalized_tr_bounds[1]
#         )
#     elif qmc:
#         pert = draw_sobol_samples(
#             bounds=normalized_tr_bounds, n=n_discrete_points, q=1
#         ).squeeze(1)
#     else:
#         pert = torch.rand(
#             n_discrete_points,
#             d,
#             dtype=normalized_tr_bounds.dtype,
#             device=normalized_tr_bounds.device,
#         )
#         pert = (
#             normalized_tr_bounds[1] - normalized_tr_bounds[0]
#         ) * pert + normalized_tr_bounds[0]

#     # find cases where we are not perturbing any dimensions
#     mask = (
#         torch.rand(
#             n_discrete_points,
#             d,
#             dtype=normalized_tr_bounds.dtype,
#             device=normalized_tr_bounds.device,
#         )
#         <= prob_perturb
#     )
#     ind = (~mask).all(dim=-1).nonzero()
#     # perturb `n_perturb` of the dimensions
#     n_perturb = ceil(d * prob_perturb)
#     perturb_mask = torch.zeros(d, dtype=mask.dtype, device=mask.device)
#     perturb_mask[:n_perturb].fill_(1)
#     for idx in ind:
#         mask[idx] = perturb_mask[torch.randperm(d, device=normalized_tr_bounds.device)]
#     # Create candidate points
#     X_cand[mask] = pert[mask]
#     return X_cand


# def sample_tr_pure_discrete_subset(
#     best_X: Tensor,
#     n_discrete_points: int,
#     length_discrete: int,
#     binary_dims: List[int],
#     ordinal_dims: List[int],
#     ordinal_config: List[int]
# ) -> Tensor:
#     """Sample discrete neighbors using exclusively local search within the TR[cite: 9]."""
#     d_bin = len(binary_dims)
#     d_ord = len(ordinal_dims)
    
#     # # Expand the center point
#     # X_cand = best_X.repeat(n_discrete_points, 1)
    
#     # FIX BUG 2: Safe handling of multi-row Pareto fronts
#     if best_X.shape[0] == 1:
#         X_cand = best_X.repeat(n_discrete_points, 1)
#     else:
#         rand_indices = torch.randint(best_X.shape[0], (n_discrete_points,), device=best_X.device)
#         X_cand = best_X[rand_indices].clone()

#     # 1. Perturb Binary Dimensions (Bit-Flipping)
#     X_cand_bin = X_cand[:, binary_dims].clone()
#     for i in range(n_discrete_points):
#         # Pick a random Hamming distance bounded by length_discrete[cite: 9]
#         bit_change = torch.randint(1, min(length_discrete, d_bin) + 1, (1,)).item()
#         modified_bits = torch.randperm(d_bin)[:bit_change]
        
#         for bit in modified_bits:
#             # Flip bit: 1 -> 0, 0 -> 1[cite: 9]
#             X_cand_bin[i, bit.item()] = 1 - X_cand_bin[i, bit.item()]
    
#     # 2. Perturb Ordinal Dimension (Stepping)
#     X_cand_ord = X_cand[:, ordinal_dims].clone()
#     for i in range(n_discrete_points):
#         for idx, bit in enumerate(ordinal_dims):
#             n_cat = ordinal_config[idx]
#             current_val = int(X_cand_ord[i, idx].item())
            
#             # Step purely up or down by 1 within bounds
#             step = torch.randint(-1, 2, (1,)).item() 
#             if step != 0:
#                 new_val = max(0, min(n_cat - 1, current_val + step))
#                 X_cand_ord[i, idx] = new_val

#     # Recombine
#     X_cand[:, binary_dims] = X_cand_bin
#     X_cand[:, ordinal_dims] = X_cand_ord
    
#     return X_cand

def sample_tr_pure_discrete_subset(best_X, n_discrete_points, length_discrete, binary_dims, ordinal_dims, ordinal_config, use_log_warp=False):
    # FIX BUG 2: Safe handling of multi-row Pareto fronts
    if best_X.shape[0] == 1:
        # X_cand = best_X.repeat(n_discrete_points, 1)
        # Force the center into a strictly 2D shape [1, d] before repeating
        # This strips any [1, 1, d] ghost dimensions inherited from the TR state
        X_cand = best_X.view(1, -1).repeat(n_discrete_points, 1)
    else:
        rand_indices = torch.randint(best_X.shape[0], (n_discrete_points,), device=best_X.device)
        X_cand = best_X[rand_indices].clone()

    X_cand_bin = X_cand[:, binary_dims]
    X_cand_ord = X_cand[:, ordinal_dims]
    
    d_bin = len(binary_dims)
    d_ord = len(ordinal_dims)

    total_dims = d_bin + d_ord

    # 1. Calculate the Trust Region Ratio
    tr_ratio = float(length_discrete) / float(total_dims)

    # # FIX BUG 9: Joint Hamming Budget logic
    # for i in range(n_discrete_points):
    #     total_hamming = length_discrete
    #     ordinal_moves = 0
        
    #     # 2. Adaptive Ordinal Stepping
    #     # Determine ordinal step (costs 1 budget if it moves)
    #     for idx in range(d_ord):
    #         n_cat = ordinal_config[idx]
    #         current_val = int(X_cand_ord[i, idx].item())

    #         # Scale the maximum allowable jump by the TR ratio
    #         dynamic_radius = max(1, int(n_cat * tr_ratio))

    #         # step = torch.randint(-1, 2, (1,)).item()

    #         # Sample uniformly within the dynamic radius
    #         # step = torch.randint(-dynamic_radius, dynamic_radius + 1, (1,)).item()

    #         left_radius  = min(dynamic_radius, current_val)
    #         right_radius = min(dynamic_radius, n_cat - 1 - current_val)
    #         if left_radius + right_radius == 0:
    #             step = 0
    #         else:
    #             step = torch.randint(-left_radius, right_radius + 1, (1,)).item()
    #         new_val = current_val + step  # already guaranteed in-bounds

    #         if step != 0:
    #             # new_val = max(0, min(n_cat - 1, current_val + step))
    #             if new_val != current_val:
    #                 X_cand_ord[i, idx] = new_val
    #                 ordinal_moves += 1 # Consumes max 1.0 of the budget

        # FIX: Implement log-uniform Jeffreys prior for scale parameter exploration
    # In morbo/utils.py -> sample_tr_pure_discrete_subset
    for i in range(n_discrete_points):
        total_hamming = length_discrete
        ordinal_moves = 0
        
        for idx in range(d_ord):
            n_cat = ordinal_config[idx] 
            current_val = int(X_cand_ord[i, idx].item())
            
            if use_log_warp:
                # Native log-space draw (Jeffreys)
                import math
                z_max = math.log(n_cat)
                z_curr = math.log(current_val + 1)
                r_log = tr_ratio * z_max
                z_min_bound = max(0.0, z_curr - r_log)
                z_max_bound = min(z_max, z_curr + r_log)
                
                u = torch.rand(1).item()
                z_new = z_min_bound + u * (z_max_bound - z_min_bound)
                new_val = int(math.exp(z_new)) - 1
            else:
                # Linear uniform draw bounded by TR ratio
                dynamic_radius = max(1, int(n_cat * tr_ratio))
                left_radius = min(dynamic_radius, current_val)
                right_radius = min(dynamic_radius, n_cat - 1 - current_val)
                
                step = 0
                if left_radius + right_radius > 0:
                    step = torch.randint(-left_radius, right_radius + 1, (1,)).item()
                new_val = current_val + step

            new_val = max(0, min(n_cat - 1, new_val))
            if new_val != current_val:
                X_cand_ord[i, idx] = new_val
                ordinal_moves += 1
        
        # # Spend remaining budget on binary dimensions
        # binary_budget = max(1, total_hamming - ordinal_moves)
        # bit_change = torch.randint(1, min(binary_budget, d_bin) + 1, (1,)).item()
        # Spend remaining budget on binary dimensions
        binary_budget = total_hamming - ordinal_moves
        
        if binary_budget > 0:
            # # # If ordinal didn't move, we MUST flip at least 1 bit so the point isn't identical
            # # min_bits = 1 if ordinal_moves == 0 else 0
            
            # # if min_bits == 0:
            # #     # We can optionally flip bits up to the binary budget
            # #     bit_change = torch.randint(0, min(binary_budget, d_bin) + 1, (1,)).item()
            # # else:
            # #     # We MUST flip at least 1 bit
            # #     bit_change = torch.randint(1, min(binary_budget, d_bin) + 1, (1,)).item()
            # # Determine how many binary bits to flip
            # if ordinal_moves == 0:
            #     # The ordinal dimension didn't move. We MUST force at least 1 bit flip 
            #     # so the candidate isn't identical to the center (wasting a slot).
            #     min_bits = 1 if d_bin > 0 else 0

            #     # The maximum bits we can flip is constrained by the budget or total binary dims
            #     max_bits = min(binary_budget, d_bin)

            #     # Failsafe: only draw if max_bits is valid and >= min_bits
            #     if max_bits >= min_bits:
            #         bit_change = torch.randint(min_bits, max_bits + 1, (1,)).item()
            #     else:
            #         bit_change = 0
            # else:
            #     bit_change = torch.randint(0, min(binary_budget, d_bin) + 1, (1,)).item()
            min_bits = 1 if ordinal_moves == 0 and d_bin > 0 else 0
            max_bits = min(binary_budget, d_bin)

            if max_bits >= min_bits:
                bit_change = torch.randint(min_bits, max_bits + 1, (1,)).item()
            else:
                bit_change = 0

                
            # Apply the bit flips
            if bit_change > 0:
                modified_bits = torch.randperm(d_bin)[:bit_change]
                for bit in modified_bits:
                    X_cand_bin[i, bit.item()] = 1 - X_cand_bin[i, bit.item()]

    X_cand[:, binary_dims] = X_cand_bin
    X_cand[:, ordinal_dims] = X_cand_ord
    
    return X_cand

def sample_tr_discrete_points_subset_d(
    best_X: Tensor,
    normalized_tr_bounds: Tensor,
    n_discrete_points: int,
    length: float,
    qmc: bool = False,
    trunc_normal_perturb: bool = False,
    prob_perturb: float = None,
    cat_dims: Optional[List[int]] = None,      # --- ADDED ---
    cont_dims: Optional[List[int]] = None,     # --- ADDED ---
    length_discrete: Optional[int] = None,     # --- ADDED ---
    config: Optional[List[int]] = None,        # --- ADDED ---
    # FIX BUG F: Add missing fields to signature
    binary_dims: Optional[List[int]] = None,
    ordinal_dims: Optional[List[int]] = None,
    ordinal_config: Optional[List[int]] = None,
) -> Tensor:
    r"""Sample discrete for TS by perturbing a subset of dims of `best_X`."""
    assert normalized_tr_bounds.ndim == 2
    d = normalized_tr_bounds.shape[-1]
    if prob_perturb is None:
        prob_perturb = min(20.0 / d, 1.0)

    # Fix Bug 2: Handle multi-row Pareto fronts correctly
    if best_X.shape[0] == 1:
        # X_cand = best_X.repeat(n_discrete_points, 1)
        # Force the center into a strictly 2D shape [1, d] before repeating
        # This strips any [1, 1, d] ghost dimensions inherited from the TR state
        X_cand = best_X.view(1, -1).repeat(n_discrete_points, 1)
    else:
        rand_indices = torch.randint(best_X.shape[0], (n_discrete_points,), device=best_X.device)
        X_cand = best_X[rand_indices].clone()
        
    X_cand_bin = X_cand[:, binary_dims]
    X_cand_ord = X_cand[:, ordinal_dims]

    # if cat_dims is not None and cont_dims is None:
    #     for i in range(n_discrete_points):
    #         remaining_budget = length_discrete

    #         # 1. Optionally perturb Ordinal first (costs 1 budget if used)
    #         if remaining_budget > 0 and torch.rand(1).item() < 0.5: # 50% chance to alter ordinal
    #             n_cat = ordinal_config[0]
    #             current_val = int(X_cand_ord[i, 0].item())
    #             step = torch.randint(-1, 2, (1,)).item()
    #             if step != 0:
    #                 new_val = max(0, min(n_cat - 1, current_val + step))
    #                 if new_val != current_val:
    #                     X_cand_ord[i, 0] = new_val
    #                     remaining_budget -= 1

    #         # 2. Spend remaining budget on Binary
    #         if remaining_budget > 0:
    #             bit_change = torch.randint(1, min(remaining_budget, len(binary_dims)) + 1, (1,)).item()
    #             modified_bits = torch.randperm(len(binary_dims))[:bit_change]
    #             for bit in modified_bits:
    #                 X_cand_bin[i, bit.item()] = 1 - X_cand_bin[i, bit.item()]

    #     X_cand[:, binary_dims] = X_cand_bin
    #     X_cand[:, ordinal_dims] = X_cand_ord
    #     return X_cand
    if cat_dims is not None and (cont_dims is None or len(cont_dims) == 0):
        d_bin = len(binary_dims) if binary_dims else 0
        d_ord = len(ordinal_dims) if ordinal_dims else 0
        total_dims = d_bin + d_ord
        
        # 1. Calculate Trust Region Ratio for dynamic stepping
        tr_ratio = float(length_discrete) / float(max(1, total_dims))

        for i in range(n_discrete_points):
            remaining_budget = length_discrete
            ordinal_moved = False

            # 2. Asymmetric Dynamic Ordinal Stepping
            if d_ord > 0 and remaining_budget > 0 and torch.rand(1).item() < 0.5:
                for idx, dim in enumerate(ordinal_dims):
                    n_cat = ordinal_config[idx]
                    current_val = int(X_cand_ord[i, idx].item())
                    
                    # Scale the maximum allowable jump by the TR ratio
                    dynamic_radius = max(1, int(n_cat * tr_ratio))
                    
                    # Asymmetric bounds to prevent probability mass waste
                    left_radius = min(dynamic_radius, current_val)
                    right_radius = min(dynamic_radius, n_cat - 1 - current_val)
                    
                    if left_radius + right_radius > 0:
                        step = torch.randint(-left_radius, right_radius + 1, (1,)).item()
                        if step != 0:
                            X_cand_ord[i, idx] = current_val + step
                            ordinal_moved = True
                            
            if ordinal_moved:
                remaining_budget -= 1

            # 3. Spend remaining budget on Binary features
            if remaining_budget > 0 and d_bin > 0:
                # If ordinal didn't move, force at least 1 bit flip to avoid duplicates
                min_bits = 1 if not ordinal_moved else 0
                max_bits = min(remaining_budget, d_bin)
                
                if max_bits >= min_bits:
                    bit_change = torch.randint(min_bits, max_bits + 1, (1,)).item()
                    if bit_change > 0:
                        modified_bits = torch.randperm(d_bin)[:bit_change]
                        for bit in modified_bits:
                            X_cand_bin[i, bit.item()] = 1 - X_cand_bin[i, bit.item()]

        X_cand[:, binary_dims] = X_cand_bin
        X_cand[:, ordinal_dims] = X_cand_ord
        return X_cand
    if cat_dims is not None and cont_dims is not None:
        # ==========================================
        # 1. CONTINUOUS PERTURBATIONS
        # ==========================================
        d_cont = len(cont_dims)
        
        if d_cont > 0:  # Only perturb if continuous dims exist
            prob_perturb_cont = min(20.0 / d_cont, 1.0) if prob_perturb == min(20.0 / d, 1.0) else prob_perturb
            normalized_tr_bounds_cont = normalized_tr_bounds[:, cont_dims]
            
            if trunc_normal_perturb:
                pert_cont = sample_tr_discrete_points(
                    X_center=X_cand[:, cont_dims], length=length, n_discrete_points=n_discrete_points, qmc=qmc
                )
                pert_cont = torch.min(
                    torch.max(pert_cont, normalized_tr_bounds_cont[0]), normalized_tr_bounds_cont[1]
                )
            elif qmc:
                pert_cont = draw_sobol_samples(
                    bounds=normalized_tr_bounds_cont, n=n_discrete_points, q=1
                ).squeeze(1)
            else:
                pert_cont = torch.rand(
                    n_discrete_points, d_cont, dtype=normalized_tr_bounds.dtype, device=normalized_tr_bounds.device,
                )
                pert_cont = (normalized_tr_bounds_cont[1] - normalized_tr_bounds_cont[0]) * pert_cont + normalized_tr_bounds_cont[0]

            mask_cont = (
                torch.rand(n_discrete_points, d_cont, dtype=normalized_tr_bounds.dtype, device=normalized_tr_bounds.device)
                <= prob_perturb_cont
            )
            ind_cont = (~mask_cont).all(dim=-1).nonzero()
            n_perturb_cont = ceil(d_cont * prob_perturb_cont)
            perturb_mask_cont = torch.zeros(d_cont, dtype=mask_cont.dtype, device=mask_cont.device)
            perturb_mask_cont[:n_perturb_cont].fill_(1)
            
            for idx in ind_cont:
                mask_cont[idx] = perturb_mask_cont[torch.randperm(d_cont, device=normalized_tr_bounds.device)]
                
            X_cand_cont = X_cand[:, cont_dims].clone()
            X_cand_cont[mask_cont] = pert_cont[mask_cont]
            X_cand[:, cont_dims] = X_cand_cont

        # ==========================================
        # 2. CATEGORICAL PERTURBATIONS (CASMOPOLITAN logic)
        # ==========================================
        # d_cat = len(cat_dims)
        # # Bounded bit changes using Hamming distance (length_discrete)
        # bit_change = int(min(max(length_discrete, 1), d_cat))
        # X_cand_cat = X_cand[:, cat_dims].clone()
        
        # for i in range(n_discrete_points):
        #     modified_bits = torch.randperm(d_cat)[:bit_change]
        #     for bit in modified_bits:
        #         n_cat = config[bit.item()]
        #         # # Randomly sample a new categorical configuration state
        #         # X_cand_cat[i, bit.item()] = torch.randint(0, n_cat, (1,)).item()
        #         current_val = int(X_cand_cat[i, bit.item()].item())
                
        #         # Create a list of all valid options EXCEPT the current one
        #         options = [val for val in range(n_cat) if val != current_val]
                
        #         # Pick uniformly from the strictly different options
        #         new_val = options[torch.randint(0, len(options), (1,)).item()]
        #         X_cand_cat[i, bit.item()] = new_val
                
        # # --- EXPLICIT ROUNDING FIX ---
        # # Snap categorical variables to exact integers to kill any float drift
        # X_cand[:, cat_dims] = torch.round(X_cand_cat)

        
        # ==========================================
        # 2. CATEGORICAL PERTURBATIONS (CASMOPOLITAN logic)
        # ==========================================
        d_cat = len(cat_dims)
        max_dist = int(min(max(length_discrete, 1), d_cat))
        X_cand_cat = X_cand[:, cat_dims].clone()
        
        for i in range(n_discrete_points):
            # FIX 1: Pick a random distance to allow LOCAL fine-tuning
            bit_change = torch.randint(1, max_dist + 1, (1,)).item()
            
            modified_bits = torch.randperm(d_cat)[:bit_change]
            for bit in modified_bits:
                n_cat = config[bit.item()]
                current_val = int(X_cand_cat[i, bit.item()].item())
                
                # FIX 2: Strictly force it to pick a DIFFERENT category
                options = [val for val in range(n_cat) if val != current_val]
                new_val = options[torch.randint(0, len(options), (1,)).item()]
                
                X_cand_cat[i, bit.item()] = new_val
                
        # Snap categorical variables to exact integers to kill any float drift
        X_cand[:, cat_dims] = torch.round(X_cand_cat)

    else:
        # ==========================================
        # ORIGINAL MORBO LOGIC (PURE CONTINUOUS)
        # ==========================================
        if trunc_normal_perturb:
            pert = sample_tr_discrete_points(
                X_center=X_cand, length=length, n_discrete_points=n_discrete_points, qmc=qmc
            )
            pert = torch.min(torch.max(pert, normalized_tr_bounds[0]), normalized_tr_bounds[1])
        elif qmc:
            pert = draw_sobol_samples(bounds=normalized_tr_bounds, n=n_discrete_points, q=1).squeeze(1)
        else:
            pert = torch.rand(n_discrete_points, d, dtype=normalized_tr_bounds.dtype, device=normalized_tr_bounds.device)
            pert = (normalized_tr_bounds[1] - normalized_tr_bounds[0]) * pert + normalized_tr_bounds[0]

        mask = (torch.rand(n_discrete_points, d, dtype=normalized_tr_bounds.dtype, device=normalized_tr_bounds.device) <= prob_perturb)
        ind = (~mask).all(dim=-1).nonzero()
        n_perturb = ceil(d * prob_perturb)
        perturb_mask = torch.zeros(d, dtype=mask.dtype, device=mask.device)
        perturb_mask[:n_perturb].fill_(1)
        
        for idx in ind:
            mask[idx] = perturb_mask[torch.randperm(d, device=normalized_tr_bounds.device)]
        X_cand[mask] = pert[mask]

    return X_cand

def get_tr_center(X: Tensor, f_obj: Tensor) -> Tensor:
    r"""Find the best point in the trust region.

    Args:
        X: a `n x d`-dim tensor of points
        f_obj: a `n`-dim tensor of scalarized objective values. In the noiseless,
            setting these can be (scalarized) observed values. In the noisy setting,
            these can be (scalarized) posterior means.
    Returns:
        Tensor: a `1 x d`-dim tensor containing the trust region center point.
    """
    if f_obj.ndim != 1:
        raise BotorchTensorDimensionError(
            f"f_obj must have 1 dimension, got {f_obj.ndim} dimensions."
        )
    return X[f_obj.argmax()].view(1, -1)


# def get_indices_in_hypercube(
#     X_center: Tensor, X: Tensor, length: float, eps: float = 1e-10
# ) -> Tensor:
#     r"""Get indices of observed points inside of trust region.

#     Args:
#         X_center: a `1 x d`-dim tensor containing the trust region center point.
#             `X_center` must be normalized to be within `[0, 1]^d`.
#         X: `n x d`-dim tensor containing all data points collected by this trust region.
#         length: the edge length of the trget_indices_in_hypercubeust region's hypercube.
#         eps: absolute tolerance for evaluating equality (necessary on CUDA).

#     Returns:
#         A `n'`-dim tensor containing the points inside the hypercube.
#     """
#     return ((X - X_center).abs() - length / 2 <= eps).all(dim=1).nonzero().view(-1)

from typing import Optional, List
from torch import Tensor

def get_indices_in_hypercube(
    X_center: Tensor, 
    X: Tensor, 
    length: float, 
    eps: float = 1e-10,
    cat_dims: Optional[List[int]] = None,
    cont_dims: Optional[List[int]] = None,
    length_discrete: Optional[int] = None,
    # --- ADDED TOPOLOGY ARGS ---
    binary_dims: Optional[List[int]] = None,
    ordinal_dims: Optional[List[int]] = None,
    ordinal_config: Optional[List[int]] = None,
    use_log_warp=False
) -> Tensor:
    r"""Get indices of observed points inside of trust region.

    Args:
        X_center: a `1 x d`-dim tensor containing the trust region center point.
        X: `n x d`-dim tensor containing all data points collected by this trust region.
        length: the edge length of the trust region's continuous hypercube.
        eps: absolute tolerance for evaluating equality (necessary on CUDA).
        cat_dims: List of indices for categorical dimensions.
        cont_dims: List of indices for continuous dimensions.
        length_discrete: Maximum Hamming distance for categorical dimensions.

    Returns:
        A `n'`-dim tensor containing the points inside the hypercube.
    """
    eps = 1e-6
    
    # # Fix Bug 4: Proper boolean routing
    # has_cat = cat_dims is not None and len(cat_dims) > 0
    # has_cont = cont_dims is not None and len(cont_dims) > 0

    has_bin = binary_dims is not None and len(binary_dims) > 0
    has_ord = ordinal_dims is not None and len(ordinal_dims) > 0
    has_cont = cont_dims is not None and len(cont_dims) > 0

    # if has_cat and not has_cont and length_discrete is not None:
    #     # 1. Pure Discrete Logic: Hamming distance only
    #     hamming_distances = ((X[:, cat_dims] - X_center[:, cat_dims]).abs() > eps).sum(dim=1)
    #     return (hamming_distances <= length_discrete).nonzero().view(-1)
        
    # elif has_cat and has_cont and length_discrete is not None:
    #     # 2. Mixed Logic (Casmopolitan): L-infinity for continuous, Hamming for discrete
    #     cont_mask = ((X[:, cont_dims] - X_center[:, cont_dims]).abs() - length / 2 <= eps).all(dim=1)
    #     hamming_distances = ((X[:, cat_dims] - X_center[:, cat_dims]).abs() > eps).sum(dim=1)
    #     cat_mask = hamming_distances <= length_discrete
        
    #     in_tr_mask = cont_mask & cat_mask
    #     return in_tr_mask.nonzero().view(-1)
        
    # else:
    #     # 3. Fallback to the original MORBO pure continuous logic
    #     return ((X - X_center).abs() - length / 2 <= eps).all(dim=1).nonzero().view(-1)

    # 1. Pure Discrete Logic: L1 Normalized Distance
    if (has_bin or has_ord) and not has_cont and length_discrete is not None:
        total_dist = torch.zeros(X.shape[0], device=X.device, dtype=X.dtype)
        
        if has_bin:
            total_dist += ((X[:, binary_dims] - X_center[:, binary_dims]).abs() > eps).sum(dim=1).float()
            
        if has_ord:
            for idx, dim in enumerate(ordinal_dims):
                max_range = max(1, ordinal_config[idx] - 1)
                # raw_gap = (X[:, dim] - X_center[:, dim]).abs()
                # total_dist += (raw_gap / max_range).float()
                # Log-warped distance matching the UnifiedL1DiscreteKernel
                max_range = max(1, ordinal_config[idx] - 1)

                if use_log_warp:
                    num_gap = (torch.log1p(X[:, dim]) - torch.log1p(X_center[:, dim])).abs()
                    den_gap = torch.log1p(torch.tensor(max_range, dtype=X.dtype, device=X.device))
                else:
                    num_gap = (X[:, dim] - X_center[:, dim]).abs()
                    den_gap = torch.tensor(max_range, dtype=X.dtype, device=X.device)

                total_dist += (num_gap / den_gap).float()

                
        return (total_dist <= length_discrete + eps).nonzero().view(-1)
        
    # 2. Mixed Logic (Casmopolitan)
    elif (has_bin or has_ord) and has_cont and length_discrete is not None:
        # cont_mask = ((X[:, cont_dims] - X_center[:, cont_dims]).abs() - length / 2 <= eps).all(dim=1)
        
        # total_dist = torch.zeros(X.shape[0], device=X.device, dtype=X.dtype)
        # if has_bin:
        #     total_dist += ((X[:, binary_dims] - X_center[:, binary_dims]).abs() > eps).sum(dim=1).float()
            
        # if has_ord:
        #     for idx, dim in enumerate(ordinal_dims):
        #         max_range = max(1, ordinal_config[idx] - 1)
        #         raw_gap = (X[:, dim] - X_center[:, dim]).abs()
        #         total_dist += (raw_gap / max_range).float()
        
        # cat_mask = total_dist <= length_discrete + eps
        # return (cont_mask & cat_mask).nonzero().view(-1)
        cont_mask = ((X[:, cont_dims] - X_center[:, cont_dims]).abs() - length / 2 <= eps).all(dim=1)
        
        total_dist = torch.zeros(X.shape[0], device=X.device, dtype=X.dtype)
        if has_bin:
            total_dist += ((X[:, binary_dims] - X_center[:, binary_dims]).abs() > eps).sum(dim=1).float()
            
        if has_ord:
            for idx, dim in enumerate(ordinal_dims):
                max_range = max(1, ordinal_config[idx] - 1)
                
                # Log-warped distance matching the UnifiedL1DiscreteKernel
                num_gap = (torch.log1p(X[:, dim]) - torch.log1p(X_center[:, dim])).abs()
                den_gap = torch.log1p(torch.tensor(max_range, dtype=X.dtype, device=X.device))
                
                total_dist += (num_gap / den_gap).float()
        
        cat_mask = total_dist <= length_discrete + eps
        return (cont_mask & cat_mask).nonzero().view(-1)
        
    # 3. Fallback (Original Continuous MORBO)
    else:
        return ((X - X_center).abs() - length / 2 <= eps).all(dim=1).nonzero().view(-1)


def get_fitted_model(
    X: Tensor,
    Y: Tensor,
    use_ard: bool,
    max_cholesky_size: int,
    cat_dims: Optional[List[int]] = None,     # --- ADDED FOR MIXED SPACdef get_fitted_model(E ---
    cont_dims: Optional[List[int]] = None,    # --- ADDED FOR MIXED SPACE ---
    state_dict: Optional[Dict[str, Tensor]] = None,
    input_transform: Optional[InputTransform] = None,
    outcome_transform: Optional[OutcomeTransform] = None,
    fit_gpytorch_options: Optional[Dict[str, Any]] = None,
    # ADD THESE THREE LINES:
    binary_dims: Optional[List[int]] = None,
    ordinal_dims: Optional[List[int]] = None,
    ordinal_config: Optional[List[int]] = None,
    use_log_warp: bool = False,
    use_unified_kernel: bool = True,
    use_mixture_kernel: bool = False,
    use_casmo_mixed_kernel: bool = False,
    use_asymmetric_kernel: bool = False,
) -> Model:
    # --- ADD THESE TWO LINES AT THE VERY TOP ---
    print("Fitting a model")
    use_fast_mvms = True if X.shape[0] > max_cholesky_size else False
    with gpytorch_settings.fast_computations(
        log_prob=use_fast_mvms,
        covar_root_decomposition=use_fast_mvms,
        solves=use_fast_mvms,
    ):
        
        X = X.to(dtype=torch.float64)
        Y = Y.to(dtype=torch.float64)
        # --- ADD THIS BLOCK: Strip discrete duplicates ---
        # Convert to numpy to easily find unique rows, then re-apply to tensors
        X_np = X.cpu().numpy()
        _, unique_idx = np.unique(X_np, axis=0, return_index=True)
        unique_idx = np.sort(unique_idx)
        X = X[unique_idx]
        Y = Y[unique_idx]
        # -------------------------------------------------

        # ---------------------------------------------------------
        # BUG FIX: STRICT INPUT TRANSFORM ROUTING
        # ---------------------------------------------------------
        is_pure_discrete = cont_dims is None or len(cont_dims) == 0

        if is_pure_discrete:
            # Forcefully override upstream transforms. Discrete spaces MUST stay raw.
            input_transform = None
        elif input_transform is None:
            # Fallback for continuous/mixed spaces if no transform was passed
            from botorch.models.transforms.input import Normalize
            input_transform = Normalize(d=X.shape[-1], indices=cont_dims)

        models = []
        for i in range(Y.shape[-1]):
            # ard_num_dims = X.shape[-1] if use_ard else 1
            # covar_module = ScaleKernel(
            #     MaternKernel(
            #         nu=2.5,
            #         ard_num_dims=ard_num_dims,
            #         lengthscale_constraint=Interval(0.05, 4.0),
            #     ),
            # )
            # ---------------------------------------------------------
            # MIXED-VARIABLE SURROGATE MODELING (CASMOPOLITAN + MORBO)
            # ---------------------------------------------------------
            # if cat_dims is not None and cont_dims is not None:
            # Fix Bug 5
    
            # 1. PURE DISCRETE (Your New Architecture)
            # if binary_dims and not cont_dims:
            #     # covar_module = ScaleKernel(
            #     # UnifiedDiscreteKernel(
            #     #     binary_dims=binary_dims,
            #     #     ordinal_dims=ordinal_dims,
            #     #     ordinal_config=ordinal_config,
            #     # )
            #     # Safely initialize everything to 1.0 (binary default)
            #     total_dims = len(binary_dims) + len(ordinal_dims)
            #     unified_config = [1.0] * total_dims
                
            #     # Precisely overwrite the exact column indices for ordinal variables
            #     # if ordinal_dims and ordinal_config:
            #     #     for idx, dim in enumerate(ordinal_dims):
            #     #         unified_config[dim] = max(1, ordinal_config[idx] - 1)

            #     for idx, dim in enumerate(ordinal_dims):
            #         n_binary = len(binary_dims)                              # 66
            #         unified_config[dim] = max(1, (ordinal_config[idx] - 1) // n_binary)
            #         # = max(1, 374324 // 66) = 5671

            #     # Reorder to match global dimension indices (binary_dims + ordinal_dims)
            #     # covar_module = ScaleKernel(
            #     #     UnifiedL1DiscreteKernel(config=unified_config),
            #     #     # ADD THIS LINE:
            #     #     lengthscale_constraint=gpytorch.constraints.Interval(0.01, 2.5),
            #     # )
            #     covar_module = ScaleKernel(
            #         UnifiedL1DiscreteKernel(
            #             config=unified_config,
            #             lengthscale_constraint=gpytorch.constraints.Interval(0.01, 2.5),
            #         )
            #     )

            # 1. PURE DISCRETE (Your New Architecture)
            # if binary_dims and not cont_dims:
            #     n_binary = len(binary_dims)
            #     total_dims = n_binary + len(ordinal_dims)

            #     if i == 0:
            #         # F1: depth's effect is conditional on which features were picked.
            #         # Genuine interaction → keep the single combined product kernel.
            #         ls_max_ord_f1 = float(len(binary_dims))   # 66.0 — scale-invariant

            #         unified_config = [1.0] * total_dims
            #         if ordinal_dims and ordinal_config:
            #             for idx, dim in enumerate(ordinal_dims):
            #                 unified_config[dim] = max(1, ordinal_config[idx] - 1)
                            
            #         covar_module = ScaleKernel(
            #             UnifiedL1DiscreteKernel(
            #                 config=unified_config,
            #                 use_log_warp=use_log_warp,
            #                 lengthscale_constraint=Interval(0.01, ls_max_ord_f1),
            #             )
            #         )
            #     else:
            #         # Compute cost: depth carries an independent, near-fixed marginal
            #         # penalty regardless of feature selection. Split binary/ordinal into
            #         # separate kernels so a depth match alone preserves correlation even
            #         # when the binary block is completely different.
            #         binary_kern = UnifiedL1DiscreteKernel(
            #             config=[1.0] * n_binary,
            #             active_dims=binary_dims,
            #             use_log_warp=use_log_warp,
            #             lengthscale_constraint=Interval(0.01, 2.5),
            #         )
            #         ordinal_kern = UnifiedL1DiscreteKernel(
            #             config=[max(1, c - 1) for c in ordinal_config] if ordinal_config else [1.0],
            #             active_dims=ordinal_dims,
            #             use_log_warp=use_log_warp,
            #             # Force the GP to assign a minimum penalty to ordinal distances
            #             lengthscale_constraint=Interval(0.01, 2.5),
            #             # lengthscale_constraint=Interval(1e-5, 2.5)
            #         )
            #         covar_module = ScaleKernel(binary_kern) + ScaleKernel(ordinal_kern)
            # APPLY THE CASMOPOLITAN MIXED KERNEL HERE:
            if use_casmo_mixed_kernel and cat_dims is not None and cont_dims is not None:
                print("Mixed Arch. using integer with continuous kernel")

                covar_module = ScaleKernel(
                    MixtureKernel(
                        categorical_dims=cat_dims,
                        continuous_dims=cont_dims,
                        integer_dims=cont_dims, # Enforce WrappedMatern rounding on the continuous packet depth
                        categorical_ard=use_ard,
                        continuous_ard=use_ard,
                    )
                )
            # 1. PURE DISCRETE (Your Custom Architectures)
            elif binary_dims and not cont_dims:
                print("Pure Discrete Arch")
                n_binary = len(binary_dims)
                total_dims = n_binary + len(ordinal_dims)
                ls_max_ord = float(n_binary)
                # ---------------------------------------------------------
                # PATH 1: The Domain-Aware Asymmetric Baseline (Your Original Intent)
                # F1 (i==0) gets Unified/Interactive, Cost (i==1) gets Additive
                # ---------------------------------------------------------
                if use_asymmetric_kernel:
                    print("Diff interaction between seperate kernels for binary and ordinal per objective")
                    if i == 0:
                        # F1 SCORE: Highly Interactive (Unified Product Kernel)
                        unified_config = [1.0] * total_dims
                        if ordinal_dims and ordinal_config:
                            for idx, dim in enumerate(ordinal_dims):
                                unified_config[dim] = max(1, ordinal_config[idx] - 1)
                                
                        covar_module = ScaleKernel(
                            UnifiedL1DiscreteKernel(
                                config=unified_config,
                                use_log_warp=use_log_warp,
                                lengthscale_constraint=Interval(0.01, ls_max_ord),
                            )
                        )
                    else:
                        # COMPUTE COST: Largely Independent (Additive Kernel)
                        binary_kern = UnifiedL1DiscreteKernel(
                            config=[1.0] * n_binary,
                            active_dims=binary_dims,
                            lengthscale_constraint=Interval(0.01, 2.5),
                        )
                        ordinal_kern = UnifiedL1DiscreteKernel(
                            config=[max(1, c - 1) for c in ordinal_config] if ordinal_config else [1.0],
                            active_dims=ordinal_dims,
                            use_log_warp=use_log_warp,
                            lengthscale_constraint=Interval(0.01, 2.5),
                        )
                        covar_module = ScaleKernel(binary_kern) + ScaleKernel(ordinal_kern)
                        
                # PATH A: Mixture Kernel (Learnable Lambda)
                elif use_mixture_kernel:
                    print("learnable lambda interaction between seperate kernels for binary and ordinal ")
                    covar_module = ScaleKernel(
                        DiscreteMixtureKernel(
                            binary_dims=binary_dims,
                            ordinal_dims=ordinal_dims,
                            ordinal_config=ordinal_config,
                            use_log_warp=use_log_warp,
                            lengthscale_constraint=Interval(0.01, 2.5),
                        )
                    )
                
                # PATH B: Unified Product Kernel (Strong Interactions)
                elif use_unified_kernel:
                    print("only one kernel for binary and ordinal ")

                    unified_config = [1.0] * total_dims
                    if ordinal_dims and ordinal_config:
                        for idx, dim in enumerate(ordinal_dims):
                            unified_config[dim] = max(1, ordinal_config[idx] - 1)
                            
                    covar_module = ScaleKernel(
                        UnifiedL1DiscreteKernel(
                            config=unified_config,
                            use_log_warp=use_log_warp,
                            lengthscale_constraint=Interval(0.01, ls_max_ord),
                        )
                    )
                
                # PATH C: Independent Additive Kernels (No Interactions)
                else:
                    print("only additive interaction between seperate kernels for binary and ordinal ")

                    binary_kern = UnifiedL1DiscreteKernel(
                        config=[1.0] * n_binary,
                        active_dims=binary_dims,
                        lengthscale_constraint=Interval(0.01, 2.5),
                    )
                    ordinal_kern = UnifiedL1DiscreteKernel(
                        config=[max(1, c - 1) for c in ordinal_config] if ordinal_config else [1.0],
                        active_dims=ordinal_dims,
                        use_log_warp=use_log_warp,
                        lengthscale_constraint=Interval(0.01, 2.5),
                    )
                    covar_module = ScaleKernel(binary_kern) + ScaleKernel(ordinal_kern)
           
            likelihood = GaussianLikelihood(
                # Relaxed from 1e-6 to prevent Cholesky failures in clustered discrete spaces
                noise_constraint=GreaterThan(1e-4), 
                noise_prior=GammaPrior(0.9, 10.0),
            )
            
            model = SingleTaskGP(
                train_X=X,
                train_Y=Y[:, i : i + 1],
                covar_module=covar_module,
                likelihood=likelihood,
                outcome_transform=outcome_transform.subset_output([i])
                if outcome_transform
                else None,
                input_transform=input_transform,
            )
            models.append(model)

        # TODO: replaced with batched-MO model once MTMVN refactor
        # lands: https://github.com/cornellius-gp/gpytorch/pull/1083
        if Y.shape[-1] > 1:
            model = ModelListGP(*models)
            mll = SumMarginalLogLikelihood(model.likelihood, model)
        else:
            model = models[0]
            mll = ExactMarginalLogLikelihood(model.likelihood, model)

        # if state_dict is not None:
        #     model.load_state_dict(state_dict)
        # # 50 iterations appears to be a good compromise between fit and overhead.
        # fit_gpytorch_mll(mll, options=fit_gpytorch_options)

        from botorch.exceptions.errors import ModelFittingError
        import logging

        if state_dict is not None:
            # for any F1 level between 0.87 and 0.96
            model.load_state_dict(state_dict)
            
        try:
            # Attempt standard L-BFGS-B optimization
            fit_gpytorch_mll(mll, options=fit_gpytorch_options)
        except ModelFittingError as e:
            logging.warning(f"L-BFGS-B failed (Likely ill-conditioned TR matrix): {e}. "
                            "Skipping hyperparameter update for this batch step to preserve BO loop.")
            # If the surface is too flat/singular to fit, skipping the update is mathematically safe.
            # The GP will simply fall back to its initialized hypers (or the state_dict hypers from the previous step),
            # allowing the TS exploration to continue without crashing the 500-iteration experiment.

    if X.is_cuda:
        print(f"after fitting: {torch.cuda.memory_allocated(X.device) / (1000 ** 3)}")
    model = model.to(dtype=torch.float64)
    return model


def coalesce(x1: Optional[Tensor], x2: Optional[Tensor]) -> Optional[Tensor]:
    r"""Helper function the performs a coalesce operation.

    If x1 is not None, it is returned. Otherwise x2 is returned.

    Args:
        x1: a tensor
        x2 a tensor

    Returns:
        A tensor if either of x1 or x2 is not None, otherwise None.
    """
    if x1 is None:
        x1 = x2
    return x1


def decay_function(n: int, n0: int, n_max: int, alpha: float = 1.0) -> float:
    r"""Decay function governed by the used and remaining optimization budget.

    Decay function from:
        Regis R.G., Shoemaker C.A. Combining radial basis function
        surrogates and dynamic coordinate search in high-dimensional
        expensive black-box optimization. Engineering Optimization, 45
        (5) (2013), pp. 529-555

    Args:
        n: number of completed function evaluations
        n0: number of initial function evaluations
        n_max: maximum number of function evaluations (budget)
        alpha: hyperparameter controlling decay

    Returns:
        The probabilty of perturbing a dimension.
    """
    return 1 - alpha * log(n - n0 + 1) / log(n_max - n0 + 1)


def get_constraint_slack_and_feasibility(
    Y: Tensor, constraints: List[Callable[[Tensor], Tensor]]
) -> Tensor:
    r"""Compute feasibility.

    Args:
        Y: A `batch_shape x n x m`-dim tensor of outcomes
        constraints: A list of constraint callables mapping outcomes to the
            constraint slack.

    Returns:
        A `batch_shape x n`-dim boolean tensor indicating whether each example in Y
            is feasible.
    """
    constraint_slack = torch.stack([c(Y) for c in constraints], dim=-1)
    return constraint_slack, (constraint_slack <= 0).all(dim=-1)


# Add to the bottom of morbo/utils.py

from botorch.utils.transforms import unnormalize, normalize

def safe_unnormalize(X, bounds, indices=None):
    """Safely unnormalize only the specified continuous indices."""
    if indices is None:
        return unnormalize(X, bounds)
    
    X_out = X.clone()
    # Apply BoTorch's unnormalize strictly to the continuous dimensions
    X_out[..., indices] = unnormalize(X[..., indices], bounds[:, indices])
    return X_out

def safe_normalize(X, bounds, indices=None):
    """Safely normalize only the specified continuous indices."""
    if indices is None:
        return normalize(X, bounds)
    
    X_out = X.clone()
    # Apply BoTorch's normalize strictly to the continuous dimensions
    X_out[..., indices] = normalize(X[..., indices], bounds[:, indices])
    return X_out