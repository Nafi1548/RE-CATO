from botorch.sampling.base import MCSampler
#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import math
import time
from typing import Callable, NamedTuple, Tuple

import gpytorch
import torch
from botorch.acquisition.multi_objective.monte_carlo import (
    qExpectedHypervolumeImprovement,
)
from botorch.models.deterministic import GenericDeterministicModel
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.sampling import IIDNormalSampler, SobolQMCNormalSampler
from botorch.utils.gp_sampling import get_gp_samples
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.multi_objective.box_decompositions.box_decomposition import (
    BoxDecomposition,
)
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    NondominatedPartitioning,
    FastNondominatedPartitioning,
)
from botorch.utils.sampling import sample_simplex
from botorch.utils.transforms import normalize, unnormalize
# Top of gen.py
from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement
from botorch.sampling.normal import SobolQMCNormalSampler
from morbo.discrete_ts_batch import discrete_ts_batch_select

from morbo.state import TRBOState
from morbo.utils import (
    decay_function,
    get_indices_in_hypercube,
    sample_tr_discrete_points,
    sample_tr_discrete_points_subset_d,
    safe_unnormalize, 
    safe_normalize, # <-- ADD THIS
    sample_tr_pure_discrete_subset,
)

# Add this near the top of gen.py
# from morbo.interleaved_ts_hvi import patch_gen_py_candidate_generation

# REMOVE this line (the old patch):
# from morbo.interleaved_ts_hvi import patch_gen_py_candidate_generation

# ADD this line:
from morbo.discrete_ts_batch import discrete_ts_batch_select


from torch import Tensor
from torch.quasirandom import SobolEngine

class DeterministicSampler(MCSampler):
    """A dummy sampler for deterministic posteriors that ignores base samples."""
    def forward(self, posterior, **kwargs):
        # Simply expand the deterministic values to match the requested sample shape
        return posterior.rsample(sample_shape=self.sample_shape)





class CandidateSelectionOutput(NamedTuple):
    X_cand: Tensor
    tr_indices: Tensor


def get_partitioning(
    trbo_state: TRBOState, ref_point: Tensor, Y: Tensor
) -> BoxDecomposition:
    """Helper method for constructing a box decomposition"""
    if trbo_state.tr_hparams.use_approximate_hv_computations:
        alpha = (
            trbo_state.tr_hparams.approximate_hv_alpha
            if trbo_state.tr_hparams.approximate_hv_alpha is not None
            else get_default_partitioning_alpha(trbo_state.num_objectives)
        )
        partitioning = NondominatedPartitioning(ref_point=ref_point, Y=Y, alpha=alpha)
    else:
        partitioning = FastNondominatedPartitioning(ref_point=ref_point, Y=Y)
    return partitioning


def _make_unstandardizer(Y_mean: Tensor, Y_std: Tensor) -> Callable[[Tensor], Tensor]:
    def unstandardizer(Y: Tensor) -> Tensor:
        return Y * Y_std + Y_mean

    return unstandardizer


def preds_and_feas(
    trbo_state: TRBOState, tr_idx: int, X: Tensor
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Compute model predictions and constraint violations."""
    tkwargs = {"device": trbo_state.bounds.device, "dtype": trbo_state.bounds.dtype}
    tr = trbo_state.trust_regions[tr_idx]
    objective = tr.objective
    model = trbo_state.models[tr_idx]
    preds, dists = model.get_predictions_and_distances(X)
    # apply objective
    f_obj = objective(preds).clone()

    if trbo_state.constraints is not None:
        constraint_value = torch.stack(
            [c(preds) for c in trbo_state.constraints], dim=-1
        )
        feas = (constraint_value <= 0.0).all(dim=-1)
        violation = torch.clamp(constraint_value, 0.0).sum(dim=-1)
    else:
        feas = torch.ones(len(f_obj), device=tkwargs["device"], dtype=torch.bool)
        violation = torch.zeros(len(f_obj), **tkwargs)
    return f_obj, feas, violation, dists


def unit_rescale(x: Tensor) -> Tensor:
    """Helper function for normalizing a 1D input to [0, 1]."""
    if not x.dim() == 1:
        raise RuntimeError(f"Expected a 1D input, got shape: {list(x.shape)}")
    if x.min() == x.max():
        return 0.5 * torch.ones(x.shape, dtype=x.dtype, device=x.device)
    return (x - x.min()) / (x.max() - x.min())

