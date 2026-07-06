"""
interleaved_ts_hvi.py
=====================
Replaces random candidate sampling in gen.py with CASMOPOLITAN's interleaved
local search, guided by Thompson Sampling + HVI (MORBO Section 3.1).

What this is NOT
----------------
It is NOT random scalarization.
Random scalarization (the approach MORBO Figure 2 rejects) converts the
multi-objective problem into a single scalar using random weights:
    score(x) = min_m(y_m / λ_m)   or   mean_m + √β * std_m
Both of those collapse the objectives before evaluating them.

What this IS
------------
Each restart r draws one ε_r ~ N(0, I_M) at the very start and holds it
fixed for the duration of that restart. For every neighbor x encountered
during the search, the TS "function value" is:

    f̃_r(x)_m = μ_m(x) + σ_m(x) · ε_r_m     for each objective m

This gives a VECTOR of objective values — not a scalar. It is then scored
using HVI improvement over the current Pareto front:

    acquisition(x) = HVI(pareto ∪ {f̃_r(x)}) - HVI(pareto)

HVI is the same criterion used in MORBO Section 3.1. It rewards Pareto-diverse
improvements because the HVI of a dominated point is 0, and the HVI of a
point in a crowded Pareto region is small.

Because ε_r is fixed throughout one restart, the acquisition landscape is
stationary — greedy ascent converges rather than oscillating. Different
restarts use different ε_r, producing diverse candidates without any
scalarization.

Alignment
---------
  CASMOPOLITAN: interleaved_search structure, Hamming + L∞ TR boundary,
                greedy neighbor move with acquisition gating.
  MORBO: HVI as acquisition value, same gen.py GenericDeterministicModel
         + DeterministicSampler pipeline for final batch scoring.

The only change to the existing gen.py TS path is:
    BEFORE  →  X_cand = sample_tr_pure_discrete_subset(...)
    AFTER   →  X_cand = interleaved_ts_hvi_candidates(...)
Everything after line 320 in gen.py (the TS scoring block) is untouched.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import List, Optional, Tuple

import numpy as np
import torch
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core: TS + HVI acquisition function for one restart
# ---------------------------------------------------------------------------

class TSHVIAcquisition:
    """
    A single-restart Thompson Sampling + HVI acquisition function.

    At construction time, one epsilon vector is sampled:
        ε ~ N(0, I_M)    where M = number of objectives

    For each neighbor x encountered during the local search, the TS draw is:
        f̃_m(x) = μ_m(x) + σ_m(x) · ε_m     for each objective m

    The acquisition value is the marginal HVI contribution of f̃(x):
        a(x) = HVI(pareto_Y ∪ {f̃(x)}) - HVI(pareto_Y)

    This is positive if f̃(x) is non-dominated and contributes new Pareto
    volume, and zero (or negative, clamped to 0) otherwise.

    Because ε is fixed throughout this restart, a(x) defines a stationary
    landscape that greedy local search can follow consistently.
    """

    def __init__(
        self,
        model,                           # Fitted multi-output BoTorch GP model
        pareto_Y: torch.Tensor,          # (n_pareto, M) current Pareto front
        ref_point: torch.Tensor,         # (M,) reference point, maximisation
        tkwargs: dict,
        min_hvi_to_move: float = 0.0,    # move threshold (0 = any improvement)
    ):
        self.model = model
        self.ref_point = ref_point.to(**tkwargs)
        self.tkwargs = tkwargs
        self.min_hvi_to_move = min_hvi_to_move

        # Number of objectives from the pareto front shape or model
        self.M = pareto_Y.shape[-1] if pareto_Y.numel() > 0 else model.num_outputs

        # Draw epsilon ONCE for this restart — held fixed throughout
        self.epsilon = torch.randn(self.M, **tkwargs)  # (M,)

        # Build the hypervolume partitioning for the CURRENT Pareto front
        # This is the baseline HV that all candidates are scored against
        pareto_Y_t = pareto_Y.to(**tkwargs)
        dominated_by_ref = (pareto_Y_t <= self.ref_point).any(dim=-1)
        pareto_Y_above_ref = pareto_Y_t[~dominated_by_ref]

        if pareto_Y_above_ref.numel() > 0:
            self._partitioning = FastNondominatedPartitioning(
                ref_point=self.ref_point,
                Y=pareto_Y_above_ref,
            )
            self._baseline_hv = self._partitioning.compute_hypervolume().item()
        else:
            self._partitioning = None
            self._baseline_hv = 0.0

        # Cache: avoids re-computing GP posterior for the same point within a restart
        self._cache: dict = {}

    def __call__(self, x) -> torch.Tensor:
        """
        Compute HVI of x under this restart's TS draw.

        Parameters
        ----------
        x : numpy array (d,) or torch.Tensor (d,) or (1, d)

        Returns
        -------
        torch.Tensor scalar — HVI improvement, >= 0.
        """
        # ---- Normalise input ----
        if isinstance(x, np.ndarray):
            x_arr = x
        else:
            x_arr = x.detach().cpu().numpy()
        x_arr = x_arr.flatten()

        # ---- Cache key (discrete space: round to integer) ----
        key = tuple(x_arr.round(4).tolist())

        if key not in self._cache:
            # ---- GP marginal posterior at x ----
            with torch.no_grad():
                x_t = torch.tensor(x_arr, **self.tkwargs).unsqueeze(0)  # (1, d)
                posterior = self.model.posterior(x_t)

                # Use the model's outcome_transform to untransform the posterior
                # This ensures mean/std are in the RAW objective space
                posterior = self.model.outcome_transform.untransform_posterior(posterior)
                
                mean = posterior.mean.squeeze(0)       # (M,)
                variance = posterior.variance.squeeze(0)  # (M,)

            std = variance.clamp_min(1e-12).sqrt()    # (M,)

            # ---- TS draw using the fixed epsilon ----
            f_ts = mean + std * self.epsilon           # (M,)

            # ---- Marginal HVI of f_ts over the current Pareto baseline ----
            hvi_val = self._compute_marginal_hvi(f_ts)

            self._cache[key] = torch.tensor(hvi_val, **self.tkwargs)

        return self._cache[key]

    def _compute_marginal_hvi(self, f_ts: torch.Tensor) -> float:
        """
        HVI(pareto ∪ {f_ts}) - baseline_hv.

        If f_ts is dominated by the reference point or contributes nothing
        new to the Pareto front, returns 0.
        """
        # Candidate must dominate the reference point to contribute any HVI
        if (f_ts <= self.ref_point).any():
            return 0.0

        if self._partitioning is None:
            # No existing Pareto front — any point above ref contributes HVI
            # Build a fresh partitioning with just this point
            try:
                p = FastNondominatedPartitioning(
                    ref_point=self.ref_point,
                    Y=f_ts.unsqueeze(0),
                )
                return p.compute_hypervolume().item()
            except Exception:
                return 0.0

        # Temporarily add f_ts to the partitioning and compute new HV
        try:
            current_Y = self._partitioning.Y  # (n_pareto, M)
            combined_Y = torch.cat([current_Y, f_ts.unsqueeze(0)], dim=0)
            p_new = FastNondominatedPartitioning(
                ref_point=self.ref_point,
                Y=combined_Y,
            )
            new_hv = p_new.compute_hypervolume().item()
            return max(0.0, new_hv - self._baseline_hv)
        except Exception:
            return 0.0


# ---------------------------------------------------------------------------
# Multi-restart interleaved search: one TS draw per restart
# ---------------------------------------------------------------------------

def interleaved_ts_hvi_candidates(
    model,
    trbo_state,
    tr,
    cat_dims: List[int],
    cont_dims: List[int],
    config,
    n_restarts: int = 5,
    step: int = 200,
    interval: int = 1,
    tkwargs: Optional[dict] = None,
) -> torch.Tensor:
    """
    Generate candidates via CASMOPOLITAN interleaved search, guided by TS+HVI.

    For each of the n_restarts restarts:
      1. Construct a fresh TSHVIAcquisition (draws one new ε_r ~ N(0, I_M))
      2. Run interleaved_search using TSHVIAcquisition as the acquisition fn
      3. Collect the terminal point x_r*

    The returned tensor X_cand (n_restarts, d) is a drop-in replacement for
    the output of sample_tr_pure_discrete_subset in gen.py. Everything after
    that point — the joint TS draw + GenericDeterministicModel + HVI scoring —
    proceeds exactly as before.

    Parameters
    ----------
    model : fitted multi-output GP (one output per objective)
    trbo_state : TRBOState
    tr : trust region object (provides center, length_discrete, bounds)
    cat_dims : list of categorical/discrete dimension indices
    cont_dims : list of continuous dimension indices (empty for pure discrete)
    config : ordinal config for CASMOPOLITAN (list of category counts per dim)
    n_restarts : number of independent search restarts
    step : max steps per restart (interleaved_search `step` parameter)
    interval : steps between cont/cat switching (interleaved_search parameter)
    tkwargs : {'dtype': ..., 'device': ...}

    Returns
    -------
    X_cand : torch.Tensor, shape (n_restarts, d)
        One candidate per restart, in raw unnormalised discrete space.
        Ready to be passed to the existing TS+HVI scoring block in gen.py.
    """
    tkwargs = tkwargs or {"dtype": torch.float32, "device": "cpu"}

    from morbo.localbo_utils import interleaved_search

    # ---- TR geometry ----
    x_center = tr.X_center_normalized.squeeze(0).cpu().numpy()   # (d,)
    length_discrete = int(tr.length_discrete.item()) if hasattr(tr, "length_discrete") else len(cat_dims)

    if len(cont_dims) > 0:
        bounds = tr.bounds.cpu().numpy()   # (2, d)
        lb = bounds[0, cont_dims]
        ub = bounds[1, cont_dims]
    else:
        lb = np.array([])
        ub = np.array([])

    # ---- Current Pareto front for HVI baseline ----
    pareto_Y = trbo_state.pareto_Y_better_than_ref.clone()   # (n_pareto, M)
    ref_point = trbo_state.ref_point.clone()

    # ---- Apply objective transform to Pareto front ----
    objective = tr.objective
    pareto_Y_obj = objective(pareto_Y)                        # (n_pareto, M) after transform

    # ---- Run n_restarts independent local searches ----
    X_collected: List[np.ndarray] = []

    for r in range(n_restarts):
        # Fresh acquisition function with a new independent ε for this restart
        acq_fn = TSHVIAcquisition(
            model=model,
            pareto_Y=pareto_Y_obj,
            ref_point=ref_point,
            tkwargs=tkwargs,
        )

        # Run CASMOPOLITAN interleaved search with this acquisition function.
        # n_restart=1 here — we manage the outer restart loop ourselves so
        # each call gets a fresh ε. batch_size=1 returns the single terminal pt.
        X_r, acq_r = interleaved_search(
            x_center=x_center,
            f=acq_fn,
            cat_dims=cat_dims,
            cont_dims=cont_dims,
            config=config,
            ub=ub,
            lb=lb,
            max_hamming_dist=length_discrete,
            n_restart=1,
            batch_size=1,
            interval=interval,
            step=step,
        )
        X_collected.append(X_r[0])   # (d,) terminal point

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Restart %d/%d: terminal HVI=%.4f, ε_norm=%.3f",
                r + 1, n_restarts,
                float(acq_r[0]),
                float(acq_fn.epsilon.norm()),
            )

    X_cand_np = np.stack(X_collected, axis=0)   # (n_restarts, d)

    # Deduplicate — discrete search can produce identical terminal points
    X_cand_np = _deduplicate(X_cand_np, cat_dims)

    X_cand = torch.tensor(X_cand_np, **tkwargs)   # (n_unique, d)

    logger.info(
        "interleaved_ts_hvi: %d restarts → %d unique candidates (length_discrete=%d)",
        n_restarts, len(X_cand), length_discrete,
    )

    return X_cand


# ---------------------------------------------------------------------------
# gen.py patch: surgical replacement of the candidate generation block
# ---------------------------------------------------------------------------

def patch_gen_py_candidate_generation(
    trbo_state,
    tr,
    tr_idx: int,
    model,
    n_restarts: int = 5,
    step: int = 200,
    interval: int = 1,
    tkwargs: Optional[dict] = None,
) -> torch.Tensor:
    """
    Drop-in replacement for lines 308–320 of gen.py:

    BEFORE (gen.py lines 308–320):
        X_cand_unnormalized = sample_tr_pure_discrete_subset(
            best_X=best_X,
            n_discrete_points=trbo_state.tr_hparams.raw_samples,
            length_discrete=int(tr.length_discrete.item()),
            binary_dims=trbo_state.tr_hparams.binary_dims,
            ordinal_dims=trbo_state.tr_hparams.ordinal_dims,
            ordinal_config=trbo_state.tr_hparams.ordinal_config
        )
        X_cand = X_cand_unnormalized.clone()

    AFTER (replace with this function call):
        X_cand_unnormalized = patch_gen_py_candidate_generation(
            trbo_state=trbo_state, tr=tr, tr_idx=tr_idx, model=model,
            n_restarts=n_restarts, step=step, interval=interval, tkwargs=tkwargs
        )
        X_cand = X_cand_unnormalized.clone()

    Everything after line 320 in gen.py — the entire TS scoring block
    (joint posterior sample → GenericDeterministicModel → DeterministicSampler
    → qExpectedHypervolumeImprovement → HVI value_score) — is UNCHANGED.
    This function only changes WHERE the candidates come from.

    Why the TS scoring block after line 320 is still correct
    ---------------------------------------------------------
    The existing block draws a JOINT posterior sample over X_cand_unnormalized
    and scores by HVI. With CASMOPOLITAN candidates, this joint sample is
    drawn over fewer, higher-quality points (interleaved search finds deep
    local optima) rather than many random points. The HVI scoring is still
    MORBO-correct — no scalarization anywhere.
    """
    tkwargs = tkwargs or {"dtype": torch.float32, "device": "cpu"}

    # Extract raw dimension lists from the state
    binary_dims = getattr(trbo_state.tr_hparams, "binary_dims", None) or []
    ordinal_dims = getattr(trbo_state.tr_hparams, "ordinal_dims", None) or []
    cont_dims = getattr(trbo_state.tr_hparams, "cont_dims", None) or []
    ordinal_config = getattr(trbo_state.tr_hparams, "ordinal_config", None) or []


    # --- THE FIX: Unified Discrete Mapping ---
    # Casmopolitan requires a 1-to-1 mapping of discrete dimensions to choice counts.
    # Binary variables have 2 choices; ordinal variables have N choices.
    cat_dims = binary_dims + ordinal_dims
    config = [2] * len(binary_dims) + ordinal_config


    return interleaved_ts_hvi_candidates(
        model=model,
        trbo_state=trbo_state,
        tr=tr,
        cat_dims=cat_dims,
        binary_dims = binary_dims,
        ordinal_dims=ordinal_dims,
        ordinal_config = ordinal_config,
        cont_dims=cont_dims,
        config=config,
        n_restarts=n_restarts,
        step=step,
        interval=interval,
        tkwargs=tkwargs,
    )


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _deduplicate(X: np.ndarray, cat_dims: List[int]) -> np.ndarray:
    """Remove exact duplicate rows (integer comparison on discrete dims)."""
    if X.shape[0] <= 1 or not cat_dims:
        return X
    X_int = X[:, cat_dims].round().astype(np.int32)
    seen = set()
    keep = []
    for i, row in enumerate(X_int):
        key = tuple(row.tolist())
        if key not in seen:
            seen.add(key)
            keep.append(i)
    return X[keep]