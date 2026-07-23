#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from __future__ import annotations

from morbo.localbo_utils import interleaved_search
import numpy as np
from morbo.utils import safe_unnormalize, safe_normalizesafe_normalize

import dataclasses
import math
import time
from enum import Enum
from typing import Callable, Dict, List, Optional, OrderedDict, Tuple, Union

import torch
import gpytorch
from botorch.acquisition.multi_objective.objective import IdentityMCMultiOutputObjective
from botorch.acquisition.objective import IdentityMCObjective, MCAcquisitionObjective
from botorch.exceptions import BotorchError
from botorch.models.model import Model
from botorch.models.transforms.input import (
    ChainedInputTransform,
    Normalize,
)
from botorch.models.transforms.outcome import Standardize
from botorch.utils.constraints import get_outcome_constraint_transforms
from botorch.utils.multi_objective.box_decompositions.box_decomposition_list import (
    BoxDecompositionList,
)
from botorch.utils.multi_objective.box_decompositions.dominated import (
    DominatedPartitioning,
)
from botorch.utils.multi_objective.hypervolume import infer_reference_point
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.sampling import (
    sample_hypersphere,
    draw_sobol_samples,
    sample_simplex,
)
from botorch.utils.transforms import unnormalize
from morbo.trust_region import (
    HypervolumeTrustRegion,
    ScalarizedTrustRegion,
    TrustRegion,
    TurboHParams,
)
from morbo.utils import get_fitted_model, coalesce, decay_function, sample_tr_pure_discrete_subset
from scipy.stats.mstats import winsorize
from torch import Tensor
from torch.nn import Module, ModuleList


class TRGenStatus(Enum):
    NONE = 0
    INITIALIZE = 1
    REINITIALIZE = 2
    TS = 4


class TabuSet(Module):
    """A container for a set of Tabu pareto points.

    This class tracks points that were the center of a trust region when it was
    terminated due to the edge length being too small. These points are deemed
    to be throughly explored and therefore there would be a low chance of
    improvement from using that point as the center for some period of time
    (`tabu_tenure`).

    This is particularly improvement when doing pareto optimization and selecting
    centers based on hypervolume contribution because centers are selected greedily.
    Without a tabu set, such a strategy could cause a terminated TR to immediately
    select the same center point when it is restarted.

    """

    def __init__(
        self, dim: int, tabu_tenure: int, dtype: torch.dtype, device: torch.device
    ) -> None:
        """Initialize.

        Args:
            dim: dimension of design space
            tabu_tenure: number of BO iterations for which a point should be considered
                "tabu" after bein added to the set.
            dtype: dtype
            device: device
        """

        super().__init__()
        self.tkwargs = {"dtype": dtype, "device": device}
        self.register_buffer("_tabu_points", torch.empty((0, dim), **self.tkwargs))
        self.register_buffer("_tabu_tenures", torch.empty(0, **self.tkwargs))
        self.register_buffer(
            "_tabu_tenure", torch.tensor(tabu_tenure, device=device, dtype=torch.int64)
        )

    def add_tabu_point(self, x: Tensor) -> None:
        """Add new point to tabu set.

        Args:
            x: a `d` or `1 x d`-dim tensor with the new point
        """
        self._tabu_points = torch.cat([self._tabu_points, x.view(1, -1)], dim=0)
        self._tabu_tenures = torch.cat(
            [self._tabu_tenures, torch.tensor([self._tabu_tenure], **self.tkwargs)],
            dim=0,
        )

    def get_tabu_points(self) -> Tensor:
        """Retrieve current tabu points"""
        return self._tabu_points

    def log_iteration(self) -> None:
        """Log a BO iteration, decrement tenures, and prune set."""
        self._tabu_tenures -= 1
        still_tabu = self._tabu_tenures > 0
        self._tabu_tenures = self._tabu_tenures[still_tabu]
        self._tabu_points = self._tabu_points[still_tabu]

    def filter_pareto(self, pareto_X: Tensor) -> None:
        """Remove points for tabu set that are no longer pareto efficient.

        Using a hypervolume contributions for selecting TR centers, we will
        only select centers that are pareto efficient, so we can prune the
        tabu set to only include pareto efficient points.

        Args:
            pareto_X: a `n_pareto x d`-dim tensor of pareto points
        """
        # This could be quite memory intensive if pareto_X or the tabu set is
        # large. If we tracked both tabu points and pareto_X in hash maps
        # this would be much more efficient (linear scan of one), but constructing
        # those hashmaps by respresentin each point a tuple (key0 could be expensive.
        # TODO: re-evaluate/improve this.
        tabu_pareto_mask = (
            (self._tabu_points.unsqueeze(1) == pareto_X.unsqueeze(0))
            .all(dim=-1)
            .any(dim=-1)
        )
        self._tabu_points = self._tabu_points[tabu_pareto_mask]
        self._tabu_tenures = self._tabu_tenures[tabu_pareto_mask]


class TRBOState(Module):
    def __init__(
        self,
        dim: int,
        num_outputs: Union[int, List[int]],
        num_objectives: Union[int, List[int]],
        bounds: Tensor,
        max_evals: int,
        tr_hparams: TurboHParams,
        objective: Optional[MCAcquisitionObjective] = None,
        constraints: Optional[Tuple[Tensor, Tensor]] = None,
        num_metrics: Optional[int] = None,
        tr_gen_statuses: Optional[List[TRGenStatus]] = None,
        problem_topology: Optional[Dict[str, Any]] = None, # <-- ADDED PARAMETER
    ) -> None:
        super().__init__()
        tkwargs = {"dtype": bounds.dtype, "device": bounds.device}
        tkwargs_int = {"dtype": torch.long, "device": bounds.device}

        # These are things we keep track of in buffers
        self.register_buffer("dim", torch.tensor(dim, **tkwargs_int))
        if type(num_outputs) is int:
            num_outputs = [num_outputs]
        self.register_buffer("num_outputs", torch.tensor(num_outputs, **tkwargs_int))
        self.register_buffer(
            "num_objectives", torch.tensor(num_objectives, **tkwargs_int)
        )
        self.register_buffer("bounds", bounds)
        self.register_buffer("max_evals", torch.tensor(max_evals, **tkwargs_int))
        self.trust_regions = ModuleList([None] * tr_hparams.n_trust_regions)
        self.register_buffer("total_refill_points", torch.tensor(0, **tkwargs_int))
        self.register_buffer("_tr_gen_statuses", None)
        self.set_tr_gen_statuses(tr_gen_statuses=tr_gen_statuses)
        n_trs = tr_hparams.n_trust_regions
        self.register_buffer(
            "tr_centers", torch.full((n_trs, self.dim), float("nan"), **tkwargs)
        )
        self.register_buffer(
            "tr_center_is_active", torch.zeros(n_trs, dtype=bool, device=bounds.device)
        )
        self.tabu_set = TabuSet(dim=dim, tabu_tenure=tr_hparams.tabu_tenure, **tkwargs)

        # Objective, constraints, and things we don't need to put in buffers
        self.register_buffer("n_evals", torch.tensor(0, **tkwargs_int))
        self.tr_hparams = tr_hparams

        # --- DYNAMIC TOPOLOGY INJECTION ---
        if problem_topology is not None:
            self._inject_dynamic_topology(self.tr_hparams, problem_topology)

            
        if tr_hparams.n_initial_points < tr_hparams.min_tr_size:
            raise ValueError("`n_initial_points` must be greater than `min_tr_size`!")
        self.constraints_spec = constraints
        if constraints is not None:
            self.constraints = get_outcome_constraint_transforms(constraints)
        else:
            self.constraints = None
        if objective is None:
            if num_objectives == 1:
                if num_outputs == 1:
                    objective = IdentityMCObjective()
                else:
                    raise ValueError(
                        "Objective must be provided if num_objectives=1 and "
                        "num_outcomes > 1."
                    )
            else:
                objective = IdentityMCMultiOutputObjective()
        self.objective = objective
        if num_objectives > 1:
            if (
                tr_hparams.hypervolume
                and not tr_hparams.infer_reference_point
                and tr_hparams.max_reference_point is None
            ):
                raise BotorchError(
                    "`infer_reference_point` is `False` and no `max_reference_point` "
                    "was provided. You need to either set `infer_reference_point` to "
                    "`True` or specify `max_reference_point` when using hypervolume."
                )
        elif tr_hparams.hypervolume:
            raise BotorchError(
                "hypervolume can only be used when optimizing multiple objectives."
            )
        self._objective_kwargs = {}
        self.switch_strategy_freq = tr_hparams.switch_strategy_freq or math.inf
        self.switch_strategy = False
        self.register_buffer("X_history", torch.empty((0, self.dim), **tkwargs))
        self.register_buffer(
            "Y_history", torch.empty((0, *self.num_outputs), **tkwargs)
        )
        self.register_buffer("_restart_X", torch.empty((0, self.dim), **tkwargs))
        self.register_buffer(
            "_restart_Y", torch.empty((0, *self.num_outputs), **tkwargs)
        )
        self.TR_index_history = torch.empty(
            (0,), dtype=torch.long, device=bounds.device
        )

        # Things related to hypervolume
        if num_objectives > 1:
            self.pareto_X = torch.empty((0, self.dim), **tkwargs)
            self.pareto_Y = torch.empty((0, *self.num_outputs), **tkwargs)
            self.true_ref_point = (
                torch.tensor(self.tr_hparams.max_reference_point, **tkwargs)
                if self.tr_hparams.max_reference_point is not None
                else None
            )
        self.hv_contributions = None
        self.pareto_Y_better_than_ref = None
        self.pareto_X_better_than_ref = None
        self.hv = None
        self.ref_point = None

    def state_dict(self, destination=None, prefix="", keep_vars=False) -> OrderedDict:
        r"""Returns the state dict with the models removed."""
        state_dict = super().state_dict(destination, prefix, keep_vars)
        return OrderedDict([(k, v) for k, v in state_dict.items() if "model" not in k])

    @property
    def models(self) -> List[Model]:
        r"""Returns a list of models of the trust regions."""
        return [tr.model for tr in self.trust_regions]

    def _set_tabu_buffer_sizes(self, state_dict):
        # TODO(deriksson): _tabu_points and _tabu_tenures may have changed shapes, so we need to
        # reshape them before loading the state dict. The long-term fix is to store
        # the tensor shapes as part of the encoder decoder.
        if "tabu_set._tabu_points" in state_dict.keys():
            tkwargs = {"device": self.bounds.device, "dtype": self.bounds.dtype}
            if state_dict["tabu_set._tabu_points"].numel() == 0:
                state_dict["tabu_set._tabu_points"] = state_dict[
                    "tabu_set._tabu_points"
                ].reshape(0, self.dim)
            else:
                self.tabu_set._tabu_points = torch.zeros(
                    state_dict["tabu_set._tabu_points"].shape, **tkwargs
                )
                self.tabu_set._tabu_tenures = torch.zeros(
                    state_dict["tabu_set._tabu_tenures"].shape, **tkwargs
                )
        return state_dict

    def reload(
        self,
        state_dict: OrderedDict[str, Tensor],
        X_history: Tensor,
        Y_history: Tensor,
        TR_index_history: Tensor,
    ) -> None:
        r"""Reload the state from a state dict + historical data.

        This function assumes that `X_history`, `Y_history`, `TR_index_history`
        are exactly the same as when the state_dict was created.

        Args:
           - state_dict: State dict that we are loading
           - X_history: Historical inputs
           - Y_history: Historical outputs
           - TR_index_history: Trust region indices for the historical data
        """
        state_dict = self._set_tabu_buffer_sizes(state_dict)

        # Now we can load the state dict since all keys will have the correct shape
        self.load_state_dict(state_dict, strict=False)
        self.update(X=X_history, Y=Y_history, new_ind=TR_index_history)
        if "trust_regions.0.X_center" in state_dict:  # We have TR data to load
            # Make sure to use the correct hypervolume setting
            self.tr_hparams = TurboHParams(
                **{
                    **dataclasses.asdict(self.tr_hparams),
                    "hypervolume": "trust_regions.0.ref_point" in state_dict,
                }
            )
            for i in range(self.tr_hparams.n_trust_regions):
                self.initialize_standard(
                    tr_idx=i,
                    X_init=X_history,
                    Y_init=Y_history,
                    restart=False,
                    switch_strategy=False,
                )
            # Load again to load the TR buffers
            state_dict = self._set_tabu_buffer_sizes(state_dict)
            self.load_state_dict(state_dict, strict=False)

    @property
    def tr_gen_statuses(self) -> List[TRGenStatus]:
        return [TRGenStatus(a.cpu().item()) for a in self._tr_gen_statuses]

    def set_tr_gen_statuses(self, tr_gen_statuses) -> None:
        if tr_gen_statuses is not None:
            self._tr_gen_statuses = torch.tensor(
                [a.value for a in tr_gen_statuses],
                dtype=torch.long,
                device=self.bounds.device,
            )

    def update(
        self,
        X: Tensor,
        Y: Tensor,
        new_ind: Tensor,
    ) -> None:
        r"""Update the TRBOState.

        Args:
            X: A `q x d`-dim tensor of new designs.
            Y: A `q x m`-dim tensor of corresponding observations.
            new_ind: A `q`-dim tensor denoting the indices of trust regions
                these observations belong to.
        """
        X = X.to(self.X_history)
        Y = Y.to(self.Y_history)

        self.n_evals.add_(X.shape[0])
        self.X_history = torch.cat((self.X_history, X), dim=0)
        self.Y_history = torch.cat((self.Y_history, Y), dim=0)
        self.TR_index_history = torch.cat([self.TR_index_history, new_ind], dim=0)
        if self.tr_hparams.hypervolume and Y.shape[-1] > 1:
            # update pareto_X, pareto_Y and hypervolume
            X_all = torch.cat([self.pareto_X, X], dim=0)
            Y_all = torch.cat([self.pareto_Y, Y], dim=0)
            if self.constraints is not None:
                is_feas = torch.stack(
                    [c(Y_all) <= 0 for c in self.constraints], dim=-1
                ).all(dim=-1)
                # TODO: evaluate more principled strategies for HV TR center
                # selection if there are no feasible points. Currently, we randomly
                # select a TR center. With a scalarized (or single) objective,
                # we select the point with minimum total violation.
            else:
                is_feas = torch.ones(Y_all.shape[0], dtype=bool, device=Y.device)
            pareto_mask = is_non_dominated(self.objective(Y_all)[is_feas])
            self.pareto_X = X_all[is_feas][pareto_mask]
            self.pareto_Y = Y_all[is_feas][pareto_mask]
            # remove tabu points that are no longer pareto
            self.tabu_set.filter_pareto(pareto_X=self.pareto_X)
            pareto_obj = self.objective(self.pareto_Y)
            if pareto_obj.shape[0] > 0 and self.tr_hparams.infer_reference_point:
                ref_point = None
            elif self.tr_hparams.max_reference_point is not None:
                ref_point = torch.tensor(
                    self.tr_hparams.max_reference_point, dtype=X.dtype, device=X.device
                )
            else:  # Unlikely corner case with no PF and no reference point
                ref_point = self.objective(Y_all).min(dim=0).values
            self.ref_point = infer_reference_point(
                pareto_Y=pareto_obj,
                max_ref_point=ref_point,
            )
            if self.tr_hparams.infer_reference_point and self.tr_hparams.verbose:
                print(f"Inferring the reference point to: {self.ref_point.tolist()}")
            better_than_ref = (pareto_obj > self.ref_point).all(dim=-1)
            self.pareto_Y_better_than_ref = self.pareto_Y[better_than_ref]
            self.pareto_X_better_than_ref = self.pareto_X[better_than_ref]
            if better_than_ref.any():
                pareto_obj_better_than_ref = pareto_obj[better_than_ref]
                partitioning = DominatedPartitioning(
                    ref_point=self.ref_point, Y=pareto_obj_better_than_ref
                )
                self.hv = partitioning.compute_hypervolume().item()
                # get batch pareto frontiers `batch_paretos` where
                # batch_paretos[i] is the pareto frontier without
                # pareto_obj_better_than_ref[i]
                masks = torch.eye(
                    pareto_obj_better_than_ref.shape[0],
                    dtype=torch.bool,
                    device=self.ref_point.device,
                )
                batch_paretos = torch.cat(
                    [
                        pareto_obj_better_than_ref[~m, :].unsqueeze(dim=-3)
                        for m in masks
                    ],
                    dim=-3,
                )
                if batch_paretos.shape[-1] == 2:
                    # use batched box decomposition
                    partitioning = DominatedPartitioning(
                        Y=batch_paretos, ref_point=self.ref_point
                    )
                else:
                    # use box decomposition list
                    partitionings = [
                        DominatedPartitioning(
                            ref_point=self.ref_point, Y=batch_paretos[i]
                        )
                        for i in range(batch_paretos.shape[0])
                    ]
                    partitioning = BoxDecompositionList(*partitionings)
                self.hv_contributions = self.hv - partitioning.compute_hypervolume()
            else:
                self.hv = torch.tensor(0.0, dtype=X.dtype, device=X.device)
                self.hv_contributions = torch.empty(
                    (0,), dtype=X.dtype, device=X.device
                )
            self._filter_pareto_tr_centers()

    def _filter_pareto_tr_centers(self) -> None:
        for i, tr in enumerate(self.trust_regions):
            if tr is not None:
                self.tr_centers[i] = tr.X_center
                self.tr_center_is_active[i] = (
                    (tr.X_center == self.pareto_X).all(dim=-1).any()
                )

    def _log_pareto_tr_center(self, tr_idx: int, active: bool = True) -> None:
        self.tr_centers[tr_idx] = self.trust_regions[tr_idx].X_center
        self.tr_center_is_active[tr_idx] = active

    def update_data_across_trs(
        self, X_all: Optional[Tensor] = None, Y_all: Optional[Tensor] = None
    ) -> None:
        X_all = coalesce(X_all, self.X_history)
        Y_all = coalesce(Y_all, self.Y_history)
        if X_all is None or Y_all is None:
            raise ValueError("No data was provided and no history is stored.")

        for i, tr in enumerate(self.trust_regions):
            # NOTE: this currently shares data across trust regions and restarts.
            prev_center = tr.X_center
            use_global_model = (
                not self.tr_hparams.trim_trace and self.tr_hparams.track_history
            )
            global_model = (
                self.trust_regions[0].model if i > 0 and use_global_model else None
            )
            tr.update(
                X_all=X_all,
                Y_all=Y_all,
                update_streaks=False,
                global_model=global_model,
                **self._get_update_kwargs(tr_idx=i),
            )
            if not torch.equal(prev_center, tr.X_center):
                # update logged tr center
                self._log_pareto_tr_center(tr_idx=i)

    def check_min_points(self) -> List[bool]:
        r"""Check if each TR has enough points.

        We always return true if we aren't using Sobol as the fill_strategy since we
        will just include the closest points in that case.

        Returns:
            A list of booleans indicating whether the trust region has enough points
        """
        if self.tr_hparams.fill_strategy == "sobol":
            min_tr_size = self.tr_hparams.min_tr_size
            return [tr.Y.shape[0] >= min_tr_size for tr in self.trust_regions]
        return [True] * len(self.trust_regions)

    def update_trust_regions_and_log(
        self,
        X_cand: Tensor,
        Y_cand: Tensor,
        tr_indices: Tensor,
        batch_size: int,
        verbose: bool,
        update_streaks: bool = True,
    ) -> List[bool]:
        r"""Update the trust regions and the logs.

        Args:
            X_cand: A `q x d`-dim tensor of new candidates.
            Y_cant: A `q x m`-dim tensor of corresponding observations.
            tr_indices: A `q`-dim tensor denoting the indices of trust regions
                the candidates belong to.
            batch_size: The size of q-batch.
            verbose: A boolean denoting whether to log new best observations.
            update_streaks: A boolean denoting whether to update the success and
                failure counters.

        Returns:
            A list of booleans indicating whether each TR should be restarted.
        """
        if self.X_history is None or self.Y_history is None:
            raise ValueError(
                "`X_history` and `Y_history` are required to update the TRs but"
                " no history is stored!"
            )
        
        # ---> ADDITION 1: Initialize a tracking list on the state object
        if not hasattr(self, 'tr_lifecycle_history'):
            self.tr_lifecycle_history = []


        should_restart_tr = []
        start = time.time()
        for i, tr in enumerate(self.trust_regions):
            tr_mask = tr_indices == i
            # NOTE: This currently shares data across trust regions and restarts.
            if tr_mask.any():
                X_new = X_cand[tr_mask]
                Y_new = Y_cand[tr_mask]
                if verbose and not self.tr_hparams.hypervolume:
                    objective = self.trust_regions[i].objective
                    new_obj = objective(Y_new)
                    old_obj = objective(self.Y_history[:-batch_size])
                    if new_obj.max() > old_obj.max() and self.tr_hparams.verbose:
                        print(
                            f"{self.n_evals}) New best for TR_{i}:"
                            + f"{Y_new[new_obj.argmax()]}"
                        )
            else:
                X_new = None
                Y_new = None
            prev_center = tr.X_center
            use_global_model = (
                not self.tr_hparams.trim_trace and self.tr_hparams.track_history
            )
            global_model = (
                self.trust_regions[0].model if i > 0 and use_global_model else None
            )
            # should_restart_tr.append(
            #     tr.update(
            #         X_all=self.X_history,
            #         Y_all=self.Y_history,
            #         X_new=X_new,
            #         Y_new=Y_new,
            #         update_streaks=update_streaks,
            #         global_model=global_model,
            #         **self._get_update_kwargs(tr_idx=i),
            #     )
            # )

            # 1. Update the Trust Region
            is_restarting = tr.update(
                X_all=self.X_history,
                Y_all=self.Y_history,
                X_new=X_new,
                Y_new=Y_new,
                update_streaks=update_streaks,
                global_model=global_model,
                **self._get_update_kwargs(tr_idx=i),
            )
            should_restart_tr.append(is_restarting)

            # ---> ADDITION 2: Record the exact internal state of the TR post-update
            if tr is not None:
                self.tr_lifecycle_history.append({
                    'eval_count': int(self.n_evals.item()), 
                    'tr_idx': i,
                    'cont_length': float(tr.length.item()),
                    'disc_length': int(tr.length_discrete.item()) if hasattr(tr, 'length_discrete') else None,
                    'num_points': int(tr.X.shape[0]),
                    'n_successes': int(tr.n_successes.item()),
                    'n_failures': int(tr.n_failures.item()),
                    'restarted_this_step': bool(is_restarting)
                })


            if should_restart_tr[-1]:
                # mark center as tabu
                self.tabu_set.add_tabu_point(tr.X_center)
                # mark center as not active
                self._log_pareto_tr_center(tr_idx=i, active=False)
            elif not torch.equal(prev_center, tr.X_center):
                # update logged tr center
                self._log_pareto_tr_center(tr_idx=i)
        end = time.time()
        if self.tr_hparams.verbose:
            print(f"Time spent on model fitting: {end - start:.1f} seconds")

        # ---> ADDITION 3: Dump the history to a CSV at the end of the step
        import pandas as pd
        pd.DataFrame(self.tr_lifecycle_history).to_csv("tr_lifecycle_debug.csv", index=False)


        return should_restart_tr

    def _get_update_kwargs(self, tr_idx: int) -> Dict[str, Union[Tensor, float]]:
        """Get kwargs to TrustRegion.update.

        This method pulls the relevant data from TRBOState and creates a set
        of "invalid centers"--points that are either the center of another trust region
        or are in the tabu set.

        Args:
            tr_idx: the index of the trust region.
        """
        kwargs = {}
        if self.tr_hparams.hypervolume:
            idxr = torch.ones(
                self.tr_centers.shape[0], dtype=bool, device=self.pareto_X.device
            )
            idxr[tr_idx] = False
            is_active_mask = self.tr_center_is_active[idxr]
            other_active_centers = self.tr_centers[idxr][is_active_mask]
            invalid_centers = torch.cat(
                [other_active_centers, self.tabu_set.get_tabu_points()], dim=0
            )
            if self.tr_hparams.verbose:
                print(f"# of tabu points: {self.tabu_set.get_tabu_points().shape[0]}")
                print(f"# of invalid centers: {invalid_centers.shape[0]}")
            if invalid_centers.shape[0] == 0:
                invalid_centers = None
            kwargs.update(
                {
                    "pareto_X_better_than_ref": self.pareto_X_better_than_ref,
                    "pareto_Y_better_than_ref": self.pareto_Y_better_than_ref,
                    "ref_point": self.ref_point,
                    "current_hypervolume": self.hv,
                    "hv_contributions": self.hv_contributions,
                    "invalid_centers": invalid_centers,
                }
            )
        return kwargs

    def gen_scalarization_weights(self) -> Optional[Tensor]:
        """Generate scalarization weights."""
        scalarization_weights = None
        if self.num_objectives > 1 and not self.tr_hparams.hypervolume:
            tkwargs = {
                "dtype": self.bounds.dtype,
                "device": self.bounds.device,
            }
            if self.tr_hparams.fixed_scalarization:
                scalarization_weights = torch.full(
                    (self.num_objectives,), 1.0 / self.num_objectives, **tkwargs
                )

            else:
                scalarization_weights = sample_simplex(
                    d=self.num_objectives, n=1, **tkwargs
                ).squeeze(0)
        return scalarization_weights

    def store_new_trust_region(self, tr_idx: int, tr: TrustRegion) -> None:
        self.trust_regions[tr_idx] = tr
        self._log_pareto_tr_center(tr_idx=tr_idx)

    def check_switch_strategy(self) -> bool:
        """Check if we should switch candidate generation strategy.

        This will switch the candidate generation strategy between hypervolume and
        random scalarizations.
        """
        # Converting n_evals to a float is necessary here since otherwise
        # switch_strategy_freq will be converted to an int, which may overflow
        # if it is set to inf.
        if float(self.n_evals) >= self.switch_strategy_freq:
            use_hv_strategy = (self.n_evals // self.switch_strategy_freq) % 2 == 0
            return use_hv_strategy ^ self.tr_hparams.hypervolume
        return False

    def initialize_standard(
        self,
        tr_idx: int,
        X_init: Optional[Tensor] = None,
        Y_init: Optional[Tensor] = None,
        restart: bool = False,
        switch_strategy: bool = False,
        **init_kwargs,
    ) -> None:
        """This initializes a trust region using new data points if provided.

        If new data is not provided, all historical data is used.
        """
        if X_init is None:
            X_init = self.X_history
            Y_init = self.Y_history
        # start with hv
        tr = self.trust_regions[tr_idx]
        if restart and switch_strategy:
            # switch strategy and save updated strategy in self.tr_hparams
            # this is a no-op after the first tr, since we set `hypervolume`
            # based on `tr.tr_hparams.hypervolume`
            self.tr_hparams = TurboHParams(
                # reverse the trust_region's hypervolume setting
                **{
                    **dataclasses.asdict(self.tr_hparams),
                    "hypervolume": not tr.tr_hparams.hypervolume,
                }
            )
        tr_hparams = self.tr_hparams
        # NOTE: Always apply the decay function even if we aren't restarting. This may happen
        # if we can't restore the state and end up initializing new trust regions.
        decay = decay_function(
            n=max(tr_hparams.n_initial_points, self.n_evals),
            n0=self.tr_hparams.n_initial_points,
            n_max=max(self.max_evals, self.n_evals),
            alpha=self.tr_hparams.decay_restart_length_alpha,
        )
        # Make sure the initial length is never below the minimum length
        length_init = (
            self.tr_hparams.length_min
            + (self.tr_hparams.length_init - self.tr_hparams.length_min) * decay
        )
        # --- ADD DECAY FOR DISCRETE LENGTH ---
        length_init_discrete = int(
            self.tr_hparams.length_min_discrete
            + (self.tr_hparams.length_init_discrete - self.tr_hparams.length_min_discrete) * decay
        )
        # -------------------------------------
        tr_hparams = TurboHParams(
            # **{**dataclasses.asdict(self.tr_hparams), "length_init": length_init}
            **{
                **dataclasses.asdict(self.tr_hparams), 
                "length_init": length_init,
                "length_init_discrete": length_init_discrete # Pass the new initial discrete length
            }
        )

        # Initialize TR
        kwargs = self._get_update_kwargs(tr_idx=tr_idx)
        if tr_hparams.hypervolume:
            # NOTE: `ref_point` is passed in as part of the kwargs
            tr = HypervolumeTrustRegion(
                X_init=X_init,
                Y_init=Y_init,
                bounds=self.bounds,
                tr_hparams=tr_hparams,
                objective=self.objective,
                constraints=self.constraints,
                **kwargs,
                **init_kwargs,
            )
        else:
            scalarization_weights = self.gen_scalarization_weights()
            tr = ScalarizedTrustRegion(
                X_init=X_init,
                Y_init=Y_init,
                bounds=self.bounds,
                tr_hparams=tr_hparams,
                objective=self.objective,
                constraints=self.constraints,
                weights=scalarization_weights,
                **kwargs,
                **init_kwargs,
            )
        self.store_new_trust_region(tr_idx=tr_idx, tr=tr)

    def log_restart_points(self, X: Tensor, Y: Tensor) -> None:
        """Log restart points"""
        self.register_buffer("_restart_X", torch.cat([self._restart_X, X], dim=0))
        self.register_buffer("_restart_Y", torch.cat([self._restart_Y, Y], dim=0))

    def gen_new_restart_design(self):
        # fit model to restart data
        # Scale X from problem space bounds to [0, 1]
        # intf = Normalize(d=self.dim, bounds=self.bounds)
        # --- MODIFIED: ONLY NORMALIZE CONTINUOUS DIMENSIONS ---
        cont_indices = getattr(self.tr_hparams, "cont_dims", None)
        cat_dims = getattr(self.tr_hparams, "cat_dims", None)
        cont_dims = getattr(self.tr_hparams, "cont_dims", None)

        # FIX BUG B: Fetch the new fields into local scope
        binary_dims = getattr(self.tr_hparams, "binary_dims", None)
        ordinal_dims = getattr(self.tr_hparams, "ordinal_dims", None)
        ordinal_config = getattr(self.tr_hparams, "ordinal_config", None)
        binary_dims = getattr(self.tr_hparams, "binary_dims", None)
        ordinal_dims = getattr(self.tr_hparams, "ordinal_dims", None)
        ordinal_config = getattr(self.tr_hparams, "ordinal_config", None)
        use_log_warp = getattr(self.tr_hparams, "use_log_warp", False)

        if cont_indices is not None and len(cont_indices) == 0:
            intf = None
        else:
            # Safely normalize only the continuous dimensions for the global restart model
            intf = Normalize(d=self.dim, bounds=self.bounds, indices=cont_indices)
        
       
        # Standardize Y
        winsorized_Y = torch.from_numpy(
            winsorize(
                self._restart_Y.cpu().numpy(),
                limits=(self.tr_hparams.winsor_pct / 100.0, None),
                axis=0,
            )
        ).to(self._restart_Y)
        octf = Standardize(m=self._restart_Y.shape[-1])

        restart_model = get_fitted_model(
            X=self._restart_X,
            Y=winsorized_Y,
            use_ard=self.tr_hparams.use_ard,
            max_cholesky_size=self.tr_hparams.max_cholesky_size,
            input_transform=intf,
            outcome_transform=octf,
            cat_dims=cat_dims,    # <-- ADDED
            cont_dims=cont_dims,  # <-- ADDED
            binary_dims=self.tr_hparams.binary_dims,
            ordinal_dims=self.tr_hparams.ordinal_dims,
            ordinal_config=self.tr_hparams.ordinal_config,
            use_log_warp=self.tr_hparams.use_log_warp,
            use_unified_kernel=self.tr_hparams.use_unified_kernel,
            use_mixture_kernel=self.tr_hparams.use_mixture_kernel,

        )

        # optimize a HV scalarization
        hv_weights = (
            sample_hypersphere(
                d=self.ref_point.shape[-1],
                n=1,
                qmc=True,
                dtype=self.ref_point.dtype,
                device=self.ref_point.device,
            )
            .abs()
            .unsqueeze(1)
        )
        
        obj_baseline = self.objective(self._restart_Y)
        if self.constraints is not None:
            Y_baseline_constraint_value = torch.stack(
                [c(self._restart_Y) for c in self.constraints], dim=-1
            )
            Y_baseline_feas = (Y_baseline_constraint_value <= 0.0).all(dim=-1)
            obj_baseline = obj_baseline[Y_baseline_feas]

        cat_dims = getattr(self.tr_hparams, "cat_dims", None)
        cont_dims = getattr(self.tr_hparams, "cont_dims", None)
        config = getattr(self.tr_hparams, "config", None)

        # FIX BUG B: Fetch the new fields into local scope
        binary_dims = getattr(self.tr_hparams, "binary_dims", None)
        ordinal_dims = getattr(self.tr_hparams, "ordinal_dims", None)
        ordinal_config = getattr(self.tr_hparams, "ordinal_config", None)

        # ==============================================================
        # MIXED SPACE GLOBAL OPTIMIZATION WITH CASMOPOLITAN
        # ==============================================================
        # if cat_dims is not None and cont_dims is not None:
        if cat_dims and cont_dims:
            # Define the scalarized objective function for interleaved search
            def f_acq(x):
                # Handle both NumPy (from categorical search) and Tensor (from continuous Autograd)
                if isinstance(x, np.ndarray):
                    x_tensor = torch.tensor(x, dtype=self.ref_point.dtype, device=self.ref_point.device)
                else:
                    x_tensor = x.to(self.ref_point.dtype).to(self.ref_point.device)
                
                x_tensor = x_tensor.unsqueeze(0)
                
                # FIX: Unnormalize the continuous dimensions before passing to the model!
                x_unnorm = safe_unnormalize(x_tensor, bounds=self.bounds, indices=cont_dims)
                
                # Use the unnormalized tensor
                # Use posterior mean for stable gradient-based interleaved search
                mean = restart_model.posterior(x_unnorm).mean.squeeze(0)
                obj = self.objective(mean)
                
                if self.constraints is not None:
                    constraint_value = torch.stack([c(mean) for c in self.constraints], dim=-1)
                    feas = (constraint_value <= 0.0).all(dim=-1)
                    violation = torch.clamp(constraint_value, 0.0).sum(dim=-1)
                else:
                    feas = torch.ones(len(obj), device=self.ref_point.device, dtype=torch.bool)
                    violation = torch.zeros(len(obj), dtype=self.ref_point.dtype, device=self.ref_point.device)

                if not any(feas):
                    val = -1 * violation
                else:
                    Y_cat = torch.cat(
                        [
                            obj.unsqueeze(-2),
                            obj_baseline.unsqueeze(0).expand(obj.shape[0], *obj_baseline.shape),
                        ],
                        dim=-2,
                    )
                    val = self._compute_hv_scalarizations(Y=Y_cat, hv_weights=hv_weights).squeeze(-1)
                    # val[~feas] = -1 * violation[~feas] - 1e6 # Heavily penalize constraint violations
                    # --- AUTOGRAD FIX ---
                    # Use torch.where to avoid in-place modification of the computational graph
                    val = torch.where(feas, val, -1 * violation - 1e6)
                
                return val.squeeze()

            # Identify a good starting point for the multi-start search
            # x_center = self._restart_X[-1].cpu().numpy() # Start from the most recently evaluated point
            # Identify a good starting point for the multi-start search
            # Safely normalize the latest restart point before passing to interleaved search
            x_center_tensor = safe_normalize(
                self._restart_X[-1:], 
                bounds=self.bounds, 
                indices=cont_dims
            ).squeeze(0)
            
            x_center = x_center_tensor.cpu().numpy()
            
            # Trust region bounds for the continuous subspace (normalized)
            lb = torch.zeros(len(cont_dims)).numpy()
            ub = torch.ones(len(cont_dims)).numpy()

            # Execute CASMOPOLITAN's Interleaved Search globally
            best_x_np, best_acq = interleaved_search(
                x_center=x_center,
                f=f_acq,
                binary_dims=binary_dims,        # already in local scope above
                ordinal_dims=ordinal_dims,
                ordinal_config=ordinal_config,
                cont_dims=cont_dims,
                config=config,
                ub=ub,
                lb=lb,
                max_hamming_dist=len(cat_dims),
                n_restart=3,
                batch_size=1,
                use_log_warp=use_log_warp,
            )
            best_cand = torch.tensor(best_x_np[0], dtype=self.ref_point.dtype, device=self.ref_point.device).unsqueeze(0)
            
            # Make sure we unnormalize the continuous dimensions back to problem space
            # return unnormalize(best_cand, bounds=self.bounds, indices=cont_indices)
            return safe_unnormalize(best_cand, bounds=self.bounds, indices=cont_dims)
        
        
        elif binary_dims and not cont_dims:
            # FIX BUG 3: Pure discrete restart sampling
            X_discrete = sample_tr_pure_discrete_subset(
                best_X=self._restart_X[-1:],
                n_discrete_points=self.tr_hparams.raw_samples,
                # FIXED POLISH: Global search uses the absolute maximum joint Hamming budget
                length_discrete=len(binary_dims) + len(ordinal_dims),  # Max Hamming distance for global search
                binary_dims=binary_dims,
                ordinal_dims=ordinal_dims,
                ordinal_config=ordinal_config,
                use_log_warp=self.tr_hparams.use_log_warp
            )

            device = self.bounds.device
            dtype = self.bounds.dtype

            with torch.no_grad(), gpytorch.settings.max_eager_kernel_size(float("inf")):
                samples = restart_model.posterior(X_discrete).rsample(sample_shape=torch.Size([1])).squeeze(0)

            # 1. Apply objective
            f_obj = self.objective(samples).clone()

            # 2. Check constraints (if any)
            if self.constraints is not None:
                constraint_value = torch.stack([c(samples) for c in self.constraints], dim=-1)
                feas = (constraint_value <= 0.0).all(dim=-1)
                violation = torch.clamp(constraint_value, 0.0).sum(dim=-1)
            else:
                feas = torch.ones(len(f_obj), dtype=torch.bool, device=device)
                violation = torch.zeros(len(f_obj), dtype=dtype, device=device)

            # 3. Random scalarization to pick the absolute best candidate
            value_score = float("-inf") * torch.ones(len(f_obj), dtype=dtype, device=device)
            if feas.any():
                from botorch.utils.sampling import sample_simplex
                weights = sample_simplex(d=int(self.num_objectives.item()), n=1, device=device, dtype=dtype).squeeze()
                value_score[feas] = (f_obj[feas] @ weights)
            else:
                value_score = -violation

            # ind_best = value_score.argmax()
            # return X_discrete[ind_best, :].unsqueeze(0)

            # STEP 3: Isolate the Top-K Seeds
            K_seeds = 3 # Tunable: Number of local searches to execute
            top_k_indices = value_score.topk(min(K_seeds, len(value_score))).indices
            top_k_seeds = X_discrete[top_k_indices]

            # STEP 4: Interleaved Refinement
            # Define the scalarized objective function for Casmopolitan's local search
            def f_acq(x):
                if isinstance(x, np.ndarray):
                    x_tensor = torch.tensor(x, dtype=dtype, device=device)
                else:
                    x_tensor = x.to(dtype).to(device)
                x_tensor = x_tensor.unsqueeze(0)
                
                # Use posterior mean for stable gradient-free local hill-climbing
                mean = restart_model.posterior(x_tensor).mean.squeeze(0)
                obj = self.objective(mean)
                
                if self.constraints is not None:
                    c_val = torch.stack([c(mean) for c in self.constraints], dim=-1)
                    feas_mask = (c_val <= 0.0).all(dim=-1)
                    viol = torch.clamp(c_val, 0.0).sum(dim=-1)
                else:
                    feas_mask = torch.ones(len(obj), device=device, dtype=torch.bool)
                    viol = torch.zeros(len(obj), dtype=dtype, device=device)

                # Scalarize objectives (e.g., F1 vs Compute Cost) using the fixed weights
                val = torch.where(feas_mask, obj @ weights, -viol - 1e6)
                return val.squeeze()

            best_refined_x = None
            best_refined_score = float("-inf")

            from morbo.localbo_utils import interleaved_search
            lb_dummy = np.array([])
            ub_dummy = np.array([])

            # Execute the search initialized from each of the Top-K seeds
            for seed_idx in range(top_k_seeds.shape[0]):
                seed_np = top_k_seeds[seed_idx].cpu().numpy()
                
                refined_x_np, refined_score = interleaved_search(
                    x_center=seed_np,
                    f=f_acq,
                    binary_dims=binary_dims,
                    ordinal_dims=ordinal_dims,
                    ordinal_config=ordinal_config,
                    cont_dims=[], # Hardcoded empty for pure discrete
                    config=config,
                    ub=ub_dummy,
                    lb=lb_dummy,
                    max_hamming_dist=len(binary_dims) + len(ordinal_dims), # Unrestricted local Hamming space
                    n_restart=1, 
                    batch_size=1,
                    use_log_warp=self.tr_hparams.use_log_warp,
                )
                
                # STEP 5: Center Selection
                if refined_score > best_refined_score:
                    best_refined_score = refined_score
                    best_refined_x = refined_x_np[0]

            return torch.tensor(best_refined_x, dtype=dtype, device=device).unsqueeze(0)
        
        # ==============================================================
        # DEFAULT MORBO PURE CONTINUOUS GLOBAL OPTIMIZATION
        # ==============================================================
        else:
            bounds = torch.zeros(
                2, self.dim, dtype=self.ref_point.dtype, device=self.ref_point.device
            )
            bounds[1] = 1
            X_discrete = draw_sobol_samples(
                bounds=bounds, n=self.tr_hparams.raw_samples, q=1
            ).squeeze(1)
            with torch.no_grad():
                samples = restart_model.posterior(X_discrete).rsample().squeeze(0)
            obj = self.objective(samples)
            
            if self.constraints is not None:
                constraint_value = torch.stack(
                    [c(samples) for c in self.constraints], dim=-1
                )
                feas = (constraint_value <= 0.0).all(dim=-1)
                violation = torch.clamp(constraint_value, 0.0).sum(dim=-1)
            else:
                feas = torch.ones(len(obj), device=self.ref_point.device, dtype=torch.bool)
                violation = torch.zeros(
                    len(obj),
                    dtype=self.ref_point.dtype,
                    device=self.ref_point.device,
                )
                
            if not any(feas):
                value_score = -1 * violation
            else:
                value_score = torch.full(
                    (len(obj),),
                    float("-inf"),
                    dtype=self.ref_point.dtype,
                    device=self.ref_point.device,
                )
                Y = torch.cat(
                    [
                        obj.unsqueeze(-2),
                        obj_baseline.unsqueeze(0).expand(obj.shape[0], *obj_baseline.shape),
                    ],
                    dim=-2,
                )
                hvs = self._compute_hv_scalarizations(Y=Y, hv_weights=hv_weights).squeeze(-1)
                value_score[feas] = hvs[feas]
                
            best_idx = value_score.argmax()
            # return unnormalize(X_discrete[best_idx : best_idx + 1], self.bounds)
            return safe_unnormalize(X_discrete[best_idx : best_idx + 1], bounds=self.bounds, indices=cont_dims)

    def _compute_hv_scalarizations(self, Y: Tensor, hv_weights: Tensor) -> Tensor:
        r"""Compute HV scalarizations.

        Args:
            Y: A `sample_shape x batch_shape x n x m`-dim tensor of outcomes

        Returns:
            A `sample_shape x batch_shape x n_weights`-dim tensor of hv scalarizations.

        """
        return (
            ((Y - self.ref_point).clamp_min(0).unsqueeze(-3) / hv_weights)
            .amin(dim=-1)
            .pow(Y.shape[-1])
            .amax(dim=-1)
        )
    
    def _inject_dynamic_topology(self, tr_hparams: TurboHParams, topology: Dict[str, Any]) -> None:
        """
        Dynamically extracts the topological representation of the objective space.
        Ensures the local search algorithms only interact with a generic vector x ∈ R^d,
        completely decoupled from the physical problem formulation.
        
        Expected topology dictionary format:
        {
            "continuous": [0, 1, 2],
            "binary": [3, 4],
            "ordinal": {5: 4, 6: 3}  # index: number of choices
        }
        """
        # 1. Extract Continuous Dimensions
        cont_dims = topology.get("continuous", [])
        tr_hparams.cont_dims = cont_dims

        # 2. Extract Binary Dimensions
        binary_dims = topology.get("binary", [])
        tr_hparams.binary_dims = binary_dims

        # 3. Extract Ordinal Dimensions and Configurations
        ordinal_dict = topology.get("ordinal", {})
        ordinal_dims = list(ordinal_dict.keys())
        ordinal_config = list(ordinal_dict.values())
        
        tr_hparams.ordinal_dims = ordinal_dims
        tr_hparams.ordinal_config = ordinal_config

        # 4. Construct Legacy Casmopolitan Catch-Alls (if needed by older functions)
        tr_hparams.cat_dims = binary_dims + ordinal_dims
        
        # Build the generic configuration list Casmopolitan expects for max boundary calculations
        # Binary variables have 2 choices, ordinal variables have 'N' choices.
        config = []
        for i in range(self.dim.item()):
            if i in binary_dims:
                config.append(2)
            elif i in ordinal_dims:
                config.append(ordinal_dict[i])
            else:
                config.append(0) # 0 denotes a continuous dimension
                
        tr_hparams.config = config
    
