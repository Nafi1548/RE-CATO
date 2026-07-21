import math
import logging

import gpytorch
import numpy as np
import torch
from gpytorch.constraints.constraints import Interval
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.models import ExactGP
# from collections import Callable
from typing import Callable, List
# (or 'from collections.abc import Callable', both work perfectly!)
import random
from copy import deepcopy
import time
from morbo.kernels import *
# debug
try:
    from test_funcs import *
except ImportError:
    pass

def onehot2ordinal(x, categorical_dims):
    """Convert one-hot representation of strings back to ordinal representation."""
    from itertools import chain
    if x.ndim == 1:
        x = x.reshape(1, -1)
    categorical_dims_flattned = list(chain(*categorical_dims))
    # Select those categorical dimensions only
    x = x[:, categorical_dims_flattned]
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    res = torch.zeros(x.shape[0], len(categorical_dims), dtype=torch.float32)
    for i, var_group in enumerate(categorical_dims):
        res[:, i] = torch.argmax(x[:, var_group], dim=-1).float()
    return res


def ordinal2onehot(x, n_categories):
    """Convert ordinal to one-hot"""
    res = np.zeros(np.sum(n_categories))
    offset = 0
    for i, cat in enumerate(n_categories):
        res[offset + int(x[i])] = 1
        offset += cat
    return torch.tensor(res)


# GP Model
class GP(ExactGP):
    def __init__(self, train_x, train_y, kern, likelihood,
                 outputscale_constraint,
                 ard_dims, cat_dims=None):
        super(GP, self).__init__(train_x, train_y, likelihood)
        self.dim = train_x.shape[1]
        self.ard_dims = ard_dims
        self.cat_dims = cat_dims
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(kern, outputscale_constraint=outputscale_constraint)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)  # , cat_dims, int_dims)
        return MultivariateNormal(mean_x, covar_x)


def train_gp(train_x, train_y, use_ard, num_steps, kern='transformed_overlap', hypers={},
             cat_dims=None, cont_dims=None,
             int_constrained_dims=None,
             noise_variance=None,
             cat_configs=None,
             **params):
    """Fit a GP model where train_x is in [0, 1]^d and train_y is standardized.
    （train_x, train_y）: pairs of x and y (trained)
    noise_variance: if provided, this value will be used as the noise variance for the GP model. Otherwise, the noise
        variance will be inferred from the model.
    int_constrained_dims: **Of the continuous dimensions**, which ones additionally are constrained to have integer
        values only?
    """
    assert train_x.ndim == 2
    assert train_y.ndim == 1
    assert train_x.shape[0] == train_y.shape[0]

    # Create hyper parameter bounds
    if noise_variance is None:
        noise_variance = 0.005
        noise_constraint = Interval(1e-6, 0.1)
    else:
        if np.abs(noise_variance) < 1e-6:
            noise_variance = 0.05
            noise_constraint = Interval(1e-6, 0.1)
        else:
            noise_constraint = Interval(0.99 * noise_variance, 1.01 * noise_variance)
    if use_ard:
        lengthscale_constraint = Interval(0.01, 0.5)
    else:
        lengthscale_constraint = Interval(0.01, 2.5)  # [0.005, sqrt(dim)]
    # outputscale_constraint = Interval(0.05, 20.0)
    outputscale_constraint = Interval(0.5, 5.)

    # Create models
    likelihood = GaussianLikelihood(noise_constraint=noise_constraint).to(device=train_x.device, dtype=train_y.dtype)
    # train_x = onehot2ordinal(train_x, cat_dims)
    ard_dims = train_x.shape[1] if use_ard else None

    if kern == 'overlap':
        kernel = CategoricalOverlap(lengthscale_constraint=lengthscale_constraint, ard_num_dims=ard_dims, )
    elif kern == 'transformed_overlap':
        kernel = TransformedCategorical(lengthscale_constraint=lengthscale_constraint, ard_num_dims=ard_dims, )
    elif kern == 'ordinal':
        kernel = OrdinalKernel(lengthscale_constraint=lengthscale_constraint, ard_num_dims=ard_dims, config=cat_configs)
    elif kern == 'mixed':
        assert cat_dims is not None and cont_dims is not None, 'cat_dims and cont_dims need to be specified if you wish' \
                                                               'to use the mix kernel'
        kernel = MixtureKernel(cat_dims, cont_dims,
                               categorical_ard=use_ard, continuous_ard=use_ard,
                               integer_dims=int_constrained_dims,
                               **params)
    elif kern == 'mixed_overlap':
        kernel = MixtureKernel(cat_dims, cont_dims,
                               categorical_ard=use_ard, continuous_ard=use_ard,
                               categorical_kern_type='overlap',
                               integer_dims=int_constrained_dims,
                               **params)
    elif kern == 'unified_discrete':
        # Ensure your new kernel can be initialized here
        assert cat_dims is not None and int_constrained_dims is not None, "binary and ordinal dims must be provided"
        kernel = UnifiedDiscreteKernel(
            binary_dims=cat_dims, 
            ordinal_dims=int_constrained_dims, 
            ordinal_config=cat_configs,
            lamda=0.5
        )
    else:
        raise ValueError('Unknown kernel choice %s' % kern)

    model = GP(
        train_x=train_x,
        train_y=train_y,
        likelihood=likelihood,
        kern=kernel,
        # lengthscale_constraint=lengthscale_constraint,
        outputscale_constraint=outputscale_constraint,
        ard_dims=ard_dims,
    ).to(device=train_x.device, dtype=train_x.dtype)

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # "Loss" for GPs - the marginal log likelihood
    mll = ExactMarginalLogLikelihood(likelihood, model)

    # Initialize model hypers
    if hypers:
        model.load_state_dict(hypers)
    else:
        hypers = {}
        hypers["covar_module.outputscale"] = 1.0
        hypers["covar_module.base_kernel.lengthscale"] = np.sqrt(0.01 * 0.5)
        hypers["likelihood.noise"] = noise_variance if noise_variance is not None else 0.005
        model.initialize(**hypers)

    # Use the adam optimizer
    # optimizer = torch.optim.Adam([{"params": model.parameters()}], lr=0.03)

    # for _ in range(num_steps):
    #     optimizer.zero_grad()
    #     output = model(train_x, )
    #     loss = -mll(output, train_y).float()
    #     loss.backward()
    #     optimizer.step()

    # # Switch to eval mode
    # model.eval()
    # likelihood.eval()

    # return model
    # Use the adam optimizer
    optimizer = torch.optim.Adam([{"params": model.parameters()}], lr=0.03)

    # --- ADD THIS: Create a safe backup of the model state ---
    safe_state = {k: v.clone() for k, v in model.state_dict().items()}

    for _ in range(num_steps):
        optimizer.zero_grad()
        output = model(*model.train_inputs)
        loss = -mll(output, train_y).float()

        # SAFETY CATCH 1: Did the covariance matrix collapse?
        if torch.isnan(loss) or torch.isinf(loss):
            model.load_state_dict(safe_state)
            break

        loss.backward()

        # SAFETY CATCH 2: Did the gradients explode?
        has_nan_grad = any(torch.isnan(p.grad).any() for p in model.parameters() if p.grad is not None)
        if has_nan_grad:
            model.load_state_dict(safe_state)
            break

        optimizer.step()
        
        # If the step was mathematically healthy, update our safe backup
        safe_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Switch to eval mode
    model.eval()
    likelihood.eval()

    return model


def to_unit_cube(x, lb, ub):
    """Project to [0, 1]^d from hypercube with bounds lb and ub"""
    assert np.all(lb < ub) and lb.ndim == 1 and ub.ndim == 1 and x.ndim == 2
    xx = (x - lb) / (ub - lb)
    return xx


def from_unit_cube(x, lb, ub):
    """Project from [0, 1]^d to hypercube with bounds lb and ub"""
    assert np.all(lb < ub) and lb.ndim == 1 and ub.ndim == 1 and x.ndim == 2
    xx = x * (ub - lb) + lb
    return xx


def latin_hypercube(n_pts, dim):
    import time
    """Basic Latin hypercube implementation with center perturbation."""
    X = np.zeros((n_pts, dim))
    centers = (1.0 + 2.0 * np.arange(0.0, n_pts)) / float(2 * n_pts)
    # random.seed(random.randint(0, 1e6))
    for i in range(dim):  # Shuffle the center locataions for each dimension.
        X[:, i] = centers[np.random.permutation(n_pts)]

    # Add some perturbations within each box
    pert = np.random.uniform(-1.0, 1.0, (n_pts, dim)) / float(2 * n_pts)
    X += pert
    return X


def compute_hamming_dist(x1, x2, categorical_dims, normalize=False):
    """
    Compute the hamming distance of two one-hot encoded strings.
    :param x1:
    :param x2:
    :param categorical_dims: list of lists. e.g.
    [[1, 2], [3, 4, 5, 6]] where idx 1 and 2 correspond to the first variable, and
    3, 4, 5, 6 coresponds to the second variable with 4 possible options
    :return:
    """
    dist = 0.
    for i, var_groups in enumerate(categorical_dims):
        if not np.all(x1[var_groups] == x2[var_groups]):
            dist += 1.
    if normalize:
        dist /= len(categorical_dims)
    return dist


def compute_hamming_dist_ordinal(x1, x2, n_categories=None, normalize=False):
    """Same as above, but on ordinal representations."""
    hamming = (x1 != x2).sum()
    if normalize:
        return hamming / len(x1)
    return hamming


def sample_neighbour(x, categorical_dims):
    """Sample a neighbour (i.e. of unnormalised Hamming distance of 1) from x"""
    x_pert = deepcopy(x)
    # Sample a variable where x_pert will differ from the selected sample
    # random.seed(random.randint(0, 1e6))
    choice = random.randint(0, len(categorical_dims) - 1)
    # Change the value of that variable randomly
    var_group = categorical_dims[choice]
    # Confirm what is value of the selected variable in x (we will not sample this point again)
    for var in var_group:
        if x_pert[var] != 0:
            break
    value_choice = random.choice(var_group)
    while value_choice == var:
        value_choice = random.choice(var_group)
    x_pert[var] = 0
    x_pert[value_choice] = 1
    return x_pert


def sample_neighbour_ordinal(x, n_categories):
    """Same as above, but the variables are represented ordinally."""
    x_pert = deepcopy(x)
    # Chooose a variable to modify
    choice = random.randint(0, len(n_categories) - 1)
    # Obtain the current value.
    curr_val = x[choice]
    options = [i for i in range(n_categories[choice]) if i != curr_val]
    x_pert[choice] = random.choice(options)
    return x_pert


def random_sample_within_discrete_tr(x_center, max_hamming_dist, categorical_dims,
                                     mode='ordinal'):
    """Randomly sample a point within the discrete trust region"""
    if max_hamming_dist < 1:  # Normalised hamming distance is used
        bit_change = int(max_hamming_dist * len(categorical_dims))
    else:  # Hamming distance is not normalized
        max_hamming_dist = min(max_hamming_dist, len(categorical_dims))
        bit_change = int(max_hamming_dist)

    x_pert = deepcopy(x_center)
    # Randomly sample n bits to change.
    modified_bits = random.sample(range(len(categorical_dims)), bit_change)
    for bit in modified_bits:
        n_values = len(categorical_dims[bit])
        # Change this value
        selected_value = random.choice(range(n_values))
        # Change to one-hot encoding
        substitute_values = np.array([1 if i == selected_value else 0 for i in range(n_values)])
        x_pert[categorical_dims[bit]] = substitute_values
    return x_pert


def random_sample_within_discrete_tr_ordinal(x_center, max_hamming_dist, n_categories):
    """Same as above, but here we assume a ordinal representation of the categorical variables."""
    # random.seed(random.randint(0, 1e6))
    if max_hamming_dist < 1:
        bit_change = int(max(max_hamming_dist * len(n_categories), 1))
    else:
        bit_change = int(min(max_hamming_dist, len(n_categories)))
    x_pert = deepcopy(x_center)
    modified_bits = random.sample(range(len(n_categories)), bit_change)
    for bit in modified_bits:
        # options = np.arange(n_categories[bit])
        # x_pert[bit] = int(random.choice(options))
        options = [val for val in range(n_categories[bit]) if val != x_pert[bit]]
        if options:
            x_pert[bit] = int(np.random.choice(options))
    return x_pert


def local_search(x_center, f: Callable,
                 config,
                 max_hamming_dist,
                 n_restart: int = 1,
                 batch_size: int = 1,
                 step: int = 200):
    """
    Local search algorithm
    :param n_restart: number of restarts
    :param config:
    :param x0: the initial point to start the search
    :param x_center: the center of the trust region. In this case, this should be the optimum encountered so far.
    :param f: the function handle to evaluate x on (the acquisition function, in this case)
    :param max_hamming_dist: maximum Hamming distance from x_center
    :param step: number of maximum local search steps the algorithm is allowed to take.
    :return:
    """

    def _ls(hamming):
        """One restart of local search"""
        # x0 = deepcopy(x_center)
        x_center_local = deepcopy(x_center)
        tol = 100
        trajectory = np.array([x_center_local])
        x = x_center_local

        acq_x = f(x).detach().numpy()
        for i in range(step):
            tol_ = tol
            is_valid = False
            while not is_valid:
                neighbour = sample_neighbour_ordinal(x, config)
                if 0 < compute_hamming_dist_ordinal(x_center_local, neighbour, config) <= hamming \
                        and not any(np.equal(trajectory, neighbour).all(1)):
                    is_valid = True
                else:
                    tol_ -= 1
            if tol_ < 0:
                logging.info("Tolerance exhausted on this local search thread.")
                return x, acq_x

            acq_x = f(x).detach().numpy()
            acq_neighbour = f(neighbour).detach().numpy()
            # print(acq_x, acq_neighbour)

            if acq_neighbour > acq_x:
                logging.info(''.join([str(int(i)) for i in neighbour.flatten()]) + ' ' + str(acq_neighbour))
                x = deepcopy(neighbour)
                # trajectory = np.vstack((trajectory, deepcopy(x)))
        logging.info('local search thread ended with highest acquisition %s' % acq_x)
        # print(compute_hamming_dist_ordinal(x_center, x, n_categories), acq_x)
        # print(x_center)
        return x, acq_x

    X = []
    fX = []
    for i in range(n_restart):
        res = _ls(max_hamming_dist)
        X.append(res[0])
        fX.append(res[1])

    top_idices = np.argpartition(np.array(fX).flatten(), -batch_size)[-batch_size:]
    # select the top-k smallest
    # top_idices = np.argpartition(np.array(fX).flatten(), batch_size)[:batch_size]
    # print(np.array(fX).flatten()[top_idices])
    return np.array([x for i, x in enumerate(X) if i in top_idices]), np.array(fX).flatten()[top_idices]


import random
import numpy as np

def sample_neighbour_mixed(x, binary_dims: list[int], ordinal_dims: list[int], ordinal_config: list[int], max_hamming_dist: int, use_log_warp: bool = False):
    """
    Generates a localized neighbor in a true mixed discrete space.
    Flips a single bit for binary variables, or increments/decrements 
    by small step for ordinal variables.
    """
    # --- SAFETY GUARD: Prevent crash on empty topologies ---
    if not binary_dims and not ordinal_dims:
        return x.copy()
    
    x_pert = x.copy()
    
    total_dims = len(binary_dims) + len(ordinal_dims)
    tr_ratio = float(max_hamming_dist) / float(total_dims)

    # Decide whether to step through a binary space or an ordinal space
    perturb_binary = False
    if len(binary_dims) > 0:
        if len(ordinal_dims) == 0 or random.random() < 0.5:
            perturb_binary = True

    if perturb_binary:
        # 1. Strict Binary Topology: Flip a random bit
        choice = random.choice(binary_dims)
        x_pert[choice] = 1 - x_pert[choice]
    # else:
    #     # # 2. Strict Ordinal Topology: Step exactly +/- 1 within bounds
    #     # idx = random.randint(0, len(ordinal_dims) - 1)
    #     # choice = ordinal_dims[idx]
    #     # max_choices = ordinal_config[idx]
        
    #     # # Adaptive Scaling for Local Search
    #     # dynamic_radius = max(1, int(max_choices * tr_ratio * 0.1)) # 10% decay for local exploitation
    #     # # step = random.randint(1, dynamic_radius) * random.choice([-1, 1])
    #     # # REPLACE WITH:
    #     # import math
    #     # log_step = random.uniform(0, math.log(dynamic_radius + 1))
    #     # step_size = max(1, int(math.exp(log_step)))
    #     # step = step_size * random.choice([-1, 1])

    #     # curr_val = int(x_pert[choice])
    #     # # step = random.choice([-1, 1])
        
    #     # # Clamp strictly within the true parameter index bounds
    #     # x_pert[choice] = max(0, min(max_choices - 1, curr_val + step))

    #     # 2. Strict Ordinal Topology: Log-Warped Local Step
    #     idx = random.randint(0, len(ordinal_dims) - 1)
    #     choice = ordinal_dims[idx]
    #     max_choices = ordinal_config[idx]
        
    #     import math
    #     curr_val = int(x_pert[choice])
        
    #     # 1. Map to log-space
    #     z_max = math.log(max_choices)
    #     z_curr = math.log(curr_val + 1)
        
    #     # 2. Adaptive Scaling for Local Search (10% of the TR radius)
    #     r_log = (tr_ratio * 0.1) * z_max
        
    #     # 3. Bounded local draw
    #     z_min_bound = max(0.0, z_curr - r_log)
    #     z_max_bound = min(z_max, z_curr + r_log)
    #     z_new = random.uniform(z_min_bound, z_max_bound)
        
    #     # 4. Map back and clamp
    #     new_val = int(math.exp(z_new)) - 1
    #     x_pert[choice] = max(0, min(max_choices - 1, new_val))
    else:
        # Ordinal Topology Step
        idx = random.randint(0, len(ordinal_dims) - 1)
        choice = ordinal_dims[idx]
        max_choices = ordinal_config[idx]
        curr_val = int(x_pert[choice])
        
        if use_log_warp:
            # Log-Warped Local Step (Exploits percentage changes)
            import math
            z_max = math.log(max_choices)
            z_curr = math.log(curr_val + 1)
            r_log = (tr_ratio * 0.1) * z_max
            z_min_bound = max(0.0, z_curr - r_log)
            z_max_bound = min(z_max, z_curr + r_log)
            z_new = random.uniform(z_min_bound, z_max_bound)
            new_val = int(math.exp(z_new)) - 1
        else:
            # Strict Linear Step (Exploits absolute gaps)
            dynamic_radius = max(1, int(max_choices * tr_ratio * 0.1))
            step = random.randint(1, dynamic_radius) * random.choice([-1, 1])
            new_val = curr_val + step
            
        x_pert[choice] = max(0, min(max_choices - 1, new_val))
        
    return x_pert


def compute_mixed_discrete_distance(x1, x2, binary_dims, ordinal_dims, ordinal_config):
    """
    Computes a mixed distance where:
    - Binary changes cost 1.0 (Hamming)
    - Ordinal changes are Manhattan gaps normalized by their max range (costing up to 1.0)
    """
    bin_dist = 0.0
    if len(binary_dims) > 0:
        bin_dist = (x1[binary_dims] != x2[binary_dims]).sum()
        
    ord_dist = 0.0
    if len(ordinal_dims) > 0:
        for idx, dim in enumerate(ordinal_dims):
            max_range = max(1, ordinal_config[idx] - 1)
            raw_gap = np.abs(x1[dim] - x2[dim])
            ord_dist += (raw_gap / max_range)
            
    return float(bin_dist + ord_dist)



# def interleaved_search(x_center, f: Callable,
#                        cat_dims,
#                        cont_dims,
#                        config,
#                        ub,
#                        lb,
#                        max_hamming_dist,
#                        n_restart: int = 1,
#                        batch_size: int = 1,
#                        interval: int = 1,
#                        step: int = 200):
#     """
#     Interleaved search combining both first-order gradient-based method on the continuous variables and the local search
#     for the categorical variables.
#     Parameters
#     ----------
#     x_center: the starting point of the search
#     cat_dims: the indices of the categorical dimensions
#     cont_dims: the indices of the continuous dimensions
#     f: function handle (normally this should be the acquisition function)
#     config: the config for the categorical variables
#     lb: lower bounds (trust region boundary) for the continuous variables
#     ub: upper bounds (trust region boundary) for the continuous variables
#     max_hamming_dist: maximum hamming distance boundary (for the categorical variables)
#     n_restart: number of restarts of the optimisaiton
#     batch_size:
#     interval: number of steps to switch over (to start with, we optimise with n_interval steps on the continuous
#         variables via a first-order optimiser, then we switch to categorical variables (with the continuous ones fixed)
#         and etc.
#     step: maximum number of search allowed.

#     Returns
#     -------

#     """
#     # todo: the batch setting needs to be changed. For the continuous dimensions, we cannot simply do top-n indices.

#     from torch.quasirandom import SobolEngine
#     from scipy.optimize import minimize, Bounds

#     # select the initialising points for both the continuous and categorical variables and then hstack them together
#     # x0_cat = np.array([deepcopy(sample_neighbour_ordinal(x_center[cat_dims], config)) for _ in range(n_restart)])
#     x0_cat = np.array([deepcopy(random_sample_within_discrete_tr_ordinal(x_center[cat_dims], max_hamming_dist, config))
#                        for _ in range(n_restart)])
#     # x0_cat = np.array([deepcopy(x_center[cat_dims]) for _ in range(n_restart)])
#     seed = np.random.randint(int(1e6))
#     sobol = SobolEngine(len(cont_dims), scramble=True, seed=seed)
#     x0_cont = sobol.draw(n_restart).cpu().detach().numpy()
#     x0_cont = lb + (ub - lb) * x0_cont
#     x0 = np.hstack((x0_cat, x0_cont))
#     tol = 100
#     lb, ub = torch.tensor(lb, dtype=torch.float32), torch.tensor(ub, dtype=torch.float32)
    
#     def _interleaved_search(x0):
#         x = deepcopy(x0)
#         acq_x = f(x).detach().numpy()
#         x_cat, x_cont = x[cat_dims], x[cont_dims]
#         n_step = 0
#         while n_step <= step:
#             # First optimise the continuous part, freezing the categorical part
#             def f_cont(x_cont_):
#                 """The function handle for continuous optimisation"""
#                 x_ = torch.cat((x_cat_torch, x_cont_)).float()
#                 return -f(x_)

#             x_cont_torch = torch.tensor(x_cont, dtype=torch.float32).requires_grad_(True)
#             x_cat_torch = torch.tensor(x_cat, dtype=torch.float32)
#             optimizer = torch.optim.Adam([{"params": x_cont_torch}], lr=0.1)
#             for _ in range(interval):
#                 old_x_cont = x_cont_torch.clone().detach()  # <--- SAVE BACKUP
#                 optimizer.zero_grad()
#                 acq = f_cont(x_cont_torch).float()
#                 try:
#                     acq.backward()
#                     # print(x_cont_torch, acq, x_cont_torch.grad)
#                     optimizer.step()
#                 except RuntimeError:
#                     print('Exception occured during backpropagation. NaN encountered?')
#                     pass
#                 with torch.no_grad():
#                     # <--- SAFETY CATCH: If Adam stepped into a NaN, revert and stop!
#                     if torch.isnan(x_cont_torch).any():
#                         x_cont_torch.data = old_x_cont
#                         break 
                        
#                     # Ugly way to do clipping
#                     x_cont_torch.data = torch.max(torch.min(x_cont_torch, ub), lb)

#             x_cont = x_cont_torch.detach().numpy()
#             del x_cont_torch

#             # Then freeze the continuous part and optimise the categorical part
#             for j in range(interval):
#                 is_valid = False
#                 tol_ = tol
#                 while not is_valid:
#                     neighbour = sample_neighbour_ordinal(x_cat, config)
#                     if 0 <= compute_hamming_dist_ordinal(x_center[cat_dims], neighbour, config) <= max_hamming_dist:
#                         is_valid = True
#                     else:
#                         tol_ -= 1
#                 if tol_ < 0:
#                     logging.info("Tolerance exhausted on this local search thread.")
#                     break
#                 # acq_x = f(np.hstack((x_cat, x_cont))).detach().numpy()
#                 acq_neighbour = f(np.hstack((neighbour, x_cont))).detach().numpy()
#                 if acq_neighbour > acq_x:
#                     x_cat = deepcopy(neighbour)
#                     acq_x = acq_neighbour
#             # print(x_cat, x_cont, acq_x)
#             n_step += interval

#         x = np.hstack((x_cat, x_cont))
#         return x, acq_x

#     X, fX = [], []
#     for i in range(n_restart):
#         res = _interleaved_search(x0[i, :])
#         X.append(res[0])
#         fX.append(res[1])
#     top_idices = np.argpartition(np.array(fX).flatten(), -batch_size)[-batch_size:]
#     return np.array([x for i, x in enumerate(X) if i in top_idices]), np.array(fX).flatten()[top_idices]


def interleaved_search(x_center, f: Callable,
                       binary_dims: List[int],     # ADD THIS
                        ordinal_dims: List[int],    # ADD THIS
                       ordinal_config: List[int],  # ADD THIS
                       cont_dims,
                       config,
                       ub,
                       lb,
                       max_hamming_dist,
                       n_restart: int = 1,
                       batch_size: int = 1,
                       interval: int = 1,
                       step: int = 200,
                       use_log_warp: bool = False):
    """
    Interleaved search combining both first-order gradient-based method on the continuous variables and the local search
    for the categorical variables.
    """
    import logging
    from copy import deepcopy
    import numpy as np
    import torch
    from torch.quasirandom import SobolEngine
    from scipy.optimize import minimize, Bounds

    # # 1. FIXED INITIALIZATION: Construct x0 in GLOBAL dimensional order from the start
    # x0 = np.tile(x_center, (n_restart, 1))
    # cat_dims = binary_dims + ordinal_dims
    # for i in range(n_restart):
    #     x0[i, cat_dims] = random_sample_within_discrete_tr_ordinal(
    #         x_center[cat_dims], max_hamming_dist, config
    #     )
    
    # --- TOPOLOGY FALLBACK ---
    # If the legacy runner bypassed explicit dimensions, reconstruct them from config
    # Binary variables have 2 choices. Ordinal variables have >2 choices.
    if not binary_dims and not ordinal_dims and config is not None:
        binary_dims = [i for i, c in enumerate(config) if c == 2]
        ordinal_dims = [i for i, c in enumerate(config) if c > 2]
        ordinal_config = [c for c in config if c > 2]

    # 1. FIXED INITIALIZATION: Construct x0 in GLOBAL dimensional order from the start
    x0 = np.tile(x_center, (n_restart, 1))
    
    # --- ROBUSTNESS PATCH: Recover topology if legacy Cato scripts bypassed binary_dims ---
    binary_dims = binary_dims or []
    ordinal_dims = ordinal_dims or []
    
    if not binary_dims and not ordinal_dims and config:
        # Fallback: Reconstruct categorical dimensions from the global config
        cat_dims = [i for i, c in enumerate(config) if c > 0]
    else:
        cat_dims = binary_dims + ordinal_dims

    # --- SLICING PATCH: Localize the configuration for the discrete sampler ---
    # The sampler expects n_categories to match the length of the slice, not the global array.
    if config and len(config) >= len(cat_dims):
        cat_config = [config[i] for i in cat_dims]
    else:
        cat_config = config or []

    for i in range(n_restart):
        if len(cat_dims) > 0:  # Safety guard: Prevent empty array slicing
            x0[i, cat_dims] = random_sample_within_discrete_tr_ordinal(
                x_center[cat_dims], max_hamming_dist, cat_config
            )


    seed = np.random.randint(int(1e6))
    # sobol = SobolEngine(len(cont_dims), scramble=True, seed=seed)
    # In localbo_utils_35.py
    if len(cont_dims) > 0:
        sobol = SobolEngine(len(cont_dims), scramble=True, seed=seed)
        # Perform the initialization logic
        x0[:, cont_dims] = lb + (ub - lb) * sobol.draw(n_restart).cpu().detach().numpy()
        x0_cont = lb + (ub - lb) * sobol.draw(n_restart).cpu().detach().numpy()
        
        # Overwrite the continuous dimensions in their correct global positions
        x0[:, cont_dims] = x0_cont
    else:
        # If pure discrete, the logic inside interleaved_search 
        # must not attempt to index or transform cont_dims
        pass
    
    tol = 100
    lb, ub = torch.tensor(lb, dtype=torch.float32), torch.tensor(ub, dtype=torch.float32)
    
    def _interleaved_search(x_start):
        x = deepcopy(x_start)  
        
        if isinstance(x, np.ndarray):
            acq_x = f(torch.tensor(x, dtype=torch.float32)).detach().cpu().numpy()
        else:
            acq_x = f(x).detach().cpu().numpy()

        cat_dims = binary_dims + ordinal_dims
        x_cat = x[cat_dims]
        x_cont = x[cont_dims]
        n_step = 0
        
        if len(cont_dims) > 0:
            # x_cont_torch = torch.tensor(x_cont, dtype=torch.float32).requires_grad_(True)
            x_cont_torch = torch.tensor(x_cont, dtype=lb.dtype, device=lb.device).requires_grad_(True)
            lb_t = torch.tensor(lb, dtype=x_cont_torch.dtype, device=x_cont_torch.device)
            ub_t = torch.tensor(ub, dtype=x_cont_torch.dtype, device=x_cont_torch.device)
            optimizer = torch.optim.Adam([x_cont_torch], lr=0.1)

        while n_step <= step:
            # ---------------------------------------------------------
            # CONTINUOUS STEP
            # ---------------------------------------------------------
            if len(cont_dims) > 0:
                def f_cont(x_cont_):
                    x_full = torch.tensor(x, dtype=x_cont_.dtype, device=x_cont_.device)
                    cont_dims_tensor = torch.tensor(cont_dims, dtype=torch.long, device=x_cont_.device)
                    x_full = x_full.index_copy(0, cont_dims_tensor, x_cont_)
                    if x_full.ndim == 1:
                        x_full = x_full.unsqueeze(0)
                    return -f(x_full).squeeze()

                for _ in range(interval):
                    old_x_cont = x_cont_torch.clone().detach() 
                    optimizer.zero_grad()
                    acq = f_cont(x_cont_torch).float()

                    try:
                        acq.backward()
                        optimizer.step()
                    except RuntimeError:
                        logging.warning('Exception occurred during backpropagation. NaN encountered?')
                        pass

                    with torch.no_grad():
                        if torch.isnan(x_cont_torch).any():
                            x_cont_torch.data = old_x_cont
                            break 
                        x_cont_torch.data = torch.max(torch.min(x_cont_torch, ub_t), lb_t)

                x_cont = x_cont_torch.detach().cpu().numpy()
                x[cont_dims] = x_cont 

                with torch.no_grad():
                    if isinstance(x, np.ndarray):
                        acq_x = f(torch.tensor(x, dtype=torch.float32)).detach().cpu().numpy()
                    else:
                        acq_x = f(x).detach().cpu().numpy()


            # ---------------------------------------------------------
            # CATEGORICAL STEP
            # ---------------------------------------------------------
            if len(cat_dims) > 0:
                for j in range(interval):
                    is_valid = False
                    tol_ = tol
                    while not is_valid:
                        # neighbour = sample_neighbour_ordinal(x_cat, config)
                        # if 0 <= compute_hamming_dist_ordinal(x_center[cat_dims], neighbour, config) <= max_hamming_dist:
                        #     is_valid = True
                        # Inside your inner local search thread loop:
                        # neighbour = sample_neighbour_mixed(x, binary_dims, ordinal_dims, ordinal_config)
                        # total_discrete_dist = compute_mixed_discrete_distance(
                        #     x_center, neighbour, binary_dims, ordinal_dims
                        # )
    
                        # if 0 < total_discrete_dist <= max_hamming_dist:
                        #     is_valid = True
                        # else:
                        #     tol_ -= 1

                        neighbour = sample_neighbour_mixed(x, binary_dims, ordinal_dims, ordinal_config, max_hamming_dist, use_log_warp=use_log_warp)
                        
                        # Pass ordinal_config to normalize the distance
                        total_discrete_dist = compute_mixed_discrete_distance(
                            x_center, neighbour, binary_dims, ordinal_dims, ordinal_config
                        )
                        if 0 < total_discrete_dist <= max_hamming_dist:
                            is_valid = True
                        else:
                            tol_ -= 1
                            
                    if tol_ < 0:
                        logging.info("Tolerance exhausted on this local search thread.")
                        break
                    
                    # FAST EVALUATION: Overwrite slice of the global array directly
                    # x_eval = x.copy()
                    # x_eval[cat_dims] = neighbour
                    # FAST EVALUATION
                    x_eval = neighbour.copy() # Since neighbour is already the full proposed vector 
                    if isinstance(x_eval, np.ndarray):
                        acq_neighbour = f(torch.tensor(x_eval, dtype=torch.float32)).detach().cpu().numpy()
                    else:
                        acq_neighbour = f(x_eval).detach().cpu().numpy()
    
                    if acq_neighbour > acq_x:
                        # x_cat = deepcopy(neighbour)
                        # x[cat_dims] = x_cat  # Sync global array
                        # acq_x = acq_neighbour
                        # UPDATE 1: Overwrite the full global tracking array
                        x = deepcopy(neighbour)
    
                        # UPDATE 2: Keep the local slice reference synchronized
                        x_cat = x[cat_dims] 
    
                        acq_x = acq_neighbour
    
    
            n_step += interval

        # Optionally place after the while loop if you want cleanup:
        if len(cont_dims) > 0:
            del x_cont_torch, optimizer 
        # x is already in the correct global format, no need to hstack!
        return x, acq_x

    X, fX = [], []
    for i in range(n_restart):
        res = _interleaved_search(x0[i, :])
        X.append(res[0])
        fX.append(res[1])
        
    top_indices = np.argpartition(np.array(fX).flatten(), -batch_size)[-batch_size:]
    return np.array([X[i] for i in top_indices]), np.array(fX).flatten()[top_indices]