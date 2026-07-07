# Implementation of various kernels

from gpytorch.kernels import Kernel
from gpytorch.kernels.matern_kernel import MaternKernel
from gpytorch.kernels.rbf_kernel import RBFKernel
from gpytorch.constraints import Interval
import torch
import numpy as np


class MixtureKernel(Kernel):
    """
    The implementation of the mixed categorical and continuous kernel first proposed in CoCaBO, but re-implemented
    in gpytorch.

    Note that gpytorch uses the pytorch autodiff engine, and there is no need to manually define the derivatives of
    the kernel hyperparameters w.r.t the log-marinal likelihood as in the gpy implementation.
    """
    has_lengthscale = True

    def __init__(self, categorical_dims,
                 continuous_dims,
                 integer_dims=None,
                 lamda=0.5,
                 categorical_kern_type='transformed_overlap',
                 continuous_kern_type='mat52',
                 categorical_lengthscale_constraint=None,
                 continuous_lengthscale_constraint=None,
                 categorical_ard=True,
                 continuous_ard=True,
                 **kwargs):
        """

        Parameters
        ----------
        categorical_dims: the dimension indices that are categorical/discrete
        continuous_dims: the dimension indices that are continuous
        integer_dims: the **continuous indices** that additionally require integer constraint.
        lamda: \in [0, 1]. The trade-off between product and additive kernels. If this argument is not supplied, then
            lambda will be optimised as if it is an additional kernel hyperparameter
        categorical_kern_type: 'overlap', 'type2'
        continuous_kern_type: 'rbf' or 'mat52' (Matern 5/2)
        categorical_lengthscale_constraint: if supplied, the constraint on the lengthscale of the categorical kernel
        continuous_lengthscale_constraint: if supplied the constraint on the lengthscale of the continuous kernel
        categorical_ard: bool: whether to use Automatic Relevance Determination (ARD) for categorical dimensions
        continuous_ard: bool: whether to use ARD for continouous dimensions
        kwargs: additional parameters.
        """
        super(MixtureKernel, self).__init__(has_lengthscale=True, **kwargs)
        self.optimize_lamda = lamda is None
        self.fixed_lamda = lamda if not self.optimize_lamda else None
        self.categorical_dims = categorical_dims
        self.continuous_dims = continuous_dims
        if integer_dims is not None:
            integer_dims_np = np.asarray(integer_dims).flatten()
            cont_dims_np = np.asarray(self.continuous_dims).flatten()
            if not np.all(np.in1d(integer_dims_np, cont_dims_np)):
                raise ValueError("if supplied, all continuous dimensions with integer constraint must be themselves "
                                 "contained in the continuous_dimensions!")
            # Convert the integer dims in terms of indices of the continous dims
            integer_dims = np.where(np.in1d(self.continuous_dims, integer_dims))[0]

        self.register_parameter(name='raw_lamda', parameter=torch.nn.Parameter(torch.ones(1)))
        # The lambda must be between 0 and 1.
        self.register_constraint('raw_lamda', Interval(0., 1.))

        # Initialise the
        if categorical_kern_type == 'overlap':
            self.categorical_kern = CategoricalOverlap(lengthscale_constraint=categorical_lengthscale_constraint,
                                                       ard_num_dims=len(categorical_dims) if categorical_ard else None)
        elif categorical_kern_type == 'transformed_overlap':
            self.categorical_kern = TransformedCategorical(lengthscale_constraint=categorical_lengthscale_constraint,
                                                           ard_num_dims=len(
                                                                categorical_dims) if categorical_ard else None)
        else:
            raise NotImplementedError("categorical kernel type %s is not implemented. " % categorical_kern_type)

        # By default, we use the Matern 5/2 kernel
        if continuous_kern_type == 'mat52':
            self.continuous_kern = WrappedMatern(nu=2.5, ard_num_dims=len(continuous_dims) if continuous_ard else None,
                                                 integer_dims=integer_dims,
                                                 lengthscale_constraint=continuous_lengthscale_constraint)
        elif continuous_kern_type == 'rbf':
            self.continuous_kern = WrappedRBF(ard_num_dims=len(continuous_dims) if continuous_ard else None,
                                              integer_dims=integer_dims,
                                              lengthscale_constraint=continuous_lengthscale_constraint)
        else:
            raise NotImplementedError("continuous kernel type %s is not implemented. " % continuous_kern_type)

    @property
    def lamda(self):
        if self.optimize_lamda:
            return self.raw_lamda_constraint.transform(self.raw_lamda)
        else:
            return self.fixed_lamda

    @lamda.setter
    def lamda(self, value):
        self._set_lamda(value)

    def _set_lamda(self, value):
        if self.optimize_lamda:
            if not isinstance(value, torch.Tensor):
                value = torch.as_tensor(value).to(self.raw_lamda)
            self.initialize(raw_lamda=self.raw_lamda_constraint.inverse_transform(value))
        else:
            # Manually restrict the value of lamda between 0 and 1.
            if value <= 0:
                self.fixed_lamda = 0.
            elif value >= 1:
                self.fixed_lamda = 1.
            else:
                self.fixed_lamda = value

    def forward(self, x1, x2, diag=False,
                x1_cont=None, x2_cont=None, **params):
        """
        Note that here I also give options to pass the categorical and continuous inputs separately (instead of jointly)
        because the categorical dimensions will not be differentiable, and thus there would be problems when we optimize
        the acquisition function.

        When passed separately, x1 and x2 refer the categorical (non-differentiable) data, whereas x1_cont and x2_cont
        are the continuous (differentiable) data.
        Parameters
        ----------
        x1
        x2
        diag
        x1_cont
        x2_cont
        params

        Returns
        -------

        """
        if x1_cont is None and x2_cont is None:
            assert x1.shape[1] == len(self.categorical_dims) + len(self.continuous_dims), \
                'dimension mismatch. Expected number of dimensions %d but got %d in x1' % \
                (len(self.categorical_dims) + len(self.continuous_dims), x1.shape[1])
            x1_cont, x2_cont = x1[:, self.continuous_dims], x2[:, self.continuous_dims]
            # the categorical kernels are not differentiable w.r.t inputs, detach them to ensure the computing graph of
            # the autodiff engine is not broken.
            x1_cat, x2_cat = x1[:, self.categorical_dims].detach(), x2[:, self.categorical_dims].detach()
        # else:
        #     assert x1.shape[1] == len(self.categorical_dims)
        #     assert x1_cont.shape[1] == len(self.continuous_dims)
        #     x1_cat, x2_cat = x1, x2
        # # same in cocabo.
        # return (1. - self.lamda) * (self.categorical_kern.forward(x1_cat, x2_cat, diag, **params) +
        #                             self.continuous_kern.forward(x1_cont, x2_cont, diag, **params)) + \
        #        self.lamda * self.categorical_kern.forward(x1_cat, x2_cat, diag, **params) * \
        #        self.continuous_kern.forward(x1_cont, x2_cont, diag, **params)
        else:
            assert x1.shape[1] == len(self.categorical_dims)
            assert x1_cont.shape[1] == len(self.continuous_dims)
            x1_cat, x2_cat = x1, x2
            
        # --- FIX FOR PURELY CATEGORICAL OR PURELY CONTINUOUS SPACES ---
        if x1_cont.shape[1] == 0:
            return self.categorical_kern.forward(x1_cat, x2_cat, diag, **params)
        if x1_cat.shape[1] == 0:
            return self.continuous_kern.forward(x1_cont, x2_cont, diag, **params)
        # --------------------------------------------------------------

        # same in cocabo.
        return (1. - self.lamda) * (self.categorical_kern.forward(x1_cat, x2_cat, diag, **params) +
                                    self.continuous_kern.forward(x1_cont, x2_cont, diag, **params)) + \
               self.lamda * self.categorical_kern.forward(x1_cat, x2_cat, diag, **params) * \
               self.continuous_kern.forward(x1_cont, x2_cont, diag, **params)


def wrap(x1, x2, integer_dims):
    """The wrapping transformation for integer dimensions according to Garrido-Merchán and Hernández-Lobato (2020)."""
    if integer_dims is not None:
        for i in integer_dims:
            x1[:, i] = torch.round(x1[:, i])
            x2[:, i] = torch.round(x2[:, i])
    return x1, x2


class WrappedMatern(MaternKernel):
    """Matern kernels wrapped integer type of inputs according to
    Garrido-Merchán and Hernández-Lobato in
    "Dealing with Categorical and Integer-valued Variables in Bayesian Optimization with Gaussian Processes"

    Note: we deal with the categorical-valued variables using the kernels specifically used to deal with
    categorical variables (instead of the one-hot transformation).
    """

    def __init__(self, integer_dims=None, **kwargs):
        super(WrappedMatern, self).__init__(**kwargs)
        self.integer_dims = integer_dims

    def forward(self, x1, x2, diag=False, **params):
        x1, x2 = wrap(x1, x2, self.integer_dims)
        return super().forward(x1, x2, diag=diag, **params)


class WrappedRBF(RBFKernel, WrappedMatern):
    """Similar to above, but applied to RBF."""

    def __init__(self, integer_dims=None, **kwargs):
        super(WrappedRBF, self).__init__(**kwargs)
        self.integer_dims = integer_dims

    def forward(self, x1, x2, diag=False, **params):
        x1, x2 = wrap(x1, x2, self.integer_dims)
        return super().forward(x1, x2, diag=diag, **params)


class CategoricalOverlap(Kernel):
    """Implementation of the categorical overlap kernel.
    This is the most basic form of the categorical kernel that essentially invokes a Kronecker delta function
    between any two elements.
    """

    has_lengthscale = True
    def __init__(self, **kwargs):
        super(CategoricalOverlap, self).__init__(has_lengthscale=True, **kwargs)

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        # First, convert one-hot to ordinal representation

        diff = x1[:, None] - x2[None, :]
        # nonzero location = different cat
        diff[torch.abs(diff) > 1e-5] = 1
        # invert, to now count same cats
        diff1 = torch.logical_not(diff).float()
        if self.ard_num_dims is not None and self.ard_num_dims > 1:
            k_cat = torch.sum(self.lengthscale * diff1, dim=-1) / torch.sum(self.lengthscale)
        else:
            # dividing by number of cat variables to keep this term in range [0,1]
            k_cat = torch.sum(diff1, dim=-1) / x1.shape[1]
        if diag:
            return torch.diag(k_cat).float()
        return k_cat.float()


class TransformedCategorical(CategoricalOverlap):
    """
    Second kind of transformed kernel of form:
    $$ k(x, x') = \exp(\frac{\lambda}{n}) \sum_{i=1}^n [x_i = x'_i] )$$ (if non-ARD)
    or
    $$ k(x, x') = \exp(\frac{1}{n} \sum_{i=1}^n \lambda_i [x_i = x'_i]) $$ if ARD
    """

    has_lengthscale = True

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, exp='rbf', **params):
        diff = x1[:, None] - x2[None, :]
        diff[torch.abs(diff) > 1e-5] = 1
        diff1 = torch.logical_not(diff).float()

        def rbf(d, ard):
            if ard:
                return torch.exp(torch.sum(d * self.lengthscale, dim=-1) / torch.sum(self.lengthscale))
            else:
                return torch.exp(self.lengthscale * torch.sum(d, dim=-1) / x1.shape[1])

        def mat52(d, ard):
            raise NotImplementedError

        if exp == 'rbf':
            k_cat = rbf(diff1, self.ard_num_dims is not None and self.ard_num_dims > 1)
        elif exp == 'mat52':
            k_cat = mat52(diff1, self.ard_num_dims is not None and self.ard_num_dims > 1)
        else:
            raise ValueError('Exponentiation scheme %s is not recognised!' % exp)
        if diag:
            return torch.diag(k_cat).float()
        return k_cat.float()


class OrdinalKernel(Kernel):
    """
    The ordinal version of TransformedCategorical2 kernel (replace the Kronecker delta with
    the distance metric).
    config: the number of vertices per dimension
    """
    def __init__(self, config, **kwargs):
        super(OrdinalKernel, self).__init__(has_lengthscale=True, **kwargs)
        if not isinstance(config, torch.Tensor):
            config = torch.tensor(config).view(-1)
        self.config = config

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        # expected x1 and x2 are of shape N x D respectively
        diff = (x1[:, None] - x2[None, :]) / self.config
        dist = 1. - torch.abs(diff)
        if self.ard_num_dims is not None and self.ard_num_dims > 1:
            k_cat = torch.exp(
                torch.sum(
                    dist * self.lengthscale, dim=-1
                ) / torch.sum(self.lengthscale)
            )
        else:
            k_cat = torch.exp(
                self.lengthscale * torch.sum(dist, dim=-1) / x1.shape[1]
            )
        if diag:
            return torch.diag(k_cat).float()
        return k_cat.float()


if __name__ == '__main__':
    # Test whether the ordinal kernel is doing ok
    import numpy as np
    import matplotlib.pyplot as plt
    x1 = torch.tensor([[13.,  4.],
        [43., 15.],
        [32., 19.],
        [41.,  9.],
        [47., 44.],
        [48., 21.],
        [15., 24.],
        [20., 13.],
        [36., 46.],
        [19., 17.],
        [35.,  6.],
        [39., 50.],
        [24., 10.],
        [45., 18.],
        [29.,  3.],
        [17., 27.],
        [25., 16.],
        [37., 29.],
        [16.,  2.],
        [ 3., 38.]])

    o = OrdinalKernel(config=[51, 51])
    o.lengthscale = 1.
    K = o.forward(x1, x1).detach().numpy()
    plt.imshow(K)
    plt.colorbar()
    plt.show()


# Add this to the bottom of kernels.py

import torch
from gpytorch.kernels import Kernel
import gpytorch

import torch
from gpytorch.kernels import Kernel

class UnifiedL1DiscreteKernel(Kernel):
    """
    Unified L1 similarity kernel for mixed binary and ordinal spaces.
    
    Exploits the mathematical equivalence: for a binary variable with range C=1,
    the normalized L1 similarity equals the Kronecker delta (categorical overlap).
    This eliminates the need for a lambda mixture parameter.
    
    Args:
        config: 1D tensor of length D.
                - For binary variables: config[i] = 1  (range of {0,1})
                - For ordinal variables: config[j] = max_value = n_choices - 1
    """
    has_lengthscale = True

    def __init__(self, config,use_log_warp=False, **kwargs):
        if not isinstance(config, torch.Tensor):
            config = torch.tensor(config, dtype=torch.float32)
            
        # FIX 1: Safely handle ard_num_dims to prevent kwargs collision
        kwargs.setdefault("ard_num_dims", config.shape[0])
        # Explicitly enforce a strict, identical constraint on all dimensions
        # This prevents the GP from artificially silencing the ordinal dimension
        # lengthscale_constraint = gpytorch.constraints.Interval(0.01, 2.0)
        # kwargs.setdefault("lengthscale_constraint", lengthscale_constraint)
        kwargs.setdefault("ard_num_dims", config.shape[0])
        self.use_log_warp = use_log_warp
        super().__init__(**kwargs)
        
       

        # # Register as buffer: describes the geometric space, not a learned parameter
        # self.register_buffer("config", config.clamp_min(1.0).view(1, -1))  # (1, D)
        # --- REPLACE WITH THIS LINE ---
        # Clamps the minimum range to 1.0 for binary vars, 
        # preserves max range (e.g., 9) for integer vars.
        self.register_buffer("config", config.clamp_min(1.0).view(1, -1))
        
    # def forward(self, x1, x2, diag=False, **params):
    #     ls = self.lengthscale.view(1, -1) 
        
    #     if diag:
    #         diff_diag = torch.abs(x1 - x2) / self.config
    #         # k = exp( - sum (distance * ls) )
    #         k_diag = torch.exp(-(diff_diag * ls).sum(dim=-1))
    #         return k_diag.float()

    #     diff = torch.abs(x1.unsqueeze(-2) - x2.unsqueeze(-3)) / self.config
        
    #     # REMOVE the 1.0 - diff similarity inversion
    #     # REMOVE the division by ls_sum
    #     # Use standard exponential decay on the L1 distance
    #     k = torch.exp(-(diff * ls).sum(dim=-1))
        
    #     return k.float()

    def forward(self, x1, x2, diag=False, **params):
        ls = self.lengthscale.view(1, -1) 
        is_ordinal = (self.config > 1.5).float()
        is_binary = 1.0 - is_ordinal

        # Linear distance for binary is always applied
        diff_bin = torch.abs(x1.unsqueeze(-2) - x2.unsqueeze(-3)) / self.config

        if getattr(self, "use_log_warp", True):
            # Log-warped distance for ordinal
            num = torch.abs(torch.log1p(x1.unsqueeze(-2)) - torch.log1p(x2.unsqueeze(-3)))
            den = torch.log1p(self.config)
            diff_ord = num / den
        else:
            # Standard L1 normalized gap for ordinal
            diff_ord = torch.abs(x1.unsqueeze(-2) - x2.unsqueeze(-3)) / self.config

        diff = (is_binary * diff_bin) + (is_ordinal * diff_ord)
        k = torch.exp(-(diff * ls).sum(dim=-1))

        return k.float()

    # def forward(self, x1, x2, diag=False, **params):
    #     # FIX 2: Safely flatten lengthscale for broadcasting (handles ARD and non-ARD)
    #     # self.lengthscale is (1, 1, D) for ARD, or (1, 1, 1) for non-ARD
    #     ls = self.lengthscale.view(1, -1) 
        
    #     if diag:
    #         # Compute diagonal directly: O(N * D)
    #         diff_diag = torch.abs(x1 - x2) / self.config
    #         sim_diag  = (1.0 - diff_diag).clamp_min(0.0)
            
    #         k_diag = torch.exp(
    #             (sim_diag * ls).sum(dim=-1) / ls.sum()
    #         )
    #         return k_diag.float()

    #     # # Full matrix: (N, M, D)
    #     # diff = torch.abs(x1.unsqueeze(1) - x2.unsqueeze(0)) / self.config.unsqueeze(0)
    #     # similarity = (1.0 - diff).clamp_min(0.0)

    #     # FULL MATRIX FIX: Use negative unsqueezing to preserve batch dimensions
    #     # x1: (..., N, D) -> (..., N, 1, D)
    #     # x2: (..., M, D) -> (..., 1, M, D)
    #     # diff: (..., N, M, D)
    #     diff = torch.abs(x1.unsqueeze(-2) - x2.unsqueeze(-3)) / self.config
    #     similarity = (1.0 - diff).clamp_min(0.0)

    #     # # Weighted average similarity, then exponentiate
    #     # k = torch.exp(
    #     #     (similarity * self.lengthscale).sum(dim=-1) / self.lengthscale.sum()
    #     # )

    #     # In UnifiedL1DiscreteKernel.forward
    #     ls_sum = ls.sum().clamp_min(1e-9) # Prevent division by zero
    #     k = torch.exp((similarity * ls).sum(dim=-1) / ls_sum)

    #     return k.float()
class DiscreteMixtureKernel(Kernel):
    """
    CoCaBO-style Mixture Kernel strictly for discrete spaces.
    Mixes a binary kernel and an ordinal kernel using a learnable lambda.
    """
    has_lengthscale = True

    def __init__(self, binary_dims, ordinal_dims, ordinal_config, use_log_warp=True, lamda=0.5, **kwargs):
        super().__init__(**kwargs)
        
        # Register the learnable mixture weight
        self.register_parameter(name='raw_lamda', parameter=torch.nn.Parameter(torch.tensor(lamda)))
        self.register_constraint('raw_lamda', Interval(0., 1.))
        
        # Initialize the sub-kernels
        self.binary_kern = UnifiedL1DiscreteKernel(
            config=[1.0] * len(binary_dims),
            active_dims=binary_dims,
            **kwargs
        )
        
        self.ordinal_kern = UnifiedL1DiscreteKernel(
            config=[max(1, c - 1) for c in ordinal_config] if ordinal_config else [1.0],
            active_dims=ordinal_dims,
            use_log_warp=use_log_warp,
            **kwargs
        )

    @property
    def lamda(self):
        return self.raw_lamda_constraint.transform(self.raw_lamda)

    def forward(self, x1, x2, diag=False, **params):
        k_bin = self.binary_kern(x1, x2, diag=diag, **params)
        k_ord = self.ordinal_kern(x1, x2, diag=diag, **params)
        
        # (1 - lambda) * (K_bin + K_ord) + lambda * (K_bin * K_ord)
        return (1.0 - self.lamda) * (k_bin + k_ord) + self.lamda * (k_bin * k_ord)