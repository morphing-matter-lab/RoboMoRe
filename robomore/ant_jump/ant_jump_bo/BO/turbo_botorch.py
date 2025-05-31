from copy import deepcopy
from hashlib import new
import os
import math
import time
import numpy as np
from dataclasses import dataclass

import torch
from botorch.acquisition import qExpectedImprovement
# from botorch.fit import fit_gpytorch_model
from botorch.generation import MaxPosteriorSampling
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.test_functions import Ackley
from botorch.utils.transforms import unnormalize
from torch.quasirandom import SobolEngine
import torch.nn.functional as F
import gpytorch
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel, LinearKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import HorseshoePrior





device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float
SMOKE_TEST = os.environ.get("SMOKE_TEST")



@dataclass
class TurboState:
    dim: int
    batch_size: int
    length: float = 0.8
    length_min: float = 0.5 ** 7
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = float("nan")  # Note: Post-initialized
    success_counter: int = 0
    success_tolerance: int = 10  # Note: The original paper uses 3
    best_value: float = -float("inf")
    restart_triggered: bool = False

    def __post_init__(self):
        self.failure_tolerance = math.ceil(
            min(1000/self.batch_size, max([4.0 / self.batch_size, float(self.dim) / self.batch_size]))
        )


def update_state(state, Y_next):
    if max(Y_next) > state.best_value + 1e-3 * math.fabs(state.best_value):
        state.success_counter += 1
        state.failure_counter = 0
    else:
        state.success_counter = 0
        state.failure_counter += 1
    print(state.success_counter, state.success_tolerance, state.failure_counter, state.failure_tolerance)
    if state.success_counter == state.success_tolerance:  # Expand trust region
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0
    elif state.failure_counter == state.failure_tolerance:  # Shrink trust region
        state.length /= 2.0
        state.failure_counter = 0

    state.best_value = max(state.best_value, max(Y_next).item())
    if state.length < state.length_min:
        # state.restart_triggered = True
        state.length = 0.8
    print(f'Current length: {state.length}')
    return state





def generate_cand(
        dim=None,
        state=None,
        X=None,  # Evaluated points on the domain [0, 1]^d
        Y=None,  # Function values
        x_center=None,
        model=None,
        n_candidates=None,
        bound = None
        ):
    
    if n_candidates is None:
        n_candidates = 5000

    # Scale the TR to be proportional to the lengthscales
    if state:
        if x_center is None:
            assert X.min() >= 0.0 and X.max() <= 1.0 and torch.all(torch.isfinite(Y))
            x_center = X[Y.argmax(), :].clone()
            dim = X.shape[-1]
        else:
            dim = x_center.shape[-1]
        state.center = x_center
        
        if model is not None:
            weights = model.covar_module.base_kernel.lengthscale.squeeze().detach()
            weights = weights / weights.mean()
            weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
        else:
            weights = torch.ones(dim).to(device=x_center.device, dtype=x_center.dtype)
        print('tr weights', weights.mean().item(), weights.std().item(), weights.max().item(), weights.min().item())
        if bound is not None:
            tr_lb = torch.clamp(x_center - weights * state.length / 2.0, bound[0], bound[1])
            tr_ub = torch.clamp(x_center + weights * state.length / 2.0, bound[0], bound[1])
        else:
            tr_lb = torch.clamp(x_center - weights * state.length / 2.0, 0.0, 1.0)
            tr_ub = torch.clamp(x_center + weights * state.length / 2.0, 0.0, 1.0)
        print('inside cand_tr', tr_lb.ravel()[:10], tr_ub.ravel()[:10])
    else:
        assert dim is not None
    
    sobol = SobolEngine(dim, scramble=True)
    pert = sobol.draw(n_candidates).to(dtype=dtype, device=device)
    if state:
        pert = tr_lb + (tr_ub - tr_lb) * pert
        # # Create a perturbation mask
        # prob_perturb = min(20.0 / dim, 1.0)
        # # prob_perturb = 1
        # mask = (
        #     torch.rand(n_candidates, dim, dtype=dtype, device=device)
        #     <= prob_perturb
        # )
        # ind = torch.where(mask.sum(dim=1) == 0)[0]
        # if dim > 1:
        #     mask[ind, torch.randint(0, dim - 1, size=(len(ind),), device=device)] = 1

        # # Create candidate points from the perturbations and the mask        
        # X_cand = x_center.expand(n_candidates, dim).clone()
        # # print(X_cand.dtype, pert.dtype)
        X_cand = pert
    else:
        X_cand = pert
    # if state:
    #     return X_cand, state
    # else:
    return X_cand
    

def generate_batch(
    state,
    model,  # GP model
    X,  # Evaluated points on the domain [0, 1]^d
    Y,  # Function values
    batch_size,
    n_candidates=None,  # Number of candidates for Thompson sampling
    num_restarts=10,
    raw_samples=512,
    acqf="ts",  # "ei" or "ts"
    use_mcmc=False
):
    assert acqf in ("ts", "ei")
    assert X.min() >= 0.0 and X.max() <= 1.0 and torch.all(torch.isfinite(Y))
    if n_candidates is None:
        n_candidates = 5000

    # Scale the TR to be proportional to the lengthscales
    x_center = X[Y.argmax(), :].clone()
    try:
        weights = model.covar_module.base_kernel.lengthscale.squeeze().detach()
        weights = weights / weights.mean()
        weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
        tr_lb = torch.clamp(x_center - weights * state.length / 2.0, 0.0, 1.0)
        tr_ub = torch.clamp(x_center + weights * state.length / 2.0, 0.0, 1.0)
    except: # Linear kernel
        weights = 1
        tr_lb = torch.clamp(x_center - state.length / 2.0, 0.0, 1.0)
        tr_ub = torch.clamp(x_center + state.length / 2.0, 0.0, 1.0)
    


    dim = X.shape[-1]
    sobol = SobolEngine(dim, scramble=True)
    pert = sobol.draw(n_candidates).to(dtype=dtype, device=device)
    pert = tr_lb + (tr_ub - tr_lb) * pert

    # Create a perturbation mask
    prob_perturb = min(20.0 / dim, 1.0)
    # prob_perturb = 1
    mask = (
        torch.rand(n_candidates, dim, dtype=dtype, device=device)
        <= prob_perturb
    )
    ind = torch.where(mask.sum(dim=1) == 0)[0]
    mask[ind, torch.randint(0, dim - 1, size=(len(ind),), device=device)] = 1

    # Create candidate points from the perturbations and the mask        
    X_cand = x_center.expand(n_candidates, dim).clone()

    X_cand[mask] = pert[mask]

    # Sample on the candidate points
    thompson_sampling = MaxPosteriorSampling(model=model, replacement=False)
    with torch.no_grad():  # We don't need gradients when using TS
        X_next = thompson_sampling(X_cand, num_samples=batch_size)
    
           
    return X_next


def generate_batch_multiple_tr(
    state,
    model,  # GP model
    X,  # Evaluated points on the domain [0, 1]^d
    Y,  # Function values
    batch_size,
    n_candidates=None,  # Number of candidates for Thompson sampling
    num_restarts=10,
    raw_samples=512,
    acqf="ts",  # "ei" or "ts"
    use_mcmc=False
):
    assert acqf in ("ts", "ei")
    tr_num = len(state)

    for tr_idx in range(tr_num):
        assert X[tr_idx].min() >= 0.0 and X[tr_idx].max() <= 1.0 and torch.all(torch.isfinite(Y[tr_idx]))
    if n_candidates is None:
        n_candidates = 5000
    dim = X[0].shape[1]
    # Scale the TR to be proportional to the lengthscales
    X_cand = torch.zeros(tr_num, n_candidates, dim).to(device=device, dtype=dtype)
    Y_cand = torch.zeros(tr_num, n_candidates, batch_size).to(device=device, dtype=dtype)
    tr_lb = torch.zeros(tr_num, dim).to(device=device, dtype=dtype)
    tr_ub = torch.zeros(tr_num, dim).to(device=device, dtype=dtype)
    for tr_idx in range(tr_num):
        x_center = X[tr_idx][Y[tr_idx].argmax(), :].clone()
        try:
            weights = model[tr_idx].covar_module.base_kernel.lengthscale.squeeze().detach()
            weights = weights / weights.mean()
            weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
            tr_lb[tr_idx] = torch.clamp(x_center - weights * state[tr_idx].length / 2.0, 0.0, 1.0)
            tr_ub[tr_idx] = torch.clamp(x_center + weights * state[tr_idx].length / 2.0, 0.0, 1.0)
        except: # Linear kernel
            weights = 1
            tr_lb[tr_idx] = torch.clamp(x_center - state[tr_idx].length / 2.0, 0.0, 1.0)
            tr_ub[tr_idx] = torch.clamp(x_center + state[tr_idx].length / 2.0, 0.0, 1.0)
        


        # dim = X.shape[-1]
        sobol = SobolEngine(dim, scramble=True)
        pert = sobol.draw(n_candidates).to(dtype=dtype, device=device)
        pert = tr_lb[tr_idx] + (tr_ub[tr_idx] - tr_lb[tr_idx]) * pert

        # Create a perturbation mask
        prob_perturb = min(20.0 / dim, 1.0)
        # prob_perturb = 1
        mask = (
            torch.rand(n_candidates, dim, dtype=dtype, device=device)
            <= prob_perturb
        )
        ind = torch.where(mask.sum(dim=1) == 0)[0]
        mask[ind, torch.randint(0, dim - 1, size=(len(ind),), device=device)] = 1

        # Create candidate points from the perturbations and the mask        
        X_cand[tr_idx] = x_center.expand(n_candidates, dim).clone()

        X_cand[tr_idx][mask] = pert[mask]

        # # Sample on the candidate points
        # thompson_sampling = MaxPosteriorSampling(model=model, replacement=False)
        # with torch.no_grad():  # We don't need gradients when using TS
        #     X_next[tr_idx] = thompson_sampling(X_cand, num_samples=batch_size)
        posterior = model[tr_idx].posterior(X_cand[tr_idx])
        # num_samples x batch_shape x N x m
        samples = posterior.rsample(sample_shape=torch.Size([batch_size]))
        # print('sample shape', samples.shape)
        samples = samples.reshape([batch_size, n_candidates])
        Y_cand[tr_idx] = samples.permute(1,0)
        # recover from normalized value
        Y_cand[tr_idx] = Y[tr_idx].mean() + Y_cand[tr_idx] * Y[tr_idx].std()
        
    # Compare across trust region
    y_cand = Y_cand.detach().cpu().numpy()
    X_next = torch.zeros(batch_size, dim).to(device=device, dtype=dtype)
    tr_idx_next = np.zeros(batch_size)
    for k in range(batch_size):
        i, j = np.unravel_index(np.argmax(y_cand[:, :, k]), (tr_num, n_candidates))
        # print('select', i, j)
        X_next[k] = X_cand[i, j]
        tr_idx_next[k] = i
        assert np.isfinite(y_cand[i, j, k])  # Just to make sure we never select nan or inf
        # Make sure we never pick this point again
        y_cand[i, j, :] = -np.inf
           
    return X_next, tr_idx_next