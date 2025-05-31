import numpy as np
import torch
from torch.quasirandom import SobolEngine
import gpytorch
from gpytorch.models import ExactGP, ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution

from gpytorch.mlls import VariationalELBO
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models import SingleTaskGP
# from botorch.fit import fit_gpytorch_model
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf
from botorch.acquisition import ExpectedImprovement, UpperConfidenceBound, qExpectedImprovement, qUpperConfidenceBound, qProbabilityOfImprovement
# from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.acquisition.analytic import LogExpectedImprovement
from botorch.sampling.normal import SobolQMCNormalSampler

from gp_model import BoTorchWrapper, train_fvgp
from turbo_botorch import generate_cand


acqf_dict = {
    'ei': ExpectedImprovement,
    'ucb': UpperConfidenceBound,
}

def thompson_sampling(gp, likelihood, cand_size=5000, cand_x=None, batch_size=1, return_acq=False, 
                      device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), dtype = torch.float):
    if cand_x is None:
        sobol = SobolEngine(gp.dim, scramble=True)
        cand_x = sobol.draw(cand_size).to(dtype=dtype, device=device)
    else:
        assert len(cand_x) == cand_size
    # print(gp(cand_x))
    posterior = likelihood(gp(cand_x))
    samples = posterior.rsample(sample_shape=torch.Size([batch_size])).reshape(batch_size, cand_size)
    Y_cand = samples.permute(1,0)
    del samples
    y_cand = Y_cand.detach().cpu().numpy()
    del Y_cand
    X_next = torch.zeros(min(batch_size, cand_size), cand_x.shape[1]).to(device=device, dtype=dtype)
    Acq_next = torch.zeros(min(batch_size, cand_size), 1).to(device=device, dtype=dtype)
    max_indices = []
    for k in range(batch_size):
        j = np.argmax(y_cand[:, k])
        # print('select', i, j)
        X_next[k] = cand_x[j]
        Acq_next[k] = torch.from_numpy(y_cand[j, k].reshape(-1, 1)).to(device=device, dtype=dtype)
        max_indices.append(j)
        assert np.isfinite(y_cand[j, k])  # Just to make sure we never select nan or inf
        # Make sure we never pick this point again
        
        y_cand[j, :] = -np.inf
        
    if return_acq:
        return X_next, Acq_next
    else:
        return X_next   


def upper_confidence_bound(gp,
                           likelihood, 
                           cand_size=5000, cand_x=None, beta=2, batch_size=1,
                           device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), dtype = torch.float):
    if cand_x is None:
        sobol = SobolEngine(gp.dim, scramble=True)
        cand_x = sobol.draw(cand_size).to(dtype=dtype, device=device)
    else:
        assert len(cand_x) == cand_size
    # print(gp(cand_x))
    posterior = gp.likelihood(gp(cand_x))
    lower, upper = posterior.confidence_region()
    means = posterior.mean.detach().cpu().numpy()
    var = (upper.detach().cpu().numpy()-lower.detach().cpu().numpy())/2
    ucb = means + beta * var
    
    return cand_x[ucb.argmax()].reshape(1, -1)

def probability_of_improvement(gp, likelihood, best_f, bound=None, return_acq=False,
                               batch_size=1, device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), dtype = torch.float):
    if bound is None:
        bound = torch.tensor([[0.0] * gp.dim, [1.0] * gp.dim], device=device, dtype=dtype)
    
    botorch_model = BoTorchWrapper(gp, likelihood)
    
    acq_func = qProbabilityOfImprovement(botorch_model, best_f=best_f)

    Acq_next = torch.zeros(0, 1).to(device=device, dtype=dtype)
    X_next, Acq_next = optimize_acqf(acq_func, bounds=bound, q=batch_size, num_restarts=10, raw_samples=512, sequential=True)
    # raise NotImplementedError
    if return_acq:
        return X_next, Acq_next
    else:
        return X_next   


def expected_improvement(gp, likelihood, best_f, bound=None, batch_size=1, return_acq=False, device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), dtype = torch.float):
    if bound is None:
        bound = torch.tensor([[0.0] * gp.dim, [1.0] * gp.dim], device=device, dtype=dtype)
    
    botorch_model = BoTorchWrapper(gp, likelihood)
    
    acq_func = LogExpectedImprovement(botorch_model, best_f=best_f)
    X_next = torch.zeros(0, gp.dim).to(device=device, dtype=dtype)
    Acq_next = torch.zeros(0, 1).to(device=device, dtype=dtype)
    for _ in range(batch_size):
        new_X, new_acq = optimize_acqf(acq_func, bounds=bound, q=1, num_restarts=10, raw_samples=512)
        
        X_next = torch.cat((X_next, new_X))
        Acq_next = torch.cat((Acq_next, new_acq.reshape(-1, 1)))
    # raise NotImplementedError
    if return_acq:
        return X_next, Acq_next
    else:
        return X_next   

def q_expected_improvement(gp, likelihood, best_f, bound=None, batch_size=1, return_acq=False, device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), dtype = torch.float):
    if bound is None:
        bound = torch.tensor([[0.0] * gp.dim, [1.0] * gp.dim], device=device, dtype=dtype)
    
    botorch_model = BoTorchWrapper(gp, likelihood)
    
    # define the qEI and qNEI acquisition modules using a QMC sampler
    qmc_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([256]))

    # for best_f, we use the best observed noisy values as an approximation
    acq_func = qExpectedImprovement(
        model=botorch_model,
        best_f=best_f,
        sampler=qmc_sampler,
    )
    
    # set sequential to True to get acq value for each x
    X_next, Acq_next = optimize_acqf(acq_func, bounds=bound, q=batch_size, num_restarts=10, raw_samples=512)
    Acq_next = torch.ones(batch_size, 1).to(device=device, dtype=dtype)* Acq_next.item()
    # raise NotImplementedError
    if return_acq:
        return X_next, Acq_next
    else:
        return X_next   

def q_uppper_confidence_bound(gp, likelihood, bound=None, batch_size=1, return_acq=False, device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), dtype = torch.float):
    if bound is None:
        bound = torch.tensor([[0.0] * gp.dim, [1.0] * gp.dim], device=device, dtype=dtype)
    
    botorch_model = BoTorchWrapper(gp, likelihood)
    
    # define the qEI and qNEI acquisition modules using a QMC sampler
    qmc_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([256]))

    # for best_f, we use the best observed noisy values as an approximation
    acq_func = qUpperConfidenceBound(
        model=botorch_model,
        beta=2,
        sampler=qmc_sampler,
    )
    try:
        X_next, Acq_next = optimize_acqf(acq_func, bounds=bound, q=batch_size, num_restarts=10, raw_samples=512, sequential=True)

    except Exception as e:
        print('cannot optimize, randomly select points', e)
        X_next = torch.rand(batch_size, gp.dim).to(device=device, dtype=dtype)
        Acq_next = torch.zeros(batch_size, 1).to(device=device, dtype=dtype)
    # raise NotImplementedError
    # return X_next
    if return_acq:
        return X_next, Acq_next
    else:
        return X_next   

def focal_acqf_opt(model, likelihood, state, x_center, acqf, 
                   train_loader, num_epochs=1000, max_loop_num=3, batch_size=1,
                   init=True,
                   tr = None,
                   cand_tr= None,
                   train_new_gp=True,
                   device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), 
                   dtype = torch.float
                   ):
    
    # x_center = 0.5 * torch.ones(1, model.dim).to(dtype=dtype, device=device) # center of the input region
    x_center = x_center.reshape(1, -1)
    print('start center', x_center.mean(),x_center.max(),x_center.min(), state.length)
    # cand_x = generate_cand(state=state, x_center=x_center, model=model)
    # def test_sample(num_pts):
    #     return generate_cand(state=state, x_center=x_center, n_candidates=num_pts, model=model)
    if tr == None:
        tr = [0, 1]
    ori_bound = tr
    if cand_tr is not None:
        cand_tr = cand_tr
    else:
        cand_tr = tr.clone()
    print('gp tr', tr[:, :10])
    print('cand tr', cand_tr[:, :10])
    
    # tr = [0, 1]
    n_per_loop = int(np.ceil(batch_size/max_loop_num))
    total_budget = batch_size
    budget_per_loop = []
    for i in range(max_loop_num):
        b = min(n_per_loop, total_budget)
        budget_per_loop.append(b)
        total_budget -= b
    
    budget_per_loop.reverse()
        
    new_X = torch.zeros(0, x_center.shape[1]).to(device=device, dtype=dtype)
    early_stop = False
    
    use_center_sparse_gp = False
    
    for loop in range(max_loop_num):
        if early_stop:
            budget_per_loop[loop] = np.sum(budget_per_loop[loop:])
        if acqf == 'ucb':
            cand_x = generate_cand(state=state, x_center=x_center, bound=cand_tr)
            new_x = upper_confidence_bound(model, likelihood, cand_x=cand_x, batch_size=max(1, budget_per_loop[loop]))
        elif 'q' in acqf or 'pi' in acqf or 'ei' in acqf:
            best_f = train_loader.dataset.tensors[1].max().item()
            # print(best_f)
            # raise NotImplementedError
            
            # weights = model.covar_module.base_kernel.lengthscale.squeeze().detach()
            # # weights = 1/weights
            # weights = weights / weights.mean()
            # weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
            weights = torch.ones(model.dim).to(device=device, dtype=dtype)
            # weights = torch.clamp(weights, 0, 1)
            tr_lb = torch.clamp(x_center - weights * state.length / 2.0, tr[0], tr[1])
            tr_ub = torch.clamp(x_center + weights * state.length / 2.0, tr[0], tr[1])
            print('inside bound', tr_lb, tr_ub)
            bound = torch.cat((tr_lb, tr_ub)).to(dtype=dtype, device=device)
            if acqf == 'ei':
                new_x = expected_improvement(model, likelihood, best_f=best_f, bound=bound, batch_size=max(1, budget_per_loop[loop]))
            elif acqf == 'qei':
                new_x = q_expected_improvement(model, likelihood, best_f=best_f, bound=bound, batch_size=max(1, budget_per_loop[loop]))
            elif acqf == 'qucb':
                new_x = q_uppper_confidence_bound(model, likelihood, bound=bound, batch_size=max(1, budget_per_loop[loop]))
            if acqf == 'pi':
                print('use pi')
                new_x = probability_of_improvement(model, likelihood, best_f=best_f, bound=bound, batch_size=max(1, budget_per_loop[loop]))
        else:
            cand_x = generate_cand(state=state, x_center=x_center, model=model, bound=cand_tr)
            new_x = thompson_sampling(model, likelihood, cand_x=cand_x, batch_size=max(1, budget_per_loop[loop]))
        # Exit loop if converge or hit the bound
        print('loop', loop, state.length, len(new_x))
        # print('loop', loop, state.length, len(new_x), new_x)
        # print(new_x)
        dist = torch.pdist(model.variational_strategy.inducing_points)
        print('dist', dist.max().item(), dist.min().item())
        # if torch.norm(new_x-x_center) < 1e-3 or torch.norm(new_x.max()-1) < 1e-3 or torch.norm(new_x.min()) < 1e-3:
        #     print('early break', loop, new_x, torch.norm(new_x-x_center), torch.norm(new_x.max()-1), torch.norm(new_x.min()))
        #     break
        
        if budget_per_loop[loop] > 0:
            new_X = torch.cat((new_X, new_x))
            
        if early_stop:
            print('early stop')
            break
        if loop < max_loop_num-1:
            state.length /= 2
            # x_center = new_x[-1].reshape(1, -1)
            # choose center point with most training point in the new search region, otherwise choose current best as center
            in_center_num = []
            train_x = train_loader.dataset.tensors[0]
            for x in new_x:
                # weights = model.covar_module.base_kernel.lengthscale.squeeze().detach()
                # # weights = 1/weights
                # weights = weights / weights.mean()
                # weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
                
                weights = torch.ones(model.dim).to(device=device, dtype=dtype)
                
                # weights = torch.clamp(weights, 0, 1)
                tr_lb = torch.clamp(x - weights * state.length / 2.0, ori_bound[0], ori_bound[1])
                tr_ub = torch.clamp(x + weights * state.length / 2.0, ori_bound[0], ori_bound[1])
                # tr = torch.cat((tr_lb, tr_ub)).to(dtype=dtype, device=device)
                lb = tr_lb.expand(len(train_x), len(tr_lb.ravel()))
                ub = tr_ub.expand(len(train_x), len(tr_ub.ravel()))
                clamp_x = torch.clamp(train_x, lb, ub)
                diff = (torch.abs(train_x-clamp_x)).sum(-1)
                if diff.min() > 0:
                    in_center_num.append(0)
                else:
                    in_center_num.append(len(diff[diff==0]))
            print('in center', in_center_num)
            
            
            
            if np.max(in_center_num) < model.variational_strategy.inducing_points.shape[0]:
                x_center = train_x[train_loader.dataset.tensors[1].argmax()].reshape(1, -1)
                print('too sparse, choose current best as center', np.max(in_center_num))
                choose_best = True
            else:
                valid_idx = [i for i in range(len(in_center_num)) if in_center_num[i] >= model.variational_strategy.inducing_points.shape[0]]
                # valid_idx = [i for i in range(len(in_center_num))]
                center_idx = np.random.choice(valid_idx, size=1)[0]
                print(valid_idx, center_idx, in_center_num[center_idx])
                x_center = new_x[center_idx].reshape(1, -1)
                choose_best = False
            
            # cand_x = generate_cand(state=state, x_center=x_center)
            # def test_sample(num_pts):
            #     return generate_cand(state=state, x_center=x_center, n_candidates=num_pts)
            
            # weights = model.covar_module.base_kernel.lengthscale.squeeze().detach()
            # # weights = 1/weights
            # weights = weights / weights.mean()
            # weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
            
            weights = torch.ones(model.dim).to(device=device, dtype=dtype)
            
            # weights = torch.clamp(weights, 0, 1)
            # slowly increase tr to include enough training point
            base_length = state.length
            in_center_num = 0
            increase_step = 0.01
            while(in_center_num < model.variational_strategy.inducing_points.shape[0]):
                tr_lb = torch.clamp(x_center - weights * base_length / 2.0, ori_bound[0], ori_bound[1])
                tr_ub = torch.clamp(x_center + weights * base_length / 2.0, ori_bound[0], ori_bound[1])
                lb = tr_lb.expand(len(train_x), len(tr_lb.ravel()))
                ub = tr_ub.expand(len(train_x), len(tr_ub.ravel()))
                clamp_x = torch.clamp(train_x, lb, ub)
                diff = (torch.abs(train_x-clamp_x)).sum(-1)
                in_center_num = len(diff[diff==0])
                break
                base_length += increase_step
                
            
            
            # tr_lb = torch.clamp(x_center - weights * state.length / 2.0, 0.0, 1.0)
            # tr_ub = torch.clamp(x_center + weights * state.length / 2.0, 0.0, 1.0)
            print('in center', in_center_num, base_length)
            tr = torch.cat((tr_lb, tr_ub)).to(dtype=dtype, device=device)
            # print(tr)
            print('inside tr ', tr[:, :10])

            
            
            old_model = {
            'gp': model,
            'likelihood': likelihood
            }
            # old_model = None
            if (not use_center_sparse_gp or not choose_best) and train_new_gp:
                model, likelihood, mll, max_weight = train_fvgp(train_loader, tr, num_epochs=num_epochs, 
                                            induce_size=model.variational_strategy.inducing_points.shape[0],  return_mll=True, 
                                            init=init
                                            )
                # if choose_best == True and base_length-increase_step > state.length:
                if choose_best == True and in_center_num == 1:

                    use_center_sparse_gp = True
            else:
                # if this case the gp use for the last loop is the same as the current loop
                print('use center sparse gp')
            # if max_weight < 1:
            #     early_stop = True
    return new_X



def focal_acqf_opt_sample(model, likelihood, state, x_center, acqf, 
                   train_loader, num_epochs=1000, max_loop_num=3, batch_size=1,
                   init=True,
                   tr = None,
                   cand_tr= None,
                   train_new_gp=True,
                   device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), 
                   dtype = torch.float
                   ):
    
    # x_center = 0.5 * torch.ones(1, model.dim).to(dtype=dtype, device=device) # center of the input region
    x_center = x_center.reshape(1, -1)
    print('start center', x_center.mean(),x_center.max(),x_center.min(), state.length)
    if tr == None:
        tr = [0, 1]
    ori_bound = tr
    if cand_tr is not None:
        cand_tr = cand_tr
    else:
        cand_tr = tr.clone()
    print('gp tr', tr[:, :10])
    print('cand tr', cand_tr[:, :10])
    
    # tr = [0, 1]
    # n_per_loop = int(np.ceil(batch_size/max_loop_num))
    # total_budget = batch_size
    # budget_per_loop = []
    # for i in range(max_loop_num):
    #     b = min(n_per_loop, total_budget)
    #     budget_per_loop.append(b)
    #     total_budget -= b
    
    # budget_per_loop.reverse()
        
    new_X = torch.zeros(0, x_center.shape[1]).to(device=device, dtype=dtype)
    new_Acq = torch.zeros(0, 1).to(device=device, dtype=dtype)
    depth_record = torch.zeros(0, 1).to(device=device, dtype=dtype)
    early_stop = False
    
    use_center_sparse_gp = False
    
    for loop in range(max_loop_num):
        # if early_stop:
        #     budget_per_loop[loop] = np.sum(budget_per_loop[loop:])
        # if acqf == 'ucb':
        #     cand_x = generate_cand(state=state, x_center=x_center, bound=cand_tr)
        #     new_x = upper_confidence_bound(model, likelihood, cand_x=cand_x, batch_size=batch)
        if 'q' in acqf or 'pi' in acqf or 'ei' in acqf:
            best_f = train_loader.dataset.tensors[1].max().item()
            # print(best_f)
            # raise NotImplementedError
            
            # weights = model.covar_module.base_kernel.lengthscale.squeeze().detach()
            # # weights = 1/weights
            # weights = weights / weights.mean()
            # weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
            weights = torch.ones(model.dim).to(device=device, dtype=dtype)
            # weights = torch.clamp(weights, 0, 1)
            tr_lb = torch.clamp(x_center - weights * state.length / 2.0, tr[0], tr[1])
            tr_ub = torch.clamp(x_center + weights * state.length / 2.0, tr[0], tr[1])
            print('inside bound', tr_lb, tr_ub)
            bound = torch.cat((tr_lb, tr_ub)).to(dtype=dtype, device=device)
            if acqf == 'ei':
                new_x, new_acq = expected_improvement(model, likelihood, best_f=best_f, bound=bound, batch_size=batch_size, return_acq=True)
            elif acqf == 'qei':
                new_x, new_acq = q_expected_improvement(model, likelihood, best_f=best_f, bound=bound, batch_size=batch_size, return_acq=True)
            elif acqf == 'qucb':
                new_x, new_acq = q_uppper_confidence_bound(model, likelihood, bound=bound, batch_size=batch_size, return_acq=True)
            if acqf == 'pi':
                print('use pi')
                new_x, new_acq = probability_of_improvement(model, likelihood, best_f=best_f, bound=bound, batch_size=batch_size, return_acq=True)
        else:
            cand_x = generate_cand(state=state, x_center=x_center, model=model, bound=cand_tr)
            new_x, new_acq = thompson_sampling(model, likelihood, cand_x=cand_x, batch_size=batch_size, return_acq=True)
        # Exit loop if converge or hit the bound
        print('loop', loop, state.length, len(new_x))
        # print('loop', loop, state.length, len(new_x), new_x)
        # print(new_x)
        # dist = torch.pdist(model.variational_strategy.inducing_points)
        # print('dist', dist.max().item(), dist.min().item())
        # if torch.norm(new_x-x_center) < 1e-3 or torch.norm(new_x.max()-1) < 1e-3 or torch.norm(new_x.min()) < 1e-3:
        #     print('early break', loop, new_x, torch.norm(new_x-x_center), torch.norm(new_x.max()-1), torch.norm(new_x.min()))
        #     break
        # print(new_x, new_acq)
        # raise NotImplementedError
        new_X = torch.cat((new_X, new_x))
        # print(new_x)
        # print(new_acq)
        # raise NotImplementedError
        new_Acq = torch.cat((new_Acq, new_acq.reshape(-1, 1)))
        depth_record = torch.cat((depth_record, torch.ones(len(new_x), 1).to(device=device, dtype=dtype)*loop))
        # if budget_per_loop[loop] > 0:
        #     new_X = torch.cat((new_X, new_x))
        if loop < max_loop_num-1:
            state.length /= 2
            # x_center = new_x[-1].reshape(1, -1)
            # choose center point with most training point in the new search region, otherwise choose current best as center
            # in_center_num = []
            train_x = train_loader.dataset.tensors[0]
            # for x in new_x:
            #     # weights = model.covar_module.base_kernel.lengthscale.squeeze().detach()
            #     # # weights = 1/weights
            #     # weights = weights / weights.mean()
            #     # weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
                
            #     weights = torch.ones(model.dim).to(device=device, dtype=dtype)
                
            #     # weights = torch.clamp(weights, 0, 1)
            #     tr_lb = torch.clamp(x - weights * state.length / 2.0, ori_bound[0], ori_bound[1])
            #     tr_ub = torch.clamp(x + weights * state.length / 2.0, ori_bound[0], ori_bound[1])
            #     # tr = torch.cat((tr_lb, tr_ub)).to(dtype=dtype, device=device)
            #     lb = tr_lb.expand(len(train_x), len(tr_lb.ravel()))
            #     ub = tr_ub.expand(len(train_x), len(tr_ub.ravel()))
            #     clamp_x = torch.clamp(train_x, lb, ub)
            #     diff = (torch.abs(train_x-clamp_x)).sum(-1)
            #     if diff.min() > 0:
            #         in_center_num.append(0)
            #     else:
            #         in_center_num.append(len(diff[diff==0]))
            # print('in center', in_center_num)
            
            
            
            # if np.max(in_center_num) < model.variational_strategy.inducing_points.shape[0]:
            x_center = train_x[train_loader.dataset.tensors[1].argmax()].reshape(1, -1)
            # print('too sparse, choose current best as center')
            choose_best = True
            # else:
            #     valid_idx = [i for i in range(len(in_center_num)) if in_center_num[i] >= model.variational_strategy.inducing_points.shape[0]]
            #     # valid_idx = [i for i in range(len(in_center_num))]
            #     center_idx = np.random.choice(valid_idx, size=1)[0]
            #     print(valid_idx, center_idx, in_center_num[center_idx])
            #     x_center = new_x[center_idx].reshape(1, -1)
            #     choose_best = False
            
            # cand_x = generate_cand(state=state, x_center=x_center)
            # def test_sample(num_pts):
            #     return generate_cand(state=state, x_center=x_center, n_candidates=num_pts)
            
            # weights = model.covar_module.base_kernel.lengthscale.squeeze().detach()
            # # weights = 1/weights
            # weights = weights / weights.mean()
            # weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
            
            weights = torch.ones(model.dim).to(device=device, dtype=dtype)
            
            # weights = torch.clamp(weights, 0, 1)
            # slowly increase tr to include enough training point
            base_length = state.length
            in_center_num = 0
            increase_step = 0.01
            # while(in_center_num < model.variational_strategy.inducing_points.shape[0]):
            tr_lb = torch.clamp(x_center - weights * base_length / 2.0, ori_bound[0], ori_bound[1])
            tr_ub = torch.clamp(x_center + weights * base_length / 2.0, ori_bound[0], ori_bound[1])
            lb = tr_lb.expand(len(train_x), len(tr_lb.ravel()))
            ub = tr_ub.expand(len(train_x), len(tr_ub.ravel()))
            clamp_x = torch.clamp(train_x, lb, ub)
            diff = (torch.abs(train_x-clamp_x)).sum(-1)
            in_center_num = len(diff[diff==0])
                # break
                # base_length += increase_step
                
            
            
            # tr_lb = torch.clamp(x_center - weights * state.length / 2.0, 0.0, 1.0)
            # tr_ub = torch.clamp(x_center + weights * state.length / 2.0, 0.0, 1.0)
            print('in center', in_center_num, base_length)
            tr = torch.cat((tr_lb, tr_ub)).to(dtype=dtype, device=device)
            # print(tr)
            print('inside tr ', tr[:, :10])

            
            
            old_model = {
            'gp': model,
            'likelihood': likelihood
            }
            # old_model = None
            if (not use_center_sparse_gp or not choose_best) and train_new_gp:
                model, likelihood, mll, max_weight = train_fvgp(train_loader, tr, num_epochs=num_epochs, 
                                            induce_size=model.variational_strategy.inducing_points.shape[0],  return_mll=True, 
                                            init=init
                                            )
                # if choose_best == True and base_length-increase_step > state.length:
                if choose_best == True and in_center_num == 1:

                    use_center_sparse_gp = True
            else:
                # if this case the gp use for the last loop is the same as the current loop
                print('use center sparse gp')
            # if max_weight < 1:
            #     early_stop = True
            

    # print(new_X.shape)
    # print(new_X, new_Acq, depth_record)
    
    # select batch size of points according to the acquisition function value
    # new_Acq = (new_Acq-new_Acq.mean())/new_Acq.std()
    base_temp = 1
    while(True):
        select_prob = torch.softmax(base_temp*new_Acq, dim=0)
        if len(select_prob[select_prob>0]) >= batch_size:
            break
        base_temp/=2
    print('sele', len(select_prob[select_prob>0]), base_temp)
    try:
        topk_idx = np.random.choice(range(len(new_X)), size=batch_size, p=select_prob.detach().cpu().numpy().ravel(), replace=False)
        topk_idx = np.sort(topk_idx)
    except:
        print(new_Acq, new_X, new_Acq.shape, new_X.shape, select_prob, select_prob.shape)
    # print(select_prob)
    # print(topk_idx)
    next_X = new_X[topk_idx]
    next_depth = depth_record[topk_idx]
    mean_acq = []
    for i in range(max_loop_num):
        mean_acq.append(new_Acq[depth_record==i].mean().item())
    print(f'Acq value of each depth: {mean_acq}')
    # print(next_depth)
    # raise NotImplementedError
    return next_X, next_depth