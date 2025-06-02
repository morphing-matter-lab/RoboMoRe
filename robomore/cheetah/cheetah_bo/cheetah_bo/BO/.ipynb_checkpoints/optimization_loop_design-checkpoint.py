import os
import time
import argparse
from copy import deepcopy
# import multiprocessing as mp
import pickle as pkl
import numpy as np
import torch
from torch.quasirandom import SobolEngine
from torch.utils.data import TensorDataset, DataLoader
# from functions.design_bench_fun import DesignBenchFunction # recommend to use another python env
import gpytorch

from botorch.test_functions import SyntheticTestFunction, Levy, Shekel, Michalewicz
from botorch.utils.transforms import unnormalize, normalize
from botorch import test_functions
# from botorch.models import ApproximateGPyTorchModel
from gp_model import train_gp, train_svgp, train_fvgp, train_spgp, train_NNGP
from gp_fun import GPFunction
from turbo_botorch import TurboState, update_state, generate_cand
from acqf import thompson_sampling, expected_improvement, q_expected_improvement, \
    upper_confidence_bound, q_uppper_confidence_bound, focal_acqf_opt, focal_acqf_opt_sample, probability_of_improvement
from mujoco_gym_env_design import MujocoDesignEnv


parser = argparse.ArgumentParser(description='BO experiment')


parser.add_argument('--model', default='fvgp', type=str, help='GP model name')
parser.add_argument('--algo', default='bo', type=str, help='BO optimizer name')
parser.add_argument('--acqf', default='ts', type=str, help='BO acquisition function name')
parser.add_argument('--induce_size', default=20, type=int, help='Induce size of sparse GP')
parser.add_argument('--max_loop_num', default=3, type=int, help='Optimization depth for Focal BO')
parser.add_argument('--use_depth', default=0, type=int, help='1 for use in-depth optimization 0 for not use')

parser.add_argument('--round', default=10, type=int, help='Round of optimization')
parser.add_argument('--init_num', default=20, type=int, help='Initial point number of each round')
parser.add_argument('--eval_num', default=300, type=int, help='Evaluation number of each round')
parser.add_argument('--batch_size', default=10, type=int, help='Batch size of each iteration')
parser.add_argument('--auto', default=0, type=int, help='whether adaptivaly adjust the depth')

parser.add_argument('--task', default='Ackley', type=str, help='Optimization task name')

args = parser.parse_args()

if __name__ == '__main__':

    # mp.set_start_method('spawn', force=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float
    torch.manual_seed(0)
    gp_model = args.model
    # gp_model = 'fvgp'
    max_loop_num = args.max_loop_num
    # gp_model = 'svgp'
    init = False

    # algo = 'turbo'
    algo = args.algo

    acqf = args.acqf

    # Problem settings
    n_initial_samples = args.init_num # number of initial random samples
    # additional_evals = 290  # the number of evaluations after the initial random samples
    # n_samples_per_step = 10  # number of new samples per step
    # induce_size=20
    n_evals = args.eval_num  # number of optimization steps
    n_repeat = args.round
    n_samples_per_step = args.batch_size  # number of new samples per step
    induce_size = args.induce_size
    # additional_evals = 19000  # the number of evaluations after the initial random samples
    # n_samples_per_step = 200  # number of new samples per step
    # induce_size=1000
    print(args.task)
    
    # Define the objective function (the Branin function)
    if args.task == 'GP':
        dim=5
        objective = GPFunction(dim=dim, ls=0.5)
        bounds = torch.cat((0*torch.ones(1, dim), 1*torch.ones(1, dim))).to(dtype=dtype, device=device)
    elif args.task == 'Ant':
        objective = MujocoDesignEnv('Ant-v2', 10, minimize=False)
        dim = objective.dim
        bounds = torch.cat((-1*torch.ones(1, dim), 1*torch.ones(1, dim))).to(dtype=dtype, device=device) # bounds of the problem
    elif args.task == 'Humanoid':
        objective = MujocoDesignEnv('Humanoid-v2', 10, minimize=False)
        dim = objective.dim
        bounds = torch.cat((-1*torch.ones(1, dim), 1*torch.ones(1, dim))).to(dtype=dtype, device=device) # bounds of the problem
    elif args.task == 'Walker2d':
        objective = MujocoDesignEnv('Walker2d-v2', 10, minimize=False)
        dim = objective.dim
        bounds = torch.cat((-1*torch.ones(1, dim), 1*torch.ones(1, dim))).to(dtype=dtype, device=device) # bounds of the problem
    elif args.task == 'HalfCheetah':
        objective = MujocoDesignEnv('HalfCheetah-v2', 10, minimize=False)
        dim = objective.dim
        bounds = torch.cat((-1*torch.ones(1, dim), 1*torch.ones(1, dim))).to(dtype=dtype, device=device) # bounds of the problem
    elif args.task == 'Hopper':
        objective = MujocoDesignEnv('Hopper-v2', 10, minimize=False)
        dim = objective.dim
        bounds = torch.cat((-1*torch.ones(1, dim), 1*torch.ones(1, dim))).to(dtype=dtype, device=device) # bounds of the problem
    

    elif args.task == 'CheetahDesign':
        objective = MujocoDesignEnv('GPTCheetahEnv', 10, minimize=False)
        dim = objective.dim
        bounds = torch.cat((torch.from_numpy(objective.lb).reshape(1, -1), 
                        torch.from_numpy(objective.ub).reshape(1, -1)), 0).to(dtype=dtype, device=device)
    
    elif 'Morphology' in args.task:
        objective = DesignBenchFunction(args.task)
        dim = objective.dim
        bounds = torch.cat((0*torch.ones(1, dim), 1*torch.ones(1, dim))).to(dtype=dtype, device=device) # bounds of the problem
    
        print(bounds)

    
    def eval_objective(x):
        print(x.shape)
        unnorm_x = unnormalize(x, bounds)
        # if isinstance(objective, mujoco_gym_env.MujocoGymEnv) or isinstance(objective, lunar_land.Lunar):

        ndim = len(x.shape)
        if ndim == 3:
            y = torch.zeros(x.shape[0], x.shape[1], 1).to(dtype=dtype, device=device)
            for i in range(x.shape[0]):
                for j in range(x.shape[1]):
                    y[i, j]= objective(unnorm_x[i, j])
                    print(i, j, y[i, j], end='\r')
        else:
            y = torch.zeros(x.shape[0], 1).to(dtype=dtype, device=device)
            for i in range(x.shape[0]):
                    y[i]= objective(unnorm_x[i])
                    print(f"y: {y[i]}")
        
            
        
        return y
    
    cur_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    if not isinstance(objective, SyntheticTestFunction):
        obj_name = objective.__name__
        save_path = f'./results/{cur_time}_{objective.__name__}_{algo}_{acqf}_{dim}_{n_evals}_{n_initial_samples}_{n_samples_per_step}_{induce_size}_{gp_model}_{args.use_depth}'
    else:
        obj_name = objective.__class__.__name__
        save_path = f'./results/{cur_time}_{objective.__class__.__name__}_{algo}_{acqf}_{dim}_{n_evals}_{n_initial_samples}_{n_samples_per_step}_{induce_size}_{gp_model}_{args.use_depth}'
    # if gp_model == 'fvgp':
    save_path += f'_{max_loop_num}_{init}'
    # if gp_model == 'fvgp':
    auto_suffix = '_auto' if args.auto == 1 else ''
    save_path += auto_suffix
    # n_repeat = 10
    num_epochs = 100
    gen_ini = True
    # if isinstance(objective, mujoco_gym_env.MujocoGymEnv) or isinstance(objective, lunar_land.Lunar):
    
    # Replace the hard-coded init_X with random sampling between bounds
    # Create a random initial point within the bounds of the design space
    lb = bounds[0].cpu().numpy()  # Lower bounds
    ub = bounds[1].cpu().numpy()  # Upper bounds
    
    # Sample randomly between lb and ub
    init_X_numpy = lb + (ub - lb) * np.random.rand(dim)
    init_X = torch.tensor(init_X_numpy, dtype=dtype, device=device)
    
    # The rest of the code already normalizes this, so we keep the values in the original space
    initial_X = init_X.reshape(1, -1)
    initial_Y = eval_objective(normalize(initial_X, bounds))
    initial_Y = initial_Y.reshape(-1, 1)

    timesteps = np.zeros(0)
    t0 = time.time()


    with gpytorch.settings.max_cholesky_size(float('inf')):
        for rep in range(n_repeat):
            model = likelihood = None
            max_loop_num = args.max_loop_num
            if not isinstance(objective, SyntheticTestFunction):
                train_X = initial_X[:n_initial_samples].clone()
                train_Y = initial_Y[:n_initial_samples].clone()
                if not os.path.exists(save_path):
                    os.mkdir(save_path)

                # clear cache when using GP function
                if isinstance(objective, GPFunction):
                    del objective
                    objective = GPFunction(dim=dim, ls=0.2)
                    objective.eval_x = train_X
                    objective.eval_y = train_Y.ravel()
            else:
                train_X = initial_X[rep, :n_initial_samples].clone()
                train_Y = initial_Y[rep, :n_initial_samples].clone().reshape(-1, 1)
                # print(train_Y.shape)
                # raise NotImplementedError
            best_observed_value_index = train_Y.argmax()

            if algo == 'turbo':
                state = TurboState(dim, batch_size=n_samples_per_step)
                try:
                    print(f"{obj_name} {rep} Init {len(train_X)} {state.length} Best regret: {objective.optimal_value - train_Y[best_observed_value_index]}")
                except:
                    print(f"{obj_name} {rep} Init {len(train_X)} {state.length} Best reward: {train_Y[best_observed_value_index]}")

            else:
                state = TurboState(dim, batch_size=n_samples_per_step, length=1)
                # state = TurboState(dim, batch_size=n_samples_per_step, length=1)
                try:
                    print(f"{obj_name} {rep} Init {len(train_X)} Best regret: {objective.optimal_value - train_Y[best_observed_value_index]}")
                except:
                    print(f"{obj_name} {rep} Init {len(train_X)} Best reward: {train_Y[best_observed_value_index]}")
            all_loop_nums = []
            all_max_depth = []
            all_depth_record = torch.zeros(0, 1).to(dtype=dtype, device=device)
            while len(train_X) < n_evals:
                if len(train_X) > 1:
                    norm_Y = (train_Y-train_Y.mean())/train_Y.std()
                else:
                    norm_Y = train_Y
                norm_Y = norm_Y.ravel()
                train_dataset = TensorDataset(train_X, norm_Y)
                # train_dataset = TensorDataset(train_X, train_Y)
                train_loader = DataLoader(train_dataset, batch_size=30000, shuffle=True)
                if algo == 'turbo':
                    # cand_x, state = generate_cand(state, train_X, train_Y)
                    # cand_x = generate_cand(state=state, X=train_X, Y=norm_Y)
                    # cand_x = generate_cand(state=state, X=train_X, Y=norm_Y)
                    x_center = train_X[norm_Y.argmax()]

                    # def test_sample(num_pts):
                    #     return generate_cand(state=state, X=train_X, Y=norm_Y, n_candidates=num_pts)
                    # def test_sample(num_pts):
                    #     return generate_cand(state=state, X=train_X, Y=norm_Y, n_candidates=num_pts)
                else:
                    # sobol = SobolEngine(dim.shape[1], scramble=True)
                    # cand_x = sobol.draw(5000).to(dtype=dtype, device=device)
                    # cand_x = generate_cand(dim=train_X.shape[-1])
                    # def test_sample(num_pts):
                    #     return generate_cand(dim=train_X.shape[-1], n_candidates=num_pts)
                    # cand_x = generate_cand(dim=train_X.shape[-1])
                    # def test_sample(num_pts):
                    #     return generate_cand(dim=train_X.shape[-1], n_candidates=num_pts)
                    x_center = 0.5 * torch.ones(1, dim).to(dtype=dtype, device=device) # center of the input region
                    
                if gp_model == 'gp' or len(train_X) < induce_size:
                    print("train_X", train_X.shape, train_X)
                    print("norm_Y", norm_Y.shape, norm_Y)
                    model, likelihood = train_gp(train_X, norm_Y, num_epochs=num_epochs)
                    tr = cand_tr =  torch.cat((torch.zeros(dim).reshape(1, -1), torch.ones(dim).reshape(1, -1))).to(dtype=dtype, device=device)

                    # model = train_gp(train_X, train_Y, num_epochs=num_epochs)
                else:
                    if gp_model == 'fvgp':
                        try:
                            base_length = state.length
                            in_center_num = 0
                            increase_step = 0.01
                            if algo == 'turbo':
                                # try:
                                #     weights = model.covar_module.base_kernel.lengthscale.squeeze().detach()
                                #     weights = weights / weights.mean()
                                #     weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
                                # except:
                                weights = torch.ones(dim).to(dtype=dtype, device=device)
                                print('origin tr weights', weights.mean(), weights.std(), weights.max(), weights.min())
                                tr_lb = torch.clamp(x_center - weights*base_length / 2.0, 0.0, 1.0)
                                tr_ub = torch.clamp(x_center + weights*base_length / 2.0, 0.0, 1.0)
                                tr = cand_tr = torch.cat((tr_lb.reshape(1, -1), tr_ub.reshape(1, -1))).to(dtype=dtype, device=device)
                                lb = tr_lb.expand(len(train_X), len(tr_lb.ravel()))
                                ub = tr_ub.expand(len(train_X), len(tr_ub.ravel()))
                                clamp_x = torch.clamp(train_X, lb, ub)
                                diff = (torch.abs(train_X-clamp_x)).sum(-1)
                                in_center_num = len(diff[diff==0])
                                # while(in_center_num < induce_size):
                                #     tr_lb = torch.clamp(x_center - weights*base_length / 2.0, 0.0, 1.0)
                                #     tr_ub = torch.clamp(x_center + weights*base_length / 2.0, 0.0, 1.0)
                                #     lb = tr_lb.expand(len(train_X), len(tr_lb.ravel()))
                                #     ub = tr_ub.expand(len(train_X), len(tr_ub.ravel()))
                                #     clamp_x = torch.clamp(train_X, lb, ub)
                                #     diff = (torch.abs(train_X-clamp_x)).sum(-1)
                                #     in_center_num = len(diff[diff==0])
                                #     base_length += increase_step
                                
                                print('in center', in_center_num, base_length)
                                tr = torch.cat((tr_lb.reshape(1, -1), tr_ub.reshape(1, -1))).to(dtype=dtype, device=device)
                            else:
                                tr =  torch.cat((torch.zeros(dim).reshape(1, -1), torch.ones(dim).reshape(1, -1))).to(dtype=dtype, device=device)
                                cand_tr = tr
                            # print(tr)
                        except Exception as e:
                            print('error', e)
                            # tr = torch.cat((torch.zeros(1, dim), torch.ones(1, dim))).to(dtype=dtype, device=device)
                            raise NotImplementedError
                       
                        model, likelihood = train_fvgp(train_loader, tr, num_epochs=num_epochs, induce_size=induce_size, 
                                                       init=init
                                                       )
                    elif gp_model == 'svgp':
                        model, likelihood = train_svgp(train_loader, num_epochs=num_epochs, induce_size=induce_size)
                        tr = cand_tr =  torch.cat((torch.zeros(dim).reshape(1, -1), torch.ones(dim).reshape(1, -1))).to(dtype=dtype, device=device)
                    elif gp_model == 'spgp':
                        # model = likelihood = None
                        model, likelihood = train_spgp(train_X, train_Y, num_epochs, dim, induce_size, old_model=model)
                        tr = cand_tr =  torch.cat((torch.zeros(dim).reshape(1, -1), torch.ones(dim).reshape(1, -1))).to(dtype=dtype, device=device)
                    elif gp_model == 'vecchia':
                        model, likelihood = train_vecchia(train_X, train_Y, dim, num_epochs)
                        tr = cand_tr =  torch.cat((torch.zeros(dim).reshape(1, -1), torch.ones(dim).reshape(1, -1))).to(dtype=dtype, device=device)
                    elif gp_model == 'nngp':
                        model, likelihood = train_NNGP(inducing_points=None, induce_size=induce_size, train_loader=train_loader, k=32, training_batch_size=64, epochs=num_epochs)
                        tr = cand_tr =  torch.cat((torch.zeros(dim).reshape(1, -1), torch.ones(dim).reshape(1, -1))).to(dtype=dtype, device=device)
                    else: # GP
                        model, likelihood = train_gp(train_X, norm_Y, num_epochs=num_epochs)
                        tr = cand_tr =  torch.cat((torch.zeros(dim).reshape(1, -1), torch.ones(dim).reshape(1, -1))).to(dtype=dtype, device=device)
                # Define the acquisition function

                if gp_model == 'fvgp' or args.use_depth == 1 and len(train_X) >= induce_size: # sparse to dense optimization for fvgp and vanilla BO

                    train_new_gp = True if gp_model == 'fvgp' else False
                    
                    # new_X = focal_acqf_opt(model, likelihood, deepcopy(state), x_center, acqf, 
                    #                         train_loader, num_epochs=num_epochs, 
                    #                         max_loop_num=max_loop_num, batch_size=n_samples_per_step, init=init, tr=tr, cand_tr=cand_tr, train_new_gp=train_new_gp)
                    new_X, new_depth = focal_acqf_opt_sample(model, likelihood, deepcopy(state), x_center, acqf, 
                                            train_loader, num_epochs=num_epochs, 
                                            max_loop_num=max_loop_num, batch_size=n_samples_per_step, init=init, tr=tr, cand_tr=cand_tr, train_new_gp=train_new_gp)
                    # print('x next', new_X)
                    depth_num = []
                    for l_idx in range(max_loop_num):
                        depth_num.append((new_depth == l_idx).sum().item())
                    print(f'Batch number of each depth {depth_num}')
                    all_depth_record = torch.cat((all_depth_record, new_depth), 0)
                    
                else:
                    if gp_model=="vecchia":
                        train_X, train_Y = train_X.to("cpu"), train_Y.to("cpu")
                        norm_Y, device   = norm_Y.to(device="cpu"), "cpu"
                    if acqf == 'ucb':
                        cand_x = generate_cand(dim=train_X.shape[-1])
                        cand_x = generate_cand(dim=train_X.shape[-1])
                        new_X = upper_confidence_bound(model, 
                                                        likelihood, 
                                                        cand_x=cand_x, batch_size=n_samples_per_step)
                    elif acqf == 'ei':
                        new_X = expected_improvement(model, likelihood, best_f=norm_Y.max(), batch_size=n_samples_per_step, device=device)
                    elif acqf == 'qei':
                        print('use qei')
                        new_X = q_expected_improvement(model, likelihood, best_f=norm_Y.max(), batch_size=n_samples_per_step)
                    elif acqf == 'qucb':
                        print('use qucb')
                        new_X = q_uppper_confidence_bound(model, likelihood, batch_size=n_samples_per_step)
                    elif acqf == 'pi':
                        print('use pi')
                        new_X = probability_of_improvement(model, likelihood, best_f=norm_Y.max(), batch_size=n_samples_per_step)
                    else:
                        print('use ts')
                        if algo == 'turbo':
                            cand_x = generate_cand(state=state, X=train_X, Y=norm_Y, model=model)
                        else:
                            cand_x = generate_cand(dim=train_X.shape[-1])
                        if gp_model=="vecchia":
                            cand_x = cand_x.to(device="cpu")
                        new_X = thompson_sampling(model, likelihood, cand_x=cand_x, batch_size=n_samples_per_step)
                
                # Update the training data, make sure to convert back to cuda bc of vecchia
                # Evaluate the objective function at the new samples
                device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                train_X, train_Y, norm_Y = train_X.to(device=device), train_Y.to(device=device), norm_Y.to(device=device), 
                
                new_Y = eval_objective(new_X.to(device=device))
                new_Y = new_Y.reshape(-1, 1)
                timesteps = np.hstack((timesteps, time.time()-t0))
                train_X = torch.cat([train_X, new_X.to(device=device)])
                train_Y = torch.cat([train_Y, new_Y.to(device=device)])
                
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                
                if algo == 'turbo':
                    state = update_state(state, new_Y)
                
                # Print the location of the best observed value
                best_observed_value_index = train_Y.argmax()
                try:
                    print(f'{obj_name} {gp_model} {algo} {rep} Eval {len(train_X)}'\
                        f'Best this round: {objective.optimal_value - new_Y.max()} '\
                        f'Best regret: {objective.optimal_value - train_Y[best_observed_value_index]}')\
                    
                except:
                    print(f'{obj_name} {gp_model} {algo} {rep} Eval {len(train_X)}'\
                        f'Best this round: {new_Y.max()} '\
                        f'Best reward: {train_Y[best_observed_value_index]}')
                if gp_model == 'fvgp' or args.use_depth ==1 and len(train_X) >= induce_size:
                    budget_per_loop = []
                    total_budget = n_samples_per_step
                    n_per_loop = int(np.ceil(n_samples_per_step/max_loop_num))
                    val_per_depth = []
                    
                    for i in range(max_loop_num):
                        b = min(n_per_loop, total_budget)
                        budget_per_loop.append(b)
                        total_budget -= b
                    
                    budget_per_loop.reverse()
                    start = 0
                    for b in budget_per_loop:
                        val_per_depth.append(new_Y[start:start+b].mean())
                        start += b
                    
                    # check the best depth
                    max_idx = new_Y.argmax()
                    
                        
                    depth = new_depth[max_idx].item()  
                    # for depth in range(max_loop_num):
                    #     if max_idx < np.sum(budget_per_loop[:depth+1]):
                    #         break
                    # if new_Y.max() == train_Y.max():
                    if args.auto == 1:
                        if depth + 1 < max_loop_num:
                            max_loop_num -= 1
                        else:
                            # max_loop_num = min(dim, max_loop_num+1)
                            max_loop_num += 1
                        # max_loop_num = depth + 1
                    print(f'Max point find in depth {depth+1}, set max loop num to {max_loop_num}')
                    # else:
                    #     max_loop_num += 1
                    #     print(f'Not find max {depth+1} {val_per_depth}, increase max loop num to {max_loop_num}')
                    
                        
                    all_loop_nums.append(max_loop_num)
                    all_max_depth.append(depth+1)
                    
                    # print(f'Max point find in depth {depth+1} {val_per_depth}, set max loop num to {max_loop_num}')
                    # else:
                    #     print(f'Max point find in depth {depth+1} {val_per_depth}, current max loop num to {max_loop_num}')
                    
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                if dim < 100:
                    np.savez(f'./{save_path}/{rep}.npz', x=train_X.detach().cpu().numpy(), 
                             y=train_Y.detach().cpu().numpy(), loop_num=np.array(all_loop_nums), 
                             max_depth=np.array(all_max_depth), depth_record=all_depth_record.detach().cpu().numpy(), timesteps=timesteps)
                else:
                    best_x = train_X[best_observed_value_index].detach().cpu().numpy()
                    np.savez(f'./{save_path}/{rep}.npz', y=train_Y.detach().cpu().numpy(), 
                             best_x=best_x, loop_num=np.array(all_loop_nums), 
                             max_depth=np.array(all_max_depth), depth_record=all_depth_record.detach().cpu().numpy())
