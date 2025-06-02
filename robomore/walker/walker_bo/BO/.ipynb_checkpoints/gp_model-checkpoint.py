import time
import numpy as np
import torch
from torch.quasirandom import SobolEngine
import gpytorch
from gpytorch.models import ExactGP, ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
# from test import LogVariationalStrategy as VariationalStrategy
from gpytorch.variational import VariationalStrategy
from gpytorch.variational.nearest_neighbor_variational_strategy import NNVariationalStrategy
from gpytorch.mlls import VariationalELBO
from gpytorch.constraints import Interval
from gpytorch.kernels import ScaleKernel, MaternKernel, InducingPointKernel
from gpytorch.distributions import MultivariateNormal
from gpytorch.means import ZeroMean
from copy import deepcopy
from botorch.models.gpytorch import GPyTorchModel

from focalized_elbo import FocalizedVariationalELBO

from SPGP_utils import pack_hyps, unpack_hyps, SPGP_likelihood, OGP

# import _pyvecch 
# import pyvecch
# from pyvecch.nbrs import ExactOracle
# from pyvecch.models import RFVecchia
# from pyvecch.prediction import IndependentRF
# from pyvecch.training import fit_model
# from pyvecch.input_transforms import Identity

from botorch.optim.stopping import ExpMAStoppingCriterion
from gpytorch.mlls import ExactMarginalLogLikelihood

class BoTorchWrapper(GPyTorchModel):
    def __init__(
        self,
        model, 
        likelihood,
        num_output=1,
        *args,
        **kwargs,
    ):
        r"""
        Botorch wrapper class for various GP models in gpytorch. 
        model: instance of gpytorch. GP models
        likelihood: instance of gpytorch likelihood
        num_outputs: number of outputs expected for the GP model
        """

        super().__init__()
        # Def num_outputs, model below
        # raise NotImplementedError
        self.model = model
        self.likelihood = likelihood
        self._desired_num_outputs = num_output
        

    @property
    def num_outputs(self):
        return self._desired_num_outputs
    
    
    def forward(self, X, *args, **kwargs) -> MultivariateNormal:
        X = self.transform_inputs(X)
        # if type(self.model)==pyvecch.models.rf_vecchia.RFVecchia:
        #     X = X.to(device='cpu')
        return self.model(X)
    

class GP(ExactGP):
    def __init__(self, train_x, train_y, likelihood, dim=None, kernel='Matern'):
        super(GP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()


        lengthscale_constraint = Interval(0.005, 10.0)  # [0.005, sqrt(dim)]

        
        if train_x is None:
            assert dim is not None
            self.covar_module =gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=dim, lengthscale_constraint=lengthscale_constraint))
            self.dim = dim
        else:
            self.covar_module =gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=train_x.shape[1], lengthscale_constraint=lengthscale_constraint)) 
            self.dim = train_x.shape[1]
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class SVGP(ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        super(SVGP, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = ScaleKernel(
                    MaternKernel(nu=2.5, ard_num_dims=inducing_points.shape[1], 
                        lengthscale_constraint=Interval(0.005, inducing_points.shape[1]))
                    )
        self.dim = inducing_points.shape[-1]
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    

def train_gp(train_x=None, train_y=None,
             num_epochs=1000, 
             dtype=torch.float, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    # initialize likelihood and model
    # likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=Interval(0, 1e-3)).to(dtype=dtype, device=device)
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(dtype=dtype, device=device)

        # likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=Interval(0, 1e-3)).to(dtype=dtype, device=device)

    model = GP(train_x, train_y.ravel(), likelihood).to(dtype=dtype, device=device)
    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(num_epochs):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(train_x)
        # Calc loss and backprop gradients
        loss = -mll(output, train_y.ravel())
        loss.backward()
        ls = model.covar_module.base_kernel.lengthscale
        noise = model.likelihood.noise.item()
        # print(f'Iter {i+1}/{num_epochs} - Loss: {loss:.4f} {ls:.4f} {noise:.4f}', end='\r')
        print(i, loss, round(ls.mean().item(), 5), round(ls.std().item(), 5), round(ls.max().item(), 5), round(ls.min().item(), 5), round(noise, 5), end='\r')

        optimizer.step()
        
    print('')
    return model.eval(), likelihood.eval()

def train_svgp(
            train_loader,
             induce_size=10, inducing_points=None, num_epochs=1000, 
             dtype=torch.float, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    # Define the approximate GP model
    if inducing_points is None:
        # inducing_points = draw_sobol_samples(bounds=bounds, n=1, q=induce_size).squeeze(0)
        sobol = SobolEngine(train_loader.dataset.tensors[0].size(1), scramble=True)
        inducing_points = sobol.draw(induce_size).to(dtype=dtype, device=device)
        print("inducing_points: ", inducing_points)
    else:
        assert len(inducing_points) == induce_size
    model = SVGP(inducing_points=inducing_points).to(dtype=dtype, device=device)
    # likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=Interval(0, 1e-3)).to(dtype=dtype, device=device)
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(dtype=dtype, device=device)

    model.train()
    likelihood.train()
    lr = 0.01
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': likelihood.parameters()},
    ], lr=lr)
    # Our loss object. We're using the VariationalELBO
    # print(len(train_loader.dataset))
    # raise NotImplementedError
    mll = VariationalELBO(likelihood, model, num_data=len(train_loader.dataset))
    for i in range(num_epochs):
    # Within each iteration, we will go over each minibatch of data
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(x_batch)
            # print(y_batch.shape, output.mean.shape)
            loss = -mll(output, y_batch.ravel())
            # print(loss)
            # raise NotImplementedError
            # mse_loss = (mse_loss).sum()/len(train_loader.dataset)
            ls = model.covar_module.base_kernel.lengthscale
            noise = likelihood.noise.item()
            # print(i, loss, ls, noise, end='\r')
            print(i, loss, round(ls.mean().item(), 5), round(ls.std().item(), 5), round(ls.max().item(), 5), round(ls.min().item(), 5), round(noise, 5), end='\r')
        
            loss.backward()
            optimizer.step()
    print('')
    return model.eval(), likelihood.eval()

def train_fvgp(
            train_loader,
            test_distribution,
            # state=None,
             induce_size=10, 
             inducing_points=None, 
             noise_max=None,
             num_epochs=1000, 
             weight=True,
             init=True,
             reg=True,
             deno=True,
             return_mll=False,
             return_loss=False,
             dtype=torch.float, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    # Define the approximate GP model
    train_x = train_loader.dataset.tensors[0]

    if isinstance(test_distribution, torch.Tensor):
        lb = test_distribution[0]
        ub = test_distribution[1]
        
        lb_all = lb.expand(len(train_x), lb.shape[0])
        ub_all = ub.expand(len(train_x), ub.shape[0])
        nearest_x = train_x.clone()
        nearest_x[train_x<lb_all] = lb_all[train_x<lb_all]
        nearest_x[train_x>ub_all] = ub_all[train_x>ub_all]
        
        diff = torch.abs(train_x - nearest_x).sum(-1)
        print('in region points', len(diff[diff==0]))
        num_in_region = len(diff[diff==0])
    
    if inducing_points is None:
        if isinstance(test_distribution, torch.Tensor):
            sobol = SobolEngine(train_loader.dataset.tensors[0].shape[1], scramble=True)
            pert = sobol.draw(induce_size).to(dtype=dtype, device=device)
            
            # find nearest neighbor of center if the in region point is less than inducing points
            if init:
                # if len(diff[diff==0]) < induce_size:
                #     x_center = (lb + ub)/2
                    
                #     center_dis = torch.cdist(x_center.reshape(1, -1), train_x[diff>0])
                #     nn = center_dis.topk(k = induce_size-len(diff[diff==0]), largest=False).indices.ravel()
                #     # print(nn.shape)
                    
                #     nn_x = torch.cat((train_x[diff==0], train_x[nn]))
                #     print(nn_x.shape)
                #     nn_lb = nn_x.min(0).values
                #     nn_ub = nn_x.max(0).values
                    
                #     comb_lb = torch.cat((lb.reshape(1, -1), nn_lb.reshape(1, -1)))
                #     comb_ub = torch.cat((ub.reshape(1, -1), nn_ub.reshape(1, -1)))
                    
                #     induce_lb = comb_lb.min(0).values
                #     induce_ub = comb_ub.max(0).values
                # else:
                induce_lb = lb
                induce_ub = ub
            else:
                induce_lb = torch.zeros_like(lb)
                induce_ub = torch.ones_like(ub)
            
                
            
            # print(comb_lb, induce_lb)
            
            # print(comb_ub, induce_ub)
            
            
            
            # inducing_points = lb + (ub - lb) * pert
            print('induce bound', induce_lb[:10], induce_ub[:10])
            print(lb[:10], ub[:10])
            # raise NotImplementedError
            inducing_points = induce_lb + (induce_ub - induce_lb) * pert
            sobol = SobolEngine(train_loader.dataset.tensors[0].size(1), scramble=True)
            inducing_points = sobol.draw(induce_size).to(dtype=dtype, device=device) 
            print(inducing_points.max(), inducing_points.min())
            # raise NotImplementedError
        else:
            inducing_points = test_distribution(induce_size)
    else:
        assert len(inducing_points) == induce_size
        
    model = SVGP(inducing_points=inducing_points)
    model = model.to(dtype=dtype, device=device)
    
    
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(dtype=dtype, device=device)
    
    if noise_max is not None:
        print('resitrict noise', noise_max)
        likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=Interval(0, noise_max)).to(dtype=dtype, device=device)
    # Give ls and noise hyperparameter for old model
    # if old_model is not None:
    #     model.covar_module.base_kernel.lengthscale = torch.clamp(old_model['gp'].covar_module.base_kernel.lengthscale.clone(), 0.005, ls_max-1e-5)
    #     likelihood.noise = old_model['likelihood'].noise.clone()
    

    model.train()
    likelihood.train()
    lr = 0.01
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': likelihood.parameters()},
    ], lr=lr)
    # Our loss object. We're using the VariationalELBO
    if weight:
        mll = FocalizedVariationalELBO(likelihood, model, num_data=len(train_loader.dataset))
        mll2 = VariationalELBO(likelihood, model, num_data=len(train_loader.dataset))
    else:
        mll = VariationalELBO(likelihood, model, num_data=len(train_loader.dataset))
        
    mse = torch.nn.MSELoss(reduction='none')
    last_loss = float('inf')
    converge_count = 0
    epoch_loss = []

    for i in range(num_epochs):
    # Within each iteration, we will go over each minibatch of data
        all_loss = []
        max_weight = 1
        for x_batch, y_batch in (train_loader):
            # Get nearest point from each training data
            # t1 = time.time()
            # 
            
            
            
            if isinstance(test_distribution, torch.Tensor):
                lb = test_distribution[0]
                lb = lb.expand(len(x_batch), lb.shape[0])
                ub = test_distribution[1]
                ub = ub.expand(len(x_batch), ub.shape[0])
                nearest_x = x_batch.clone()
                nearest_x[x_batch<lb] = lb[x_batch<lb]
                nearest_x[x_batch>ub] = ub[x_batch>ub]
                
                # if num_in_region < induce_size:
                #     min_dist, min_idx  = torch.cdist(x_batch, nn_x).min(0)
                #     # print(min_dist, min_idx)
                #     nearest_x[min_idx] = x_batch[min_idx]
                #     # print(x_batch.shape, nn_x.shape, dist.shape)
                #     # raise NotImplementedError
                # diff = torch.abs(x_batch - nearest_x).sum(-1)
                # print('train in region points', len(diff[diff==0]))
            
            
            optimizer.zero_grad()
            try:
                output = model(x_batch)
            except:
                print('')
                dis = torch.pdist(model.variational_strategy.inducing_points)
                print(dis.min(), dis.max())
                raise NotImplementedError
            # print(y_batch.shape, output.mean.shape)
            if weight:
                if isinstance(test_distribution, torch.Tensor):
                    mll_loss = -mll(output, y_batch.ravel(), x_batch, nearest_x)
                    mll_loss2 = -mll2(output, y_batch.ravel())
                else:
                    mll_loss = -mll(output, y_batch.ravel(), x_batch, test_distribution)
                # print(mll_loss, mll_loss2)
                
                mse_loss = mse(output.mean, y_batch)
                # print(mse_loss)
                # print(mll_loss, mse_loss.shape, mll.weight.shape)
                
                mse_loss = (mse_loss * mll.weight).sum()/len(train_loader.dataset)
                # loss = mll_loss + 100*mse_loss
                # weight_loss = torch.log(mll.weight.sum()/mll.num_in_region)
                # loss = mll_loss + weight_loss*(min(1, mll.num_in_region/induce_size))
                # weight_loss = torch.log(mll.weight.sum()/mll.num_in_region)
                if deno:
                    weight_loss = (mll.weight).sum() / num_in_region - 1
                else:
                    weight_loss = (mll.weight).sum()
                if reg:
                    # loss = mll_loss + weight_loss*(min(1, num_in_region/induce_size))
                    loss = mll_loss + weight_loss
                else:
                    loss = mll_loss
                # loss = mll_loss
                
                ls = model.covar_module.base_kernel.lengthscale
                noise = likelihood.noise.item()

                max_weight = min(mll.weight.max().item(), max_weight)
                # print(i, loss, ls, noise)
                print(i, mll.num_in_region, loss.item(), mll_loss.item(), mll_loss2.item(), mse_loss.item(), weight_loss.item(),round(ls.max().item(), 5), round(ls.min().item(), 5), round(noise, 5), converge_count, round(mll.weight.max().item(), 5), round(mll.weight.min().item(), 5),  end='\r')
            else:
                loss = -mll(output, y_batch.ravel())
                ls = model.covar_module.base_kernel.lengthscale
                noise = likelihood.noise.item()
                print(i, loss.item(),round(ls.max().item(), 5), round(ls.min().item(), 5), round(noise, 5), converge_count,  end='\r')
            all_loss.append(loss)
            loss.backward()
            optimizer.step()
        
        mean_loss = torch.stack(all_loss).mean().item()
        epoch_loss.append(mean_loss)
        last_10_loss = np.mean(epoch_loss[-5:])
        
        # print(i)
        if i > 200:
            # print(len(train_loader.dataset))
            # raise NotImplementedError
            if abs(last_10_loss-last_loss) < 1e-4:
                # print(i)
                converge_count += 1
            else:
                converge_count = 0
        # if converge_count >= 10:
        #     print('')
        #     print(f'early converge at {i}')
        #     break
        last_loss = last_10_loss
    print('')
    print('model ls', ls.mean().item(), ls.std().item(), ls.max().item(), ls.min().item())
    
    if return_mll: # for testing
        return model.eval(), likelihood.eval(), mll, max_weight
    elif return_loss:
        return model.eval(), likelihood.eval(), epoch_loss
    else:
        return model.eval(), likelihood.eval()

class SPGP(gpytorch.models.ExactGP):
    def __init__(self, trainX, trainY, dim, likelihood, inducing_size, old_model=None):
        super(SPGP, self).__init__(trainX, trainY, likelihood)
        self.train_X = trainX
        self.train_Y = trainY
        self.inducing_size = inducing_size
        self.mean_module = gpytorch.means.ConstantMean()
        self.base_covar_module = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=trainX.shape[1], 
                        lengthscale_constraint=Interval(0.005, trainX.shape[1])))
        # self.old_model = old_model
        # if old_model is None:
        self.covar_module = InducingPointKernel(self.base_covar_module, 
                                            inducing_points=trainX[:inducing_size, :].clone(), 
                                            likelihood=likelihood
                                            )
        # else:
            
        #     self.covar_module = InducingPointKernel(self.base_covar_module, 
        #                                         inducing_points=torch.from_numpy(self.old_model.ogp.BV).to(dtype=self.train_X.dtype, device=self.train_X.device),
        #                                         likelihood=likelihood
        #                                         )
            
        #     self.covar_module.base_kernel.base_kernel.lengthscale = self.old_model.covar_module.base_kernel.base_kernel.lengthscale.clone()
        #     # self.covar_module.inducing_points = 
        #     self.likelihood.noise = self.old_model.likelihood.noise.clone()
            
        self.dim = dim
    def fit_ogp(self, start = 0):
        # if self.old_model is not None:
        #     print('reuse old model')
        #     self.ogp = self.old_model.ogp
        #     new_x = self.train_X.clone().detach().cpu().numpy()[len(self.old_model.train_X):]
        #     new_y = self.train_Y.clone().detach().cpu().numpy().ravel()[len(self.old_model.train_X):]
        #     # print(new_x.shape, new_y.shape)
        #     self.ogp.fit(new_x, new_y)
        # else:
        self.ogp = OGP(dim=self.dim, noise=self.likelihood.noise.item(), covar=self.covar_module, maxBV=self.inducing_size, weighted=True)
        self.ogp.fit(self.train_X.detach().cpu().numpy()[start:], self.train_Y.detach().cpu().numpy().ravel()[start:])
            
        self.covar_module = InducingPointKernel(deepcopy(self.base_covar_module), 
                                                inducing_points=torch.from_numpy(self.ogp.BV).to(dtype=self.train_X.dtype, device=self.train_X.device), 
                                                likelihood=deepcopy(self.likelihood)
                                                )
        # print(self.ogp.BV.shape)
        # raise NotImplementedError
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        

# Function to train sparse gaussian process
def train_spgp(trainX, trainY, epochs, dim, induce_size, old_model=None, dtype=torch.float, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(dtype=dtype, device=device)
    if old_model is None:
        model = SPGP(trainX, trainY.ravel(), dim, likelihood, induce_size, old_model=old_model).to(dtype=dtype, device=device)
        len_old_data = 0
    else:
        model = old_model
        len_old_data = old_model.train_X.shape[0]
        model.train_X = trainX
        model.train_Y = trainY
        model.set_train_data(trainX, trainY.ravel(), strict=False)
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model).to(device="cuda")
    # iterator = tqdm.tqdm(range(epochs), desc="Train")
    pretrain_ratio = 0.1
    # prefit hyperparameter for OGP
    print('initial train')
    for i in range(int(epochs*pretrain_ratio)):
        # Zero backprop gradients
        optimizer.zero_grad()
        # Get output from model
        output = model(trainX)
        # Calc loss and backprop derivatives
        trainY = trainY.to(device="cuda")
        with gpytorch.settings.cholesky_jitter(1e-1):
            loss = -mll(output, trainY)
        loss.sum().backward()
        # iterator.set_postfix(loss=loss.item())
        optimizer.step()
        torch.cuda.empty_cache()
        print(i, loss.sum().item(), end='\r')
    # get OGP
    print('')
    print('get OGP')
    model.eval()
    model.fit_ogp(len_old_data)
    model.train()
    print('post train')
    for i in range(int(epochs*(1-pretrain_ratio))):
        # Zero backprop gradients
        optimizer.zero_grad()
        # Get output from model
        output = model(trainX)
        # Calc loss and backprop derivatives
        trainY = trainY.to(device="cuda")
        with gpytorch.settings.cholesky_jitter(1e-1):
            loss = -mll(output, trainY)
        loss.sum().backward()
        # iterator.set_postfix(loss=loss.item())
        optimizer.step()
        torch.cuda.empty_cache()
        print(i, loss.sum().item(), end='\r')
    
    print('')
    
    return model.eval(), likelihood.eval()

# class VecchiaGP():
#     def __init__(self, trainX, trainY, likelihood,dim=None):
#         # Convert/reshape the training data 
#         trainX     = trainX.to(device='cpu')
#         trainY     = trainY.reshape((trainY.shape[0], )).to(device='cpu')
#         z = (trainY - trainY.mean()) / trainY.std()
        
#         # Define attributes
#         self.n          = trainX.shape[0]
#         self.m          = int(7.2 * np.log10(self.n) ** 2)
#         self.dim        = dim
        
#         # Define mean and covariance modules
#         mean_module = ZeroMean()
#         if trainX is None:
#             assert dim is not None
#             covar_module = ScaleKernel(MaternKernel(ard_num_dims = self.dim, lengthscale_constraint = Interval(0.005, 10.0)))
#         else:
#             covar_module = ScaleKernel(MaternKernel(ard_num_dims = trainX.shape[1], lengthscale_constraint = Interval(0.005, 10.0)))
#         self.covar_module = covar_module
#         # print("length scale", self.covar_module.base_kernel.lengthscale)
        
#         self.neighbor_oracle = ExactOracle(trainX, z, self.m)
#         prediction_stategy = IndependentRF()
#         input_transform = Identity(d = self.dim)
#         self.model = RFVecchia(covar_module, mean_module, likelihood, 
#             self.neighbor_oracle, prediction_stategy, input_transform)
#         self.parameters = self.model.parameters()
#         self.likelihood = self.model.likelihood
#         self.model.dim = dim
    
#     def forward(self, x):
#         # print(trainX.shape, trainY.shape, neighbor_oracle)
#         return self.model(x)

# def train_vecchia(trainX, trainY, dim, epochs):
#     likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
#     model = VecchiaGP(trainX, trainY, likelihood, dim)
   
#     batch_size = np.min([model.n, 128])
#     opt_func = torch.optim.Adam
#     stopping_options = {"maxiter": epochs, "n_window": 20, "rel_tol":1e-5}

#     stop = False

#     stopping_criterion = ExpMAStoppingCriterion(**stopping_options)
#     optimizer = opt_func(model.parameters, lr = 0.05)
#     mll = ExactMarginalLogLikelihood(likelihood, model.model)
#     num_batches = model.n // batch_size
#     ind = np.arange(model.n)
#     np.random.shuffle(ind)
#     # if kwargs.get("tracking", False):
#     #     lik = []
    
#     # print(num_batches)
#     e = 0
#     verbose = False

#     while not stop:
#         # if verbose and (e % 10) == 0:    
#         #     print("epoch : {}".format(e))
            
#         e += 1
#         np.random.shuffle(ind)
#         batches = np.array_split(ind, num_batches)
#         for batch in batches:

#             bs = batch.shape[0]
#             optimizer.zero_grad()
#             x_batch, y_batch = model.neighbor_oracle[batch]
#             # print(y_batch)
#             # print(len(batch), trainX.shape)
#             print("batch: ", batch)
#             print("y_batch: ", y_batch)
#             output = model.forward(torch.from_numpy(batch).to(device='cpu'))
#             print(output)
#             loss = -1 * mll(output, y_batch) * model.n / bs
#             loss.backward()
#             optimizer.step()
            
#             stop = stopping_criterion.evaluate(fvals=loss.detach())
#             if stop: break
            
#     return model.model.eval(), model.likelihood.eval()

class NNGP(ApproximateGP):
    def __init__(self, inducing_points, likelihood, k=256, training_batch_size=256):
        m, dim = inducing_points.shape
        self.dim = dim
        self.m = m
        self.k = k
        
        variational_distribution = gpytorch.variational.MeanFieldVariationalDistribution(m)
        
        if torch.cuda.is_available():
            inducing_points = inducing_points.cuda()
            
        variational_strategy = NNVariationalStrategy(self, inducing_points, variational_distribution, k=k,
                                                     training_batch_size=training_batch_size)

        super(NNGP, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=dim, 
                                                                      lengthscale_constraint=Interval(0.005, inducing_points.shape[1])))
        
        self.likelihood = likelihood
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    
def train_NNGP(inducing_points=None, induce_size=10, train_loader=None, k=256, training_batch_size=64, epochs=20,
               dtype=torch.float, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    if inducing_points is None:
        # inducing_points = draw_sobol_samples(bounds=bounds, n=1, q=induce_size).squeeze(0)
        sobol = SobolEngine(train_loader.dataset.tensors[0].size(1), scramble=True)
        inducing_points = sobol.draw(induce_size).to(dtype=dtype, device=device)
    
    
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = NNGP(inducing_points, likelihood, k, training_batch_size)
    
    if torch.cuda.is_available():
        likelihood = likelihood.cuda()
        model = model.cuda()
        
    model.train()
    likelihood.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    # mll = VariationalELBO(likelihood, model, num_data=train_y.size(0))
    mll = VariationalELBO(likelihood, model, num_data=len(train_loader.dataset))
    
    for epoch in range(epochs):
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            # Obtain the y_batch using indices. It is important to keep the same order of train_x and train_y
            # print(train_y.shape)
            output = model(x_batch)
            train_y = y_batch.to(device="cuda")
            
            # print("output: ", output)
            # print("train_y: ", train_y)
            
            loss = -mll(output, y_batch)
            # minibatch_iter.set_postfix(loss=loss.item())
            loss.backward()
            optimizer.step()
            
    return model.eval(), likelihood.eval()
