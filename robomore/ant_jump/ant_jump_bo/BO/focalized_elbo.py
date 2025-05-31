import torch
from gpytorch.mlls import VariationalELBO



class  FocalizedVariationalELBO(VariationalELBO):

    def __init__(self, likelihood, model, num_data, beta=1, combine_terms=True):
        super().__init__(likelihood, model, num_data, beta, combine_terms)
    
    def _log_likelihood_term(self, variational_dist_f, target, **kwargs):
        # not sum for futher weighting
        return self.likelihood.expected_log_prob(target, variational_dist_f, **kwargs)
    
    
    def forward(self, variational_dist_f, target, train_x, test_distribution, **kwargs):
        r"""
        Computes the Variational ELBO given :math:`q(\mathbf f)` and :math:`\mathbf y`.
        Calling this function will call the likelihood's :meth:`~gpytorch.likelihoods.Likelihood.expected_log_prob`
        function.

        :param ~gpytorch.distributions.MultivariateNormal variational_dist_f: :math:`q(\mathbf f)`
            the outputs of the latent function (the :obj:`gpytorch.models.ApproximateGP`)
        :param torch.Tensor target: :math:`\mathbf y` The target values
        :param kwargs: Additional arguments passed to the
            likelihood's :meth:`~gpytorch.likelihoods.Likelihood.expected_log_prob` function.
        :rtype: torch.Tensor
        :return: Variational ELBO. Output shape corresponds to batch shape of the model/input data.
        """
        # return super().forward(variational_dist_f, target, **kwargs)
        # Get likelihood term and KL term
        
        # print(num_batch)
        # do not divide for futher weighting
        log_likelihood = self._log_likelihood_term(variational_dist_f, target, **kwargs)
        likelihood = torch.exp(log_likelihood)
        # weighted_likelihood = (weight*likelihood)
        # print('ll in', likelihood.max().item(), likelihood.min().item(), 
        #       log_likelihood.max().item(), log_likelihood.min().item(), 
        #     #   weight.max().item(), weight.min().item()
        #       )
        kl_divergence = self.model.variational_strategy.kl_divergence().div(self.num_data / self.beta)
        
        # Weighting by covariance
        # K_xt = self.model.covar_module(train_x, self.model.variational_strategy.sub_test_pos).to_dense() # n_train x n_test
        if isinstance(test_distribution, torch.Tensor):
            diff = torch.abs(train_x - test_distribution).sum(-1)
            # print('')
            # print(len(diff[diff==0]))
            self.num_in_region = len(diff[diff==0])
            weight = torch.ones(len(train_x)).to(dtype=train_x.dtype, device=train_x.device)
            self_cov = self.model.covar_module(train_x[0:1], train_x[0:1]).to_dense().item()
            weight = self_cov*weight
            # print(test_distribution[:10], train_x[:10])
            # raise NotImplementedError
            if len(train_x[diff > 0]) > 0:
                K_xt = self.model.covar_module(train_x[diff>0], test_distribution[diff>0]).to_dense()
                weight[diff>0] = torch.diagonal(K_xt)
            # print(train_x, test_distribution)
            # print(K_xt)
            # print(torch.diagonal(K_xt))
            # print(weight)
            weight = weight/self_cov
            
            # raise NotImplementedError
        else:
            self.model.variational_strategy.sub_test_pos = test_distribution(self.model.variational_strategy.inducing_points.shape[0]).to(dtype=train_x.dtype, device=train_x.device)
            
            K_xt = self.model.covar_module(train_x, self.model.variational_strategy.sub_test_pos).to_dense() # n_train x n_test

            # K_xt = self.model.covar_module(train_x, self.model.variational_strategy.inducing_points).to_dense() # n_train x n_test
            # print(train_x)
            # print(self.model.variational_strategy.sub_test_pos)
            # weight = torch.max(K_xt, dim=1).values
            ub = torch.max(self.model.variational_strategy.sub_test_pos, dim=0).values
            lb = torch.min(self.model.variational_strategy.sub_test_pos, dim=0).values
            in_dis = torch.zeros(len(train_x))
            # print(lb, ub)
            train_in_dis = torch.sum(torch.clamp(train_x, lb, ub)-train_x, dim=-1)
            in_dis[train_in_dis == 0] = 1
            # in_dis[lb<train_x<ub] = 1
            weight = torch.max(K_xt, dim=1).values
                # print(weight)
        
            

            # print(in_dis)
            weight[in_dis==1] = weight.clone().max()
            weight = weight/weight.sum()
        # weight = in_dis.to(device=train_x.device)
        # print('good weight', weight.max(), weight.min())
        # 
        self.weight=weight
        self.ll = log_likelihood
        # print(weight.m)
        # raise NotImplementedError
        # raise NotImplementedError
        
        # print(likelihood)
        # print(weighted_likelihood)
        # raise NotImplementedError
        log_likelihood = (weight*log_likelihood).sum()/(weight.sum() + 1e-7)
        # log_likelihood = log_likelihood.sum(-1).div(self.num_data)
        # Add any additional registered loss terms
        added_loss = torch.zeros_like(log_likelihood)
        had_added_losses = False
        for added_loss_term in self.model.added_loss_terms():
            added_loss.add_(added_loss_term.loss())
            had_added_losses = True

        # Log prior term
        log_prior = torch.zeros_like(log_likelihood)
        for name, module, prior, closure, _ in self.named_priors():
            log_prior.add_(prior.log_prob(closure(module)).sum().div(self.num_data))
        # print(log_likelihood.item(), kl_divergence.item(), log_prior.item(), added_loss.item())

        if self.combine_terms:
            return log_likelihood - kl_divergence + log_prior - added_loss
        else:
            if had_added_losses:
                return log_likelihood, kl_divergence, log_prior, added_loss
            else:
                return log_likelihood, kl_divergence, log_prior

        
        