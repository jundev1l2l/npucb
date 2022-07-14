import numpy as np
import torch
import gpytorch
from attrdict import AttrDict
'''
Reference: https://docs.gpytorch.ai/en/v1.5.1/examples/02_Scalable_Exact_GPs/Simple_GP_Regression_CUDA.html
'''

from util.base_config import BaseConfig


class GPUcbBanditConfig(BaseConfig):
    def set_subconfig_class_hash(self):
        self.subconfig_class_hash = {
            "name": str,
            "kernel": str,
            "alpha": float,
            "n_max_context": int,
            "online": bool,  # todo: delete this config from yaml files
        }


class GPUcbBandit:
    """
    DeepUCB algorithm implementation
    """
    def __init__(self, config, num_arms, dim_context, feature_extractor, rank, debug):  # different
        """
        Parameters
        ----------
        alpha : number
            ucb parameter
        """
        self.num_arms = num_arms
        self.dim_context = dim_context
        self.kernel = config.kernel if hasattr(config, "kernel") else "rbf"
        self.alpha = round(config.alpha, 2)
        self.n_max_context = config.n_max_context
        self.name = f"GP-UCB_ALPHA={self.alpha}_MAXCTX={config.n_max_context}"
        self.detail = f"GP-Ucb (α={self.alpha})"
        self.feature_extractor = feature_extractor
        self.rank = rank

    def init(self):
        self.history = AttrDict({"context": np.zeros((1, self.dim_context + 1)), "reward": np.zeros((1, 1))})

    def add_new_arms(self, num_new_arms):
        self.num_arms += num_new_arms
    
    def choose_arm(self, context):
        xc, yc, xt = self.get_input(context)
        mu, sigma = self.predict(xc, yc, xt)
        ucbs = mu + self.alpha * sigma
        
        return np.argmax(ucbs, axis=-1).item()
    
    def reward_distribution(self, context):
        xc, yc, xt = self.get_input(context)
        mu, sigma = self.predict(xc, yc, xt)
        ucbs = mu + self.alpha * sigma
        
        return mu, sigma, ucbs

    def get_input(self, context):
        xc = self.history.context
        yc = self.history.reward
        context_repeated = np.repeat(np.expand_dims(context, axis=0), self.num_arms, axis=0)
        arms = np.arange(self.num_arms).reshape(-1, 1)
        arms_repeated = arms if context.ndim == 1 else np.repeat(np.expand_dims(arms, axis=1), context.shape[0], axis=1)
        xt = np.concatenate([context_repeated, arms_repeated], axis=-1)
        
        xc, yc, xt = list(map(self.transform_input, [xc, yc, xt]))

        if self.feature_extractor is not None:
            xc = self.feature_extractor.encode(xc)
            xt = self.feature_extractor.encode(xt)
        
        if xt.shape[0] > 1:
            xc, yc = list(map(lambda x: torch.cat([x,] * xt.shape[0], dim=0),[xc, yc]))
            
        return xc, yc, xt  # [B,N,d]

    def transform_input(self, data):
        data = torch.FloatTensor(data)
        while data.ndim < 3:
            data = data.unsqueeze(dim=0)
        data = data.cpu() if self.rank < 0 else data.cuda()
        return data

    def predict(self, xc, yc, xt): # different
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ExactGPModel(xc, yc.squeeze(), likelihood, self.kernel)
        model = model.cpu() if self.rank < 0 else model.cuda()
        likelihood = likelihood.cpu() if self.rank < 0 else likelihood.cuda()
        model.train()
        likelihood.train()

        # Use the adam optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        training_iter = 50
        for i in range(training_iter):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = model(xc)
            # Calc loss and backprop gradients
            loss = -mll(output, yc).mean()
            loss.backward()
            print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                i + 1, training_iter, loss.item(),
                model.covar_module.base_kernel.lengthscale.item(),
                model.likelihood.noise.item()
            ))
            optimizer.step()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            model.eval()
            likelihood.eval()
            pred = model(xt)
            mu = pred.mean
            sigma = pred.variance**(0.5)
        
        del optimizer
        del model
        del likelihood

        return mu.cpu(), sigma.cpu()

    def update(self, context, action, reward):
        context, action, reward = list(map(self.transform_update, [context, action, reward]))
        self.update_history(context, action, reward)

    def transform_update(self, data):
        data = np.array(data)
        if data.ndim < 2:
            data = data.reshape(1, -1)
        return data

    def update_history(self, context, action, reward):
        new_context = np.concatenate([context, action], axis=-1)
        new_reward = reward
        self.history.context = np.concatenate([self.history.context, new_context], axis=0)[-self.n_max_context:]
        self.history.reward = np.concatenate([self.history.reward, new_reward], axis=0)[-self.n_max_context:]


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        kernel = {
            "rbf": gpytorch.kernels.RBFKernel(),
            "matern": gpytorch.kernels.MaternKernel(),
            "periodic": gpytorch.kernels.PeriodicKernel(),
        }[kernel]
        self.covar_module = gpytorch.kernels.ScaleKernel(kernel)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
