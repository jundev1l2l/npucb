import torch
from torch.distributions import MultivariateNormal, StudentT
from attrdict import AttrDict
import math

from util.base_config import BaseConfig


class GPDataSamplerConfig(BaseConfig):
    def set_subconfig_class_hash(self):
        self.subconfig_class_hash = {
            "name": str,
            "dim_context": int,
            "kernel_type": str,
            "t_noise": float,
            "x_range": list,
        }


class GPDataSampler:    
    def __init__(self, config):
        self.name = config.name
        self.dim_context = config.dim_context
        self.kernel_type = config.kernel_type
        self.kernel = {"rbf": RBFKernel, "matern": MaternKernel, "periodic": PeriodicKernel}.get(self.kernel_type)()
        self.t_noise = config.t_noise
        self.x_range = config.x_range

    def sample(self, num_ctx, num_tar):

        num_points = num_ctx + num_tar  # N = Nc + Nt
        x = self.x_range[0] + (self.x_range[1] - self.x_range[0]) * torch.rand([num_points, self.dim_context])  # [N, Dx]
        mean = torch.zeros(num_points)  # [N,]
        cov = self.kernel(x)  # [N, N]
        y = MultivariateNormal(mean, cov).rsample().unsqueeze(-1)  # [N, Dy=1]

        if self.t_noise is not None:
            if self.t_noise == -1:
                t_noise = 0.15 * torch.rand(y.shape)  # [N, Dy=1]
            else:
                t_noise = self.t_noise
            y += t_noise * StudentT(2.1).rsample(y.shape)

        xc = x[:num_ctx]
        yc = y[:num_ctx]
        xt = x[num_ctx:]
        yt = y[num_ctx:]

        return xc, yc, xt, yt
        #  xc: [Nc, Dx]
        #  yc: [Nt, Dy=1]
        #  xt: [Nt, Dx]
        #  yt: [Nt, Dy=1]


class RBFKernel:
    def __init__(self, sigma_eps=2e-2, max_length=0.6, max_scale=1.0):
        self.sigma_eps = sigma_eps
        self.max_length = max_length
        self.max_scale = max_scale

    # x: num_points * dim  [N,Dx=1]
    def __call__(self, x):
        length = 0.1 + (self.max_length-0.1) * torch.rand(1)
        scale = 0.1 + (self.max_scale-0.1) * torch.rand(1)

        # num_points * num_points * dim  [N,N,1]
        dist = (x.unsqueeze(-2) - x.unsqueeze(-3)) / length

        # num_points * num_points  [N,N]
        cov = scale.pow(2) * torch.exp(-0.5 * dist.pow(2).sum(-1)) + self.sigma_eps**2 * torch.eye(x.shape[-2])

        return cov  # [N,N]


class MaternKernel:
    def __init__(self, sigma_eps=2e-2, max_length=0.6, max_scale=1.0):
        self.sigma_eps = sigma_eps
        self.max_length = max_length
        self.max_scale = max_scale

    # x: batch_size * num_points * dim
    def __call__(self, x):
        length = 0.1 + (self.max_length-0.1) * torch.rand(1)
        scale = 0.1 + (self.max_scale-0.1) * torch.rand(1)

        # num_points * num_points * dim  [N,N,1]
        dist = torch.norm((x.unsqueeze(-2) - x.unsqueeze(-3))/length, dim=-1)

        cov = scale.pow(2)*(1 + math.sqrt(5.0)*dist + 5.0*dist.pow(2)/3.0) * torch.exp(-math.sqrt(5.0) * dist) + self.sigma_eps**2 * torch.eye(x.shape[-2])

        return cov

class PeriodicKernel:
    def __init__(self, sigma_eps=2e-2, max_length=0.6, max_scale=1.0):
        self.sigma_eps = sigma_eps
        self.max_length = max_length
        self.max_scale = max_scale

    # x: batch_size * num_points * dim
    def __call__(self, x):
        p = 0.1 + 0.4 * torch.rand(1)
        length = 0.1 + (self.max_length-0.1) * torch.rand(1)
        scale = 0.1 + (self.max_scale-0.1) * torch.rand(1)

        # num_points * num_points * dim  [N,N,1]
        dist = ((x.unsqueeze(-2) - x.unsqueeze(-3)).pow(2).sum(-1)).pow(0.5)
        cov = scale.pow(2) * torch.exp(- 2*(torch.sin(math.pi*dist/p)/length).pow(2)) + self.sigma_eps**2 * torch.eye(x.shape[-2])

        return cov
