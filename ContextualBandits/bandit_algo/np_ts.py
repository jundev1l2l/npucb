import numpy as np
import torch
from torch.distributions import Normal

from util.base_config import BaseConfig
from bandit_algo.np_ucb import NPUcbBandit
from model import MODEL_CONFIG_DICT


class NPTSBanditConfig(BaseConfig):
    def set_subconfig_class_hash(self):
        self.subconfig_class_hash = {
            "name": str,
            "alpha": float,
            "n_max_context": int,
            "n_bs": int,
            "backbone": MODEL_CONFIG_DICT,
            "online": bool,  # todo: delete this config from yaml files
        }


class NPTSBandit(NPUcbBandit):
    """
    NeuralProcesses + ThopmsonSampling algorithm implementation
    """
    def __init__(self, config, *args, **kwargs):
        super(NPTSBandit, self).__init__(config, *args, **kwargs)
        self.name = f"{config.backbone.name}-ThompsonSampling"
        self.detail = f"NP-TS ({config.backbone.name})"

    def choose_arm(self, context):
        xc, yc, xt = self.get_input(context)
        mu, sigma = self.predict(xc, yc, xt)
        sigma[sigma==0.0] = 1e-4  # give very small value for numerical stability
        posterior = Normal(torch.Tensor(mu), torch.Tensor(sigma))  # [B=1, Nt]
        samples = posterior.sample()
        
        return np.argmax(samples, axis=-1).squeeze()
