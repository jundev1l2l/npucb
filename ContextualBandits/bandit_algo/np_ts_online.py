import numpy as np
import torch
from torch.distributions import Normal

from util.base_config import BaseConfig
from bandit_algo.np_ucb_online import NPUcbOnlineBandit
from model import MODEL_CONFIG_DICT


class NPTSOnlineBanditConfig(BaseConfig):
    def set_subconfig_class_hash(self):
        self.subconfig_class_hash = {
            "name": str,
            "alpha": float,
            "n_max_context": int,
            "n_bs": int,
            "backbone": MODEL_CONFIG_DICT,
            "online": bool,  # todo: delete this config from yaml files
            "num_rollouts": int,
            "num_steps_per_update": int,
            "lr": float,
            "batch_size": int,
            "n_ctx_range": list,
            "n_tar_range": list,
        }


class NPTSOnlineBandit(NPUcbOnlineBandit):
    """
    NeuralProcesses + ThopmsonSampling algorithm implementation
    """
    def __init__(self, config, *args, **kwargs):
        super(NPTSOnlineBandit, self).__init__(config, *args, **kwargs)
        self.name = f"{config.backbone.name}-ThompsonSampling-Online"
        self.detail = f"NP-TS-Online ({config.backbone.name})"

    def choose_arm(self, context):
        if self.history.reward.shape[0] < self.num_rollouts:
            return np.random.randint(low=0, high=self.num_arms)
        xc, yc, xt = self.get_input(context)
        mu, sigma = self.predict(xc, yc, xt)
        sigma[sigma==0.0] = 1e-4  # give very small value for numerical stability
        posterior = Normal(torch.Tensor(mu), torch.Tensor(sigma))  # [B=1, Nt]
        samples = posterior.sample()
        
        return np.argmax(samples, axis=-1).squeeze()
