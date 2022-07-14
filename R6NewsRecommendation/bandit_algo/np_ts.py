import numpy as np
import torch
from torch.distributions import Normal

from util.base_config import BaseConfig
from model import MODEL_CONFIG_DICT
from bandit_algo.np_ucb import NPUcbBandit


class NPTSBanditConfig(BaseConfig):
    def set_subconfig_class_hash(self):
        self.subconfig_class_hash = {
            "name": str,
            "alpha": float,
            "n_max_context": int,
            "n_bs": int,
            "online": bool,
            "backbone": MODEL_CONFIG_DICT,
        }


class NPTSBandit(NPUcbBandit):
    """
    NeuralProcesses + ThopmsonSampling algorithm implementation
    """
    def __init__(self, *args, **kwargs):
        super(NPTSBandit, self).__init__(*args, **kwargs)
        self.name = f"{self.backbone.name}-ThompsonSampling"
        self.detail = f"NP-TS (backbone={self.backbone.name}, n_max_context={self.n_max_context}, n_bs={self.n_bs}, online={self.online})"

    def choose_arm(self, t, user, pool_idx, features):
        if self.online and (self.history.reward.shape[0] < self.n_rollout):
            return np.random.randint(low=0, high=len(pool_idx))
        xc, yc, xt = self.get_input(np.array(user), features, np.array(pool_idx))
        mu, sigma = self.predict(xc, yc, xt)
        posterior = Normal(torch.Tensor(mu), torch.Tensor(sigma))  # [B=1, Nt]
        samples = posterior.sample()
        return np.argmax(samples, axis=-1).squeeze()
