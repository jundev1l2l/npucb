import numpy as np
import torch
from random import sample, randint
from attrdict import AttrDict
from torch.optim import Adam

from util.base_config import BaseConfig
from model.build import build_model
from model import MODEL_CONFIG_DICT


class NPUcbBanditConfig(BaseConfig):
    def set_subconfig_class_hash(self):
        self.subconfig_class_hash = {
            "name": str,
            "alpha": float,
            "n_max_context": int,
            "n_bs": int,
            "backbone": MODEL_CONFIG_DICT,
            "online": bool,  # todo: delete this config from yaml files
        }


class NPUcbBandit:
    """
    DeepUCB algorithm implementation
    """
    def __init__(self, config, num_arms, dim_context, feature_extractor, rank, debug):
        """
        Parameters
        ----------
        alpha : number
            ucb parameter
        """
        self.num_arms = num_arms
        self.dim_context = dim_context
        self.alpha = round(config.alpha, 2)
        self.backbone = build_model(
            config=config.backbone,
            rank=rank,
            debug=debug,
        )
        self.n_max_context = config.n_max_context
        self.n_bs = config.n_bs
        self.name = f"{config.backbone.name}-UCB_ALPHA={self.alpha}_MAXCTX={config.n_max_context}"
        self.detail = f"NP-Ucb (backbone={config.backbone.name}, α={self.alpha})"
        self.feature_extractor = feature_extractor
        self.rank = rank

    def init(self):
        self.history = AttrDict({"context": np.zeros((1, self.dim_context + 1)), "reward": np.zeros((1, 1))})

    def add_new_arms(self, num_new_arms):
        self.num_arms += num_new_arms
    
    def choose_arm(self, context):
        xc, yc, xt = self.get_input(context)
        mu, sigma = self.predict(xc, yc, xt)
        ucbs = mu + self.alpha * sigma  # [B=1, Nt]
        
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

    def predict(self, xc, yc, xt):
        with torch.no_grad():
            self.backbone.eval()
            if self.backbone.name in ["BANP", "BNP"]:
                outs = self.backbone.predict(xc, yc, xt, num_samples=self.n_bs)
            else:
                outs = self.backbone.predict(xc, yc, xt, num_samples=1)
            mu, sigma = outs.loc.cpu().numpy(), outs.scale.cpu().numpy()
            mu = np.mean(mu, axis=0)
            sigma = np.mean(sigma, axis=0)
        
        """
        squeeze num_samples(Ns) dimension which is always 1
        """
        mu, sigma = list(map(lambda x: np.squeeze(x, axis=-1), [mu, sigma]))
        return mu, sigma

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
