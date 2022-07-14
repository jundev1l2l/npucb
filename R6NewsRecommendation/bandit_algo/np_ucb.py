import numpy as np
import torch
from attrdict import AttrDict

from util.base_config import BaseConfig
from model import MODEL_CONFIG_DICT
from model.build import build_model


class NPUcbBanditConfig(BaseConfig):
    def set_subconfig_class_hash(self):
        self.subconfig_class_hash = {
            "name": str,
            "alpha": float,
            "n_max_context": int,
            "n_bs": int,
            "online": bool,
            "backbone": MODEL_CONFIG_DICT,
        }


class NPUcbBandit:
    """
    DeepUCB algorithm implementation
    """
    def __init__(self, config, num_arms, dim_context, rank, debug):
        """
        Parameters
        ----------
        alpha : number
            ucb parameter
        """
        self.num_arms = num_arms
        self.dim_context = dim_context
        self.history = AttrDict({"context": np.zeros((1, self.dim_context)), "reward": np.zeros((1, 1))})
        self.alpha = round(config.alpha, 1)
        self.backbone = build_model(
            config=config.backbone,
            rank=rank,
            debug=debug,
        )
        self.n_max_context = config.n_max_context
        self.n_bs = config.n_bs
        self.online = config.online
        self.name = f"{config.backbone.name}-UCB_ALPHA={self.alpha}_MAXCTX={config.n_max_context}"
        self.detail = f"NP-Ucb (backbone={config.backbone.name}, α={self.alpha})"
        # self.n_rollout = 10

    def choose_arm(self, t, user, pool_idx, features):
        if self.online and (self.history.reward.shape[0] < self.n_rollout):
            return np.random.randint(low=0, high=len(pool_idx))
        xc, yc, xt = self.get_input(np.array(user), features, np.array(pool_idx))
        mu, sigma = self.predict(xc, yc, xt)
        ucbs = mu + self.alpha * sigma  # [B=1, Nt]
        return np.argmax(ucbs, axis=-1).squeeze()

    def get_input(self, user, features, pool_idx):
        xc = self.history.context
        yc = self.history.reward
        user_repeated = np.repeat(np.expand_dims(user, axis=0), len(pool_idx), axis=0)
        items = features[pool_idx]
        xt = np.concatenate([user_repeated, items], axis=1)
        xc, yc, xt = list(map(lambda x: torch.FloatTensor(x).unsqueeze(dim=0), [xc, yc, xt]))
        return xc, yc, xt  # [B,N,d]

    def predict(self, xc, yc, xt):
        with torch.no_grad():
            self.backbone.eval()
            if self.backbone.name in ["BANP", "BNP"]:
                outs = self.backbone.cuda().predict(xc.cuda(), yc.cuda(), xt.cuda(), num_samples=self.n_bs)
            else:
                outs = self.backbone.cuda().predict(xc.cuda(), yc.cuda(), xt.cuda(), num_samples=1)
            mu, sigma = outs.loc.cpu().numpy(), outs.scale.cpu().numpy()
            mu = np.mean(mu, axis=0)
            sigma = np.mean(sigma, axis=0)
            
        """
        squeeze num_samples(Ns) dimension which is always 1
        """
        mu, sigma = list(map(lambda x: np.squeeze(x, axis=(0, -1)), [mu, sigma]))
        return mu, sigma

    def update(self, displayed, reward, user, pool_idx, features):
        self.update_history(user, features, pool_idx, displayed, reward)
        if self.online:  # todo: online evaluation (for online update)
            outs = self.update_np()

    def update_history(self, user, features, pool_idx, displayed, reward):
        displayed_idx = pool_idx[displayed]
        item = features[displayed_idx]
        new_context = np.concatenate([np.array(user), np.array(item)], axis=-1).reshape(1, self.dim_context)
        new_reward = np.array(reward).reshape(1, 1)
        self.history.context = np.concatenate([self.history.context, new_context], axis=0)[-self.n_max_context:]
        self.history.reward = np.concatenate([self.history.reward, new_reward], axis=0)[-self.n_max_context:]
    
    def update_np(self):  # todo: updating logic (for online update)
        self.backbone.train()
        batch = self.get_batch()
        outs = self.backbone.forward(batch,
                                     num_samples=1,
                                     num_bs=10,
                                     loss="nll",
                                     reduce_ll=True)
        return outs
        
    def get_batch(self):  # todo: get_batch (for online update)
        xc = self.history.context
        batch = 0
        return batch
