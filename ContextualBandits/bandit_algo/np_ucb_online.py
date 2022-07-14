import numpy as np
import torch
from random import sample, randint
from attrdict import AttrDict
from torch.optim import Adam

from util.base_config import BaseConfig
from model import MODEL_CONFIG_DICT
from bandit_algo.np_ucb import NPUcbBandit


class NPUcbOnlineBanditConfig(BaseConfig):
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


class NPUcbOnlineBandit(NPUcbBandit):
    """
    DeepUCB algorithm implementation
    """

    def __init__(self, config, *args, **kwargs):
        """
        Parameters
        ----------
        alpha : number
            ucb parameter
        """
        self.ckpt = config.backbone.ckpt if hasattr(config.backbone, "ckpt") else None
        delattr(config.backbone, "ckpt") if hasattr(config.backbone, "ckpt") else None  # do not use the pretrained model
        super(NPUcbOnlineBandit, self).__init__(config, *args, **kwargs)
        self.name = f"{config.backbone.name}-UCB-Online_ALPHA={self.alpha}_MAXCTX={config.n_max_context}"
        self.detail = f"NP-Ucb-Online (backbone={config.backbone.name}, α={self.alpha})"
        self.num_rollouts = config.num_rollouts
        self.num_steps_per_update = config.num_steps_per_update
        self.lr = config.lr
        self.batch_size = config.batch_size
        self.n_ctx_range = config.n_ctx_range
        self.n_tar_range = config.n_tar_range
        self.optimizer = Adam(params=self.backbone.parameters(), lr=self.lr)

    def choose_arm(self, context):
        if self.history.reward.shape[0] < self.num_rollouts:
            return np.random.randint(low=0, high=self.num_arms)
        xc, yc, xt = self.get_input(context)
        mu, sigma = self.predict(xc, yc, xt)
        ucbs = mu + self.alpha * sigma  # [B=1, Nt]
        return np.argmax(ucbs, axis=-1).item()

    def update(self, context, action, reward):
        context, action, reward = list(
            map(self.transform_update, [context, action, reward]))
        self.update_history(context, action, reward)
        if self.history.reward.shape[0] < self.num_rollouts:
            self.update_np()

    def transform_update(self, data):
        data = np.array(data)
        if data.ndim < 2:
            data = data.reshape(1, -1)
        data = data.cpu() if self.rank < 0 else data.cuda()
        return data

    def update_history(self, context, action, reward):
        new_context = np.concatenate([context, action], axis=-1)
        new_reward = reward
        self.history.context = np.concatenate(
            [self.history.context, new_context], axis=0)[-self.n_max_context:]
        self.history.reward = np.concatenate(
            [self.history.reward, new_reward], axis=0)[-self.n_max_context:]

    def update_np(self):
        self.backbone.train()
        for idx in range(self.num_steps_per_update):
            self.optimizer.zero_grad()
            batch = self.get_batch()
            if self.backbone.name in ["BANP", "BNP"]:
                outs = self.backbone.forward(batch, num_samples=self.n_bs)
            else:
                outs = self.backbone.forward(batch, num_samples=1)
            outs.loss.backward()
            self.optimizer.step()

    def get_batch(self):
        batch_xc, batch_xt, batch_yc, batch_yt = [], [], [], []

        context = self.history.context
        reward = self.history.reward
        num_contexts = context.shape[0]

        n_ctx = min(randint(*self.n_ctx_range), num_contexts)
        n_tar = min(randint(*self.n_tar_range), num_contexts)

        for idx in range(self.batch_size):

            ctx_idx = sample(range(num_contexts), n_ctx)
            tar_idx = sample(range(num_contexts), n_tar)

            xc = context[ctx_idx, :]
            xt = context[tar_idx, :]
            yc = reward[ctx_idx, :]
            yt = reward[tar_idx, :]

            xc, xt, yc, yt = list(map(torch.FloatTensor, [xc, xt, yc, yt]))

            batch_xc.append(xc)
            batch_xt.append(xt)
            batch_yc.append(yc)
            batch_yt.append(yt)

        batch_xc, batch_xt, batch_yc, batch_yt = list(
            map(lambda x: torch.stack(x, dim=0), [batch_xc, batch_xt, batch_yc, batch_yt]))

        batch = AttrDict()
        batch.xc, batch.xt, batch.yc, batch.yt = batch_xc, batch_xt, batch_yc, batch_yt
        batch.x = torch.cat([batch_xc, batch_xt], dim=1)
        batch.y = torch.cat([batch_yc, batch_yt], dim=1)

        return batch
