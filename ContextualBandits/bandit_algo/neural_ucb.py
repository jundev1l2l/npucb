import numpy as np
import torch
from random import sample, randint
from attrdict import AttrDict
from torch.optim import Adam

from util.base_config import BaseConfig
from model.build import build_model
from model import MODEL_CONFIG_DICT


class NeuralUcbBanditConfig(BaseConfig):
    def set_subconfig_class_hash(self):
        self.subconfig_class_hash = {
            "name": str,
            "alpha": float,
            "backbone": MODEL_CONFIG_DICT,
        }


class NeuralUcbBandit:
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
        self.name = f"NeuralUCB_ALPHA={self.alpha}"
        self.detail = f"NeuralUcb (α={self.alpha})"
        self.feature_extractor = feature_extractor
        self.rank = rank

    def predict(self, xc, yc, xt):
        with torch.no_grad():
            self.backbone.eval()
            outs = self.backbone.predict(xt)
            mu = outs.ys.cpu().numpy()
            sigma = None  # todo
        
        return mu, sigma
