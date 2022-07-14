import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from attrdict import AttrDict
from torch.optim import Adam
from tqdm import tqdm

from util.base_config import BaseConfig
from trainer.base_trainer import BaseTrainer


class FeatureExtractorTrainerConfig(BaseConfig):
    def set_subconfig_class_hash(self):
        self.subconfig_class_hash = {
            "name": str,
            "lr": float,
            "num_epochs": int,
            "loss": str,
            "clip_loss": float,
            "val_freq": int,
            "save_freq": int,
        }


class FeatureExtractorTrainer(BaseTrainer):
    def __init__(self, config, *args, **kwargs):
        super(FeatureExtractorTrainer, self).__init__(config, *args, **kwargs)
        self.model = self.feature_extractor
        self.feature_extractor = None

    def get_batch_dict(self, batch):  # different
        x, y = list(map(lambda x: x.cuda() if self.rank >= 0 else x.cpu(), batch))
        batch_dict = AttrDict({"x": x})

        return batch_dict
