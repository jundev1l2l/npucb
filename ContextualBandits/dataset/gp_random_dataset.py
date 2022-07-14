import torch
import numpy as np
from random import sample, randint

from util.base_config import BaseConfig
from dataset.np_random_dataset import NPRandomDataset


class GPRandomDatasetConfig(BaseConfig):
    def set_subconfig_class_hash(self):
        self.subconfig_class_hash = {
            "name": str,
            "length": int,
            "n_ctx_range": list,
            "n_tar_range": list,
        }


class GPRandomDataset(NPRandomDataset):
    def __init__(self, config, *args, **kwargs):
        super(NPRandomDataset, self).__init__(config, *args, **kwargs)
        self.n_ctx_range = config.n_ctx_range
        self.n_tar_range = config.n_tar_range
        self.length = config.length
        self.batch_size = 1
        self.n = 0
        self.draw_numbers()

    def __len__(self):
        return 1000 if self.debug else self.length

    def __getitem__(self, idx):
        return self.get_np_data()

    def set_data(self):
        pass

    def get_np_data(self):
        
        xc, yc, xt, yt = self.data_sampler.sample(self.n_ctx, self.n_tar)

        self.n += 1
        if self.n % self.batch_size == 0:
            self.draw_numbers()

        return [xc, yc, xt, yt]

    def draw_numbers(self):
        self.n_ctx = randint(*self.n_ctx_range)
        self.n_tar = randint(*self.n_tar_range)
