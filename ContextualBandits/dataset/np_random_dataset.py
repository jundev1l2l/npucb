import torch
import numpy as np
from random import sample, randint

from util.base_config import BaseConfig
from dataset.base_dataset import BaseDataset


class NPRandomDatasetConfig(BaseConfig):
    def set_subconfig_class_hash(self):
        self.subconfig_class_hash = {
            "name": str,
            "mixed_per_batch": bool,
            "n_ctx_range": list,
            "n_tar_range": list,
        }


class NPRandomDataset(BaseDataset):
    def __init__(self, config, *args, **kwargs):
        super(NPRandomDataset, self).__init__(config, *args, **kwargs)
        self.n_ctx_range = config.n_ctx_range
        self.n_tar_range = config.n_tar_range
        self.batch_size = 1
        self.n = 0
        self.draw_numbers()

    def __len__(self):
        return 1024 if self.debug else 16384

    def __getitem__(self, idx):
        return self.get_np_data()

    def get_np_data(self):
        ctx_idx = sample(range(self.n_events), self.n_ctx)
        tar_idx = sample(range(self.n_events), self.n_tar)
        
        xc, yc, xt, yt, delta_c, delta_t = [], [], [], [], [], []
        
        for idx in ctx_idx:
            x, y, delta = self.get_xy(idx)
            xc.append(x)
            yc.append(y)
            delta_c.append(delta)

        for idx in tar_idx:
            x, y, delta = self.get_xy(idx)
            xt.append(x)
            yt.append(y)
            delta_t.append(delta)

        data = [xc, yc, xt, yt, delta_c, delta_t]
        data = list(map(lambda x: np.concatenate(x, axis=0), data))
        data = list(map(torch.FloatTensor, data))
        data = list(map(lambda x: x.squeeze(dim=0), data))

        self.n += 1

        if self.n % self.batch_size == 0:
            self.draw_numbers()
        
        if self.mixed_per_batch:
            if self.n % self.batch_size == 0:
                self.mixed_data_idx = np.random.randint(len(self.data_sampler)) if self.mixed_data else -1
        else:
            self.mixed_data_idx = np.random.randint(len(self.data_sampler)) if self.mixed_data else -1

        return data

    def draw_numbers(self):
        self.n_ctx = randint(*self.n_ctx_range)
        self.n_tar = randint(*self.n_tar_range)
