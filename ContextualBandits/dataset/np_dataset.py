import torch
import numpy as np
from random import sample

from util.base_config import BaseConfig
from dataset.base_dataset import BaseDataset


class NPDatasetConfig(BaseConfig):
    def set_subconfig_class_hash(self):
        self.subconfig_class_hash = {
            "name": str,
            "mixed_per_batch": bool,
            "n_ctx": int,
            "n_tar": int,
        }


class NPDataset(BaseDataset):
    def __init__(self, config, *args, **kwargs):
        super(NPDataset, self).__init__(config, *args, **kwargs)
        self.n_ctx = config.n_ctx
        self.n_tar = config.n_tar
        self.batch_size = 1
        self.n = 0

    def __len__(self):
        return 1024 if self.debug else 16384
    
    def __getitem__(self, idx):
        return self.get_np_data()

    def get_np_data(self):
        ctx_idx = sample(range(self.n_events), self.n_ctx)
        tar_idx = sample(range(self.n_events), self.n_tar)
        
        xc, yc, xt, yt = [], [], [], []

        for idx in ctx_idx:
            x, y = self.get_xy(idx, self.mixed_data_idx)
            xc.append(x)
            yc.append(y)
            
        for idx in tar_idx:
            x, y = self.get_xy(idx, self.mixed_data_idx)
            xt.append(x)
            yt.append(y)

        data = [xc, yc, xt, yt]
        data = list(map(lambda x: np.concatenate(x, axis=0), data))
        data = list(map(torch.FloatTensor, data))
        data = list(map(lambda x: x.squeeze(dim=0), data))  # todo: merged from NPRandomDataset (but not sure if its also right for NPDataset)

        self.n += 1

        if self.mixed_per_batch:
            if self.n % self.batch_size == 0:
                self.mixed_data_idx = np.random.randint(len(self.data_sampler)) if self.mixed_data else -1
        else:
            self.mixed_data_idx = np.random.randint(len(self.data_sampler)) if self.mixed_data else -1

        return data
