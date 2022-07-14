import torch
import numpy as np
from random import sample

from dataset.base_dataset import BaseDataset
from util.base_config import BaseConfig


class NPDatasetConfig(BaseConfig):
    def set_subconfig_class_hash(self):
        self.subconfig_class_hash = {
            "name": str,
            "n_ctx": int,
            "n_tar": int,
        }
        

class NPDataset(BaseDataset):
    def __init__(self, config, *args, **kwargs):
        super(NPDataset, self).__init__(config, *args, **kwargs)
        self.n_ctx = config.n_ctx
        self.n_tar = config.n_tar

    def __len__(self):
        return 1000 if self.debug else 50000

    def __getitem__(self, idx):
        return self.get_np_data()

    def get_np_data(self):
        ctx_idx = sample(range(self.num_events), self.n_ctx)
        tar_idx = sample(range(self.num_events), self.n_tar)

        xc, yc, xt, yt = [], [], [], []

        for idx in ctx_idx:
            x, y = self.get_xy(idx)
            xc.append(x)
            yc.append(y)

        for idx in tar_idx:
            x, y = self.get_xy(idx)
            xt.append(x)
            yt.append(y)

        data = [xc, yc, xt, yt]
        data = list(map(lambda x: np.concatenate(x, axis=0), data))
        data = list(map(torch.FloatTensor, data))
        data = list(map(lambda x: x.squeeze(dim=0), data))  # todo: only this line differs from np_random_dataset in ContextualBandits !!

        return data
