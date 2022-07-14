import torch
import numpy as np

from util.base_config import BaseConfig
from dataset.torch_dataset import TorchDataset


class BaseDatasetConfig(BaseConfig):
    def set_subconfig_class_hash(self):
        self.subconfig_class_hash = {
            "name": str,
        }


class BaseDataset(TorchDataset):
    def __init__(self, config, data_sampler, debug):
        self.name = config.name

        self.data_sampler = data_sampler
        self.debug = debug

        self.set_data()

    def set_data(self):
        events, features, num_arms = self.data_sampler.sample()
        self.events = events[:int(len(events)/100)] if self.debug else events
        self.num_events = len(self.events)
        self.features = features
        self.dim_context = self.features.shape[1] * 2

    def __len__(self):
        return self.num_events

    def __getitem__(self, idx):
        data = self.get_xy(idx=idx)
        data = list(map(torch.FloatTensor, data))
        return data

    def get_xy(self, idx):
        event = self.events[idx]
        user = event[2]
        displayed = event[0]
        pool_idx = event[3]

        displayed_idx = pool_idx[displayed]
        item = self.features[displayed_idx]
        x = np.concatenate([np.array(user), np.array(item)], axis=-1).reshape(1, self.dim_context)

        reward = event[1]
        y = np.array(reward).reshape(1, 1)

        return x, y
