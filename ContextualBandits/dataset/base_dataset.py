import torch
import numpy as np

from util.base_config import BaseConfig
from dataset.torch_dataset import TorchDataset


class BaseDatasetConfig(BaseConfig):
    def set_subconfig_class_hash(self):
        self.subconfig_class_hash = {
            "name": str,
            "mixed_per_batch": bool,
        }


class BaseDataset(TorchDataset):
    def __init__(self, config, data_sampler, debug):
        self.name = config.name

        self.data_sampler = data_sampler
        self.mixed_data = True if isinstance(self.data_sampler, list) else False
        self.mixed_data_idx = np.random.randint(len(self.data_sampler)) if self.mixed_data else -1
        self.mixed_per_batch = config.mixed_per_batch if hasattr(config, "mixed_per_batch") else False
        self.debug = debug

        self.set_data()

    def set_data(self):
        if self.mixed_data:
            contexts, actions, rewards, all_rewards, opt_actions, opt_rewards, deltas = list(zip(*[data_sampler.sample(return_delta=True) for data_sampler in self.data_sampler]))
        else:
            contexts, actions, rewards, all_rewards, opt_actions, opt_rewards, deltas = self.data_sampler.sample(return_delta=True)
        
        self.contexts = contexts
        self.actions = actions
        self.rewards = rewards
        self.deltas = deltas
        
        self.all_rewards = all_rewards
        self.opt_actions = opt_actions
        self.opt_rewards = opt_rewards
        
        if self.mixed_data:
            self.events = [list(zip(contexts, actions, rewards, all_rewards, opt_actions, opt_rewards, deltas)) for contexts, actions, rewards, all_rewards, opt_actions, opt_rewards, deltas in list(zip(self.contexts, self.actions, self.rewards, self.all_rewards, self.opt_actions, self.opt_rewards, self.deltas))]
            self.n_events = len(self.events[0])
        else:
            self.events = list(zip(contexts, actions, rewards, all_rewards, opt_actions, opt_rewards, deltas))
            self.n_events = len(self.events)

    def __len__(self):
        return self.n_events

    def __getitem__(self, idx):
        data = self.get_xy(idx)
        data = list(map(torch.FloatTensor, data))
        return data

    def get_xy(self, idx):
        if self.mixed_data_idx < 0:
            event = self.events[idx]
        else:
            event = self.events[self.mixed_data_idx][idx]
    
        context = event[0]
        action = event[1]
        reward = event[2]
        delta = event[6]

        x = np.concatenate([context, action], axis=-1).reshape(1, -1)
        y = np.array(reward).reshape(1, 1)

        return x, y, delta
