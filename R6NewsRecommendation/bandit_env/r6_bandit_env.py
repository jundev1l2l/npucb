import numpy as np

from util.base_config import BaseConfig


class R6BanditEnvConfig(BaseConfig):
    def set_subconfig_class_hash(self):
        self.subconfig_class_hash = {
            "name": str,
            "dim_context": int,
        }


class R6BanditEnv:
    def __init__(self, config, data_sampler, logger):
        self.dim_context = config.dim_context
        self.data_sampler = data_sampler
        self.events, self.features, self.num_arms = self.data_sampler.sample()
        self.num_contexts = len(self.events)
        self.logger = logger

    def init(self, num_steps):
        self.num_steps = self.num_contexts if num_steps < 0 else num_steps
        self.total_steps = 0
        self.reset()
        return self.get_return()

    def reset(self):
        self.curr_step = 0
        self.order = np.random.permutation(self.num_contexts)
    
    def step(self):
        self.curr_step += 1
        self.total_steps += 1
        if self.curr_step >= self.num_contexts:
            self.reset()
        return self.get_return(), (self.curr_step >= self.num_steps), self.curr_step

    def get_return(self):
        event = self.events[self.order[self.curr_step]]
        displayed = event[0]
        reward = event[1]
        user = event[2]
        pool_idx = event[3]
        features = self.features
        return displayed, reward, user, pool_idx, features
