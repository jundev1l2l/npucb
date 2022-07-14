import numpy as np

from util.base_config import BaseConfig


class BaseBanditEnvConfig(BaseConfig):
    def set_subconfig_class_hash(self):
        self.subconfig_class_hash = {
            "name": str,
        }


class BaseBanditEnv:
    def __init__(self, config, data_sampler, logger):
        self.data_sampler = data_sampler
        self.num_arms = data_sampler.num_arms
        self.dim_context = data_sampler.dim_context
        self.logger = logger

    def init(self, num_steps, total_steps=None):
        self.data = self.data_sampler.sample()
        self.contexts, self.actions, self.rewards, self.all_rewards, self.opt_actions, self.opt_rewards = self.data
        self.num_contexts = len(self.rewards)
        self.num_steps = num_steps
        self.total_steps = 0 if total_steps is None else total_steps
        self.reset()
        context = self.get_context()
        return context

    def reset(self):
        self.curr_step = 0
        self.order = np.random.permutation(self.num_contexts)
    
    def get_context(self):
        return self.contexts[self.order[self.curr_step]]

    def get_reward(self, action):
        return self.all_rewards[self.order[self.curr_step]][action]

    def get_opt_reward(self):
        return self.opt_rewards[self.order[self.curr_step]]
    
    def get_opt_action(self):
        return self.opt_actions[self.order[self.curr_step]]

    def step(self, action):
        if isinstance(action, list):
            reward = [self.get_reward(a) for a in action]
        else:
            reward = self.get_reward(action)
        opt_reward = self.get_opt_reward()
        
        self.curr_step += 1
        self.total_steps += 1
        if self.curr_step >= self.num_contexts:
            self.reset()
        
        context = self.get_context()
        terminated = (self.total_steps >= self.num_steps)

        return context, reward, opt_reward, terminated, self.total_steps
