from random import randint
from matplotlib.style import context
import numpy as np
import pandas as pd
import tensorflow as tf

from util.base_config import BaseConfig


class WheelDataSamplerConfig(BaseConfig):
    def set_subconfig_class_hash(self):
        self.subconfig_class_hash = {
            "name": str,
            "file_name": str,
            "dim_context": int,
            "num_arms": int,
            "num_contexts": int,
            "mean_list": list,
            "std_list": list,
            "mean_large": float,
            "std_large": float,
            "delta": float,
            "num_exposed_arms": int,
        }


class WheelDataSampler:
    def __init__(self, config):
        """Samples from Wheel bandit game (see https://arxiv.org/abs/1802.09127).
        Args:
          num_contexts: Number of points to sample, i.e. (context, action, rewards).
          delta: Exploration parameter: high reward in one region if norm above delta.
          mean_v: Mean reward for each action if context norm is below delta.
          std_v: Gaussian reward std for each action if context norm is below delta.
          mu_large: Mean reward for optimal action if context norm is above delta.
          std_large: Reward std for optimal action if context norm is above delta.
        Returns:
          dataset: Sampled matrix with n rows: (context, action, rewards).
          opt_vals: Vector of expected optimal (reward, action) for each context.
        """
        self.file_name = config.file_name
        self.dim_context = config.dim_context
        self.num_arms = config.num_arms
        self.num_contexts = config.num_contexts
        self.mean_list = config.mean_list
        self.std_list = config.std_list
        self.mean_large = config.mean_large
        self.std_large = config.std_large
        self.delta = config.delta
        self.num_exposed_arms = config.num_exposed_arms if hasattr(config, "num_exposed_arms") else self.num_arms

    def sample(self, return_delta=False):
        data = []
        rewards = []
        opt_actions = []
        opt_rewards = []

        if self.num_contexts < 0:
            self.num_contexts = 25000

        # sample uniform contexts in unit ball
        while len(data) < self.num_contexts:  # num_contexts N
          raw_data = np.random.uniform(-1, 1, (int(self.num_contexts / 3), self.dim_context))

          for i in range(raw_data.shape[0]):
            if np.linalg.norm(raw_data[i, :]) <= 1:
              data.append(raw_data[i, :])

        contexts = np.stack(data)[:self.num_contexts, :]  # [N,2]

        # sample rewards
        for i in range(self.num_contexts):
          r = [np.random.normal(self.mean_list[j], self.std_list[j]) for j in range(self.num_arms)]
          if np.linalg.norm(contexts[i, :]) >= self.delta:
            """
            outer part optimal: k 사분면 -> action k
            """
            r_big = np.random.normal(self.mean_large, self.std_large)

            if contexts[i, 1] > 0:
              if contexts[i, 0] > 0:
                r[1] = r_big
                opt_actions.append(1)  # action 1, +x +y
              else:
                r[2] = r_big
                opt_actions.append(2)  # action 2, -x +y
            else:
              if contexts[i, 0] <= 0:
                r[3] = r_big
                opt_actions.append(3)  # action 3, -x -y
              else:
                r[4] = r_big
                opt_actions.append(4)  # action 4, +x -y
          else:
            """
            inner part optimal: action 0 in this setting
            """
            opt_actions.append(np.argmax(self.mean_list))  # action 0, always

          opt_rewards.append(r[opt_actions[-1]])
          rewards.append(r)

        all_rewards = np.stack(rewards, axis=0)  # [N,5]

        opt_rewards = np.array(opt_rewards)  # [N,]
        opt_actions = np.array(opt_actions)  # [N,]

        actions = np.random.randint(low=0, high=self.num_exposed_arms, size=(self.num_contexts,)).reshape(-1, 1)
        rewards = np.expand_dims(all_rewards[range(self.num_contexts), actions.squeeze()], axis=-1)

        if return_delta:    
            return contexts, actions, rewards, all_rewards, opt_actions.reshape(-1, 1), opt_rewards.reshape(-1, 1), np.array([[self.delta,]] * len(contexts))
        else:
            return contexts, actions, rewards, all_rewards, opt_actions.reshape(-1, 1), opt_rewards.reshape(-1, 1)
