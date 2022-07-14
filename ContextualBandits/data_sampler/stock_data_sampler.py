# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Functions to create bandit problems from datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from random import randint
import numpy as np
import pandas as pd
import tensorflow as tf

from util.base_config import BaseConfig


class StockDataSamplerConfig(BaseConfig):
    def set_subconfig_class_hash(self):
        self.subconfig_class_hash = {
            "name": str,
            "file_name": str,
            "dim_context": int,
            "num_arms": int,
            "num_contexts": int,
            "sigma": float,
            "shuffle_rows": bool,
            "num_exposed_arms": int,
        }


class StockDataSampler:
    def __init__(self, config):
        """Samples linear bandit game from stock prices dataset.

        Args:
        file_name: Route of file containing the stock prices dataset.
        context_dim: Context dimension (i.e. vector with the price of each stock).
        num_arms: Number of actions (different linear portfolio strategies).
        self.num_contexts: Number of contexts to sample.
        sigma: Vector with additive noise levels for each action.
        shuffle_rows: If True, rows from original dataset are shuffled.

        Returns:
        dataset: Sampled matrix with rows: (context, reward_1, ..., reward_k).
        opt_vals: Vector of expected optimal (reward, action) for each context.
        """
        self.file_name = config.file_name
        self.dim_context = config.dim_context
        self.num_arms = config.num_arms
        self.num_contexts = config.num_contexts
        self.sigma = config.sigma
        self.shuffle_rows = config.shuffle_rows
        self.num_exposed_arms = config.num_exposed_arms if hasattr(config, "num_exposed_arms") else self.num_arms

    def sample(self):
        with tf.compat.v1.gfile.Open(self.file_name, 'r') as f:
            contexts = np.loadtxt(f, skiprows=1)
        if self.num_contexts == - 1:
            self.num_contexts = len(contexts)

        if self.shuffle_rows:
            np.random.shuffle(contexts)
        contexts = contexts[:self.num_contexts, :]

        betas = np.random.uniform(-1, 1, (self.dim_context, self.num_arms))
        betas /= np.linalg.norm(betas, axis=0)

        mean_rewards = np.dot(contexts, betas)
        noise = np.random.normal(scale=self.sigma, size=mean_rewards.shape)
        all_rewards = mean_rewards + noise

        actions = np.random.randint(low=0, high=self.num_exposed_arms, size=(self.num_contexts,)).reshape(-1, 1)
        rewards = np.expand_dims(all_rewards[range(self.num_contexts), actions.squeeze()], axis=-1)

        opt_actions = np.argmax(mean_rewards, axis=1).reshape(-1, 1)
        opt_rewards = np.array([mean_rewards[i, a] for i, a in enumerate(opt_actions)], dtype=float).reshape(-1, 1)

        return contexts, actions, rewards, all_rewards, opt_actions, opt_rewards
