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


class JesterDataSamplerConfig(BaseConfig):
    def set_subconfig_class_hash(self):
        self.subconfig_class_hash = {
            "name": str,
            "file_name": str,
            "dim_context": int,
            "num_arms": int,
            "num_contexts": int,
            "shuffle_rows": bool,
            "shuffle_cols": bool,
            "num_exposed_arms": int,
        }


class JesterDataSampler:
    def __init__(self, config):
        """Samples bandit game from (user, joke) dense subset of Jester dataset.

        Args:
        file_name: Route of file containing the modified Jester dataset.
        context_dim: Context dimension (i.e. vector with some ratings from a user).
        num_arms: Number of actions (number of joke ratings to predict).
        num_contexts: Number of contexts to sample.
        shuffle_rows: If True, rows from original dataset are shuffled.
        shuffle_cols: Whether or not context/action jokes are randomly shuffled.

        Returns:
        dataset: Sampled matrix with rows: (context, rating_1, ..., rating_k).
        opt_vals: Vector of deterministic optimal (reward, action) for each context.
        """
        self.file_name = config.file_name
        self.dim_context = config.dim_context
        self.num_arms = config.num_arms
        self.num_contexts = config.num_contexts
        self.shuffle_rows = config.shuffle_rows
        self.shuffle_cols = config.shuffle_cols
        self.num_exposed_arms = config.num_exposed_arms if hasattr(config, "num_exposed_arms") else self.num_arms

    def sample(self):
        with tf.compat.v1.gfile.Open(self.file_name, 'rb') as f:
            dataset = np.load(f)

        if self.shuffle_cols:
            dataset = dataset[:, np.random.permutation(dataset.shape[1])]
        if self.shuffle_rows:
            np.random.shuffle(dataset)
        if self.num_contexts == - 1:
            self.num_contexts = len(dataset)
            
        dataset = dataset[:self.num_contexts, :]

        assert self.dim_context + \
            self.num_arms == dataset.shape[1], 'Wrong data dimensions.'

        contexts = dataset[:, :self.dim_context]
        all_rewards = dataset[:, self.dim_context:]
        actions = np.random.randint(low=0, high=self.num_exposed_arms, size=(self.num_contexts,)).reshape(-1, 1)
        rewards = np.expand_dims(all_rewards[range(self.num_contexts), actions.squeeze()], axis=-1)

        opt_actions = np.argmax(dataset[:, self.dim_context:], axis=1)
        opt_rewards = np.array([dataset[i, self.dim_context + a]
                                for i, a in enumerate(opt_actions)])
        
        return contexts, actions, rewards, all_rewards, opt_actions.reshape(-1, 1), opt_rewards.reshape(-1, 1)
