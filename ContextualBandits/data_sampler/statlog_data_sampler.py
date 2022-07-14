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
from matplotlib.style import context
import numpy as np
import pandas as pd
import tensorflow as tf

from util.base_config import BaseConfig
from data_sampler.util import remove_underrepresented_classes, classification_to_bandit_problem


class StatlogDataSamplerConfig(BaseConfig):
    def set_subconfig_class_hash(self):
        self.subconfig_class_hash = {
            "name": str,
            "file_name": str,
            "dim_context": int,
            "num_arms": int,
            "num_contexts": int,
            "shuffle_rows": bool,
            "remove_underrepresented": bool,
            "num_exposed_arms": int,
        }


class StatlogDataSampler:
    def __init__(self, config):
        """Returns bandit problem dataset based on the UCI statlog data.

        Args:
        file_name: Route of file containing the Statlog dataset.
        num_contexts: Number of contexts to sample.
        shuffle_rows: If True, rows from original dataset are shuffled.
        remove_underrepresented: If True, removes arms with very few rewards.

        Returns:
        dataset: Sampled matrix with rows: (context, action rewards).
        opt_vals: Vector of deterministic optimal (reward, action) for each context.

        https://archive.ics.uci.edu/ml/datasets/Statlog+(Shuttle)
        """
        self.file_name = config.file_name
        self.dim_context = config.dim_context
        self.num_arms = config.num_arms
        self.num_contexts = config.num_contexts
        self.shuffle_rows = config.shuffle_rows  # True
        self.remove_underrepresented = config.remove_underrepresented  # False
        self.num_exposed_arms = config.num_exposed_arms if hasattr(config, "num_exposed_arms") else self.num_arms

    def sample(self):
        with tf.compat.v1.gfile.Open(self.file_name, 'r') as f:
            data = np.loadtxt(f)

        # Shuffle data
        if self.shuffle_rows:
            np.random.shuffle(data)

        # Last column is label, rest are features
        contexts = data[:, :-1]
        labels = data[:, -1].astype(int) - 1  # convert to 0 based index

        if self.remove_underrepresented:
            contexts, labels = remove_underrepresented_classes(contexts, labels)

        if self.num_contexts == - 1:
            self.num_contexts = contexts.shape[0]

        contexts = contexts[:self.num_contexts, :]
        labels = labels[:self.num_contexts,]

        contexts, all_rewards, (opt_rewards, opt_actions) = classification_to_bandit_problem(contexts, labels, self.num_arms)

        actions = np.random.randint(low=0, high=self.num_exposed_arms, size=(self.num_contexts,)).reshape(-1, 1)
        rewards = np.expand_dims(all_rewards[range(self.num_contexts), actions.squeeze()], axis=-1)

        return contexts, actions, rewards, all_rewards, opt_actions.reshape(-1, 1), opt_rewards.reshape(-1, 1)
