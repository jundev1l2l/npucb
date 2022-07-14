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

import numpy as np
import pandas as pd

from util.base_config import BaseConfig
from data_sampler.util import one_hot


class MushroomDataSamplerConfig(BaseConfig):
    def set_subconfig_class_hash(self):
        self.subconfig_class_hash = {
            "name": str,
            "file_name": str,
            "dim_context": int,
            "num_arms": int,
            "num_contexts": int,
            "r_noeat": int,
            "r_eat_safe": int,
            "r_eat_poison_bad": int,
            "r_eat_poison_good": int,
            "prob_poison_bad": float,
            "num_exposed_arms": int,
        }


class MushroomDataSampler:
    """Samples bandit game from Mushroom UCI Dataset.

    Args:
      file_name: Route of file containing the original Mushroom UCI dataset.
      num_contexts: Number of points to sample, i.e. (context, action rewards).
      r_noeat: Reward for not eating a mushroom.
      r_eat_safe: Reward for eating a non-poisonous mushroom.
      r_eat_poison_bad: Reward for eating a poisonous mushroom if harmed.
      r_eat_poison_good: Reward for eating a poisonous mushroom if not harmed.
      prob_poison_bad: Probability of being harmed by eating a poisonous mushroom.

    Returns:
      dataset: Sampled matrix with n rows: (context, eat_reward, no_eat_reward).
      opt_vals: Vector of expected optimal (reward, action) for each context.

    We assume r_eat_safe > r_noeat, and r_eat_poison_good > r_eat_poison_bad.
    """
    def __init__(self, config):
        self.file_name = config.file_name
        self.dim_context = config.dim_context
        self.num_arms = config.num_arms
        self.num_contexts = config.num_contexts
        self.r_noeat = config.r_noeat
        self.r_eat_safe = config.r_eat_safe
        self.r_eat_poison_bad = config.r_eat_poison_bad
        self.r_eat_poison_good = config.r_eat_poison_good
        self.prob_poison_bad = config.prob_poison_bad
        self.num_exposed_arms = config.num_exposed_arms if hasattr(config, "num_exposed_arms") else self.num_arms

    def sample(self):
        # first two cols of df encode whether mushroom is edible or poisonous
        df = pd.read_csv(self.file_name, header=None)
        df = one_hot(df, df.columns)
        if self.num_contexts == -1:
            self.num_contexts = len(df)
        ind = np.random.choice(range(df.shape[0]), self.num_contexts, replace=True)
        print(df.size)

        contexts = df.iloc[ind, 2:]
        no_eat_reward = self.r_noeat * np.ones((self.num_contexts, 1))
        random_poison = np.random.choice(
            [self.r_eat_poison_bad, self.r_eat_poison_good],
            p=[self.prob_poison_bad, 1 - self.prob_poison_bad],
            size=self.num_contexts)
        eat_reward = self.r_eat_safe * df.iloc[ind, 0]
        eat_reward += np.multiply(random_poison, df.iloc[ind, 1])
        eat_reward = eat_reward.values.reshape((self.num_contexts, 1))
        all_rewards = np.concatenate([no_eat_reward, eat_reward], axis=1)

        # simulate random actions and corresponding rewards
        contexts = np.array(contexts, dtype=float)
        actions = np.random.randint(low=0, high=self.num_exposed_arms, size=(self.num_contexts,)).reshape(-1, 1)
        rewards = np.expand_dims(all_rewards[range(self.num_contexts), actions.squeeze()], axis=-1)

        # compute optimal expected reward and optimal actions
        exp_eat_poison_reward = self.r_eat_poison_bad * self.prob_poison_bad + self.r_eat_poison_good * (1 - self.prob_poison_bad)
        opt_exp_reward = self.r_eat_safe * df.iloc[ind, 0] + max(
            self.r_noeat, exp_eat_poison_reward) * df.iloc[ind, 1]
        if self.r_noeat > exp_eat_poison_reward:
            # actions: no eat = 0 ; eat = 1
            opt_actions = df.iloc[ind, 0]  # indicator of edible
        else:
            # should always eat (higher expected reward)
            opt_actions = np.ones((self.num_contexts, 1))
        opt_actions = np.array(opt_actions.values, dtype=int).reshape(-1, 1)
        opt_rewards = np.array(opt_exp_reward.values, dtype=float).reshape(-1, 1)

        return contexts, actions, rewards, all_rewards, opt_actions, opt_rewards
