import numpy as np

from util.base_config import BaseConfig


class ThompsonSamplingBanditConfig(BaseConfig):
    def set_subconfig_class_hash(self):
        self.subconfig_class_hash = {
            "name": str,
        }


class ThompsonSamplingBandit:
    """
    Thompson sampling algorithm implementation
    """

    def __init__(self, config, num_arms, dim_context):
        self.name = "TS"
        self.num_arms = num_arms
        self.alpha = np.ones(self.num_arms)
        self.beta = np.ones(self.num_arms)

    def choose_arm(self, t, user, pool_idx, features):
        """
        Returns the best arm's index relative to the pool
        Parameters
        ----------
        t : number
            number of trial
        user : array
            user features
        pool_idx : array of indexes
            pool indexes for article identification
        """

        theta = np.random.beta(self.alpha[pool_idx], self.beta[pool_idx])
        return np.argmax(theta)

    def update(self, displayed, reward, user, pool_idx, features):
        """
        Updates algorithm's parameters(matrices) : A,b
        Parameters
        ----------
        displayed : index
            displayed article index relative to the pool
        reward : binary
            user clicked or not
        user : array
            user features
        pool_idx : array of indexes
            pool indexes for article identification
        """

        a = pool_idx[displayed]

        self.alpha[a] += reward
        self.beta[a] += 1 - reward

