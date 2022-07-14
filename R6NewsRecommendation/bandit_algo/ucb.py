import numpy as np

from util.base_config import BaseConfig


class Ucb1BanditConfig(BaseConfig):
    def set_subconfig_class_hash(self):
        self.subconfig_class_hash = {
            "name": str,
            "alpha": float,
        }


class Ucb1Bandit:
    """
    UCB 1 algorithm implementation
    """
    def __init__(self, config, num_arms, dim_context):
        """
        Parameters
        ----------
        alpha : number
            ucb parameter
        """
        self.num_arms = num_arms
        self.alpha = round(config.alpha, 1)
        self.q = np.zeros(num_arms)
        self.n = np.ones(num_arms)
        self.name = f"Ucb1_ALPHA={self.alpha}"
        self.detail = "UCB1 (alpha=" + str(self.alpha) + ")"


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

        ucbs = self.q[pool_idx] + np.sqrt(self.alpha * np.log(t + 1) / self.n[pool_idx])
        return np.argmax(ucbs)

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

        self.n[a] += 1
        self.q[a] += (reward - self.q[a]) / self.n[a]
