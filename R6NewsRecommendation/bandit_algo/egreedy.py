import numpy as np
from util.base_config import BaseConfig


class EgreedyBanditConfig(BaseConfig):
    def set_subconfig_class_hash(self):
        self.subconfig_class_hash = {
            "name": str,
            "epsilon": float,
        }


class EgreedyBandit:
    """
    Epsilon greedy algorithm implementation
    """

    def __init__(self, config, num_arms, dim_context):
        """
        Parameters
        ----------
        epsilon : number
            Egreedy parameter
        """
        self.num_arms = num_arms
        self.q = np.zeros(num_arms)  # average reward for each arm
        self.n = np.zeros(num_arms)  # number of times each arm was chosen
        self.e = round(config.epsilon, 2)  # epsilon parameter for Egreedy
        self.name = f"Egreedy_EPS={self.e}"
        self.detail = "Egreedy (epsilon=" + str(self.e) + ")"
        if self.e == 1.0:
            self.name = "Random"
            self.detail = "Random"
            
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

        p = np.random.rand()
        if p > self.e:
            return np.argmax(self.q[pool_idx])
        else:
            return np.random.randint(low=0, high=len(pool_idx))

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
