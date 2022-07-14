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

    def __init__(self, config, num_arms, dim_context, feature_extractor):
        """
        Parameters
        ----------
        epsilon : number
            Egreedy parameter
        """
        self.num_arms = num_arms
        self.e = round(config.epsilon, 2)  # epsilon parameter for Egreedy
        self.name = f"Egreedy_EPS={self.e}"
        self.detail = "Egreedy (epsilon=" + str(self.e) + ")"
        if self.e == 1.0:
            self.name = "Random"
            self.detail = "Random"
        self.feature_extractor = feature_extractor

    def init(self):
        self.average_reward = np.zeros(self.num_arms)  # average reward for each arm
        self.num_steps = np.zeros(self.num_arms)  # number of times each arm was chosen

    def add_new_arms(self, num_new_arms):
        self.num_arms += num_new_arms
        self.average_reward, self.num_steps = list(map(lambda x: np.concatenate([x, np.zeros(num_new_arms)]), [self.average_reward, self.num_steps]))
            
    def choose_arm(self, context):
        """
        Returns the best arm's index relative to the pool
        Parameters
        ----------
        context: array of scalars
            context vector from bandit environment
        pool_idx : array of indexes
            pool indexes for article identification
        """
        p = np.random.rand()
        if p > self.e:
            return np.argmax(self.average_reward)
        else:
            return np.random.randint(low=0, high=self.num_arms)

    def update(self, context, action, reward):
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
        self.num_steps[action] += 1
        self.average_reward[action] += (reward - self.average_reward[action]) / self.num_steps[action]
