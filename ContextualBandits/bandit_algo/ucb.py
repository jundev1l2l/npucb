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
    def __init__(self, config, num_arms, dim_context, feature_extractor):
        """
        Parameters
        ----------
        alpha : number
            ucb parameter
        """
        self.num_arms = num_arms
        self.alpha = round(config.alpha, 2)
        self.name = f"Ucb1_ALPHA={self.alpha}"
        self.detail = "UCB1 (alpha=" + str(self.alpha) + ")"
        self.feature_extractor = feature_extractor

    def init(self):
        self.q = np.zeros(self.num_arms)  # average reward for each arm
        self.n = np.zeros(self.num_arms)  # number of times each arm was chosen
        self.t = 0  # number of total trials

    def add_new_arms(self, num_new_arms):
        self.num_arms += num_new_arms
        self.q, self.n = list(map(lambda x: np.concatenate([x, np.zeros(num_new_arms)]), [self.q, self.n]))

    def choose_arm(self, context):
        for a in range(self.num_arms):
            if self.n[a] == 0:
                return a
        ucbs = self.q + self.alpha * np.sqrt(np.log(self.t) / self.n)

        return np.argmax(ucbs)

    def update(self, context, action, reward):
        self.n[action] += 1
        self.t += 1
        self.q[action] = self.q[action] * (self.n[action] - 1) / self.n[action] + (reward - self.q[action]) / self.n[action]
