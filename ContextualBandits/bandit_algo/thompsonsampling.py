import numpy as np

from util.base_config import BaseConfig


class ThompsonSamplingBanditConfig(BaseConfig):
    def set_subconfig_class_hash(self):
        self.subconfig_class_hash = {
            "name": str,
        }


class ThompsonSamplingBandit:
    """
    Gaussian-Gaussian Thompson Sampling
    """
    def __init__(self, config, num_arms, dim_context, feature_extractor):
        self.name = "TS"
        self.num_arms = num_arms
        self.feature_extractor = feature_extractor
    
    def init(self):
        self.n = np.zeros(self.num_arms)
        self.r = np.zeros(self.num_arms)
        self.loc = np.zeros(self.num_arms)
        self.scale = np.ones(self.num_arms) * 1000

    def add_new_arms(self, num_new_arms):
        self.num_arms += num_new_arms
        self.n, self.r, self.loc = list(map(lambda x: np.concatenate([x, np.zeros(num_new_arms)]), [self.n, self.r, self.loc]))
        self.scale = np.concatenate([self.scale, np.ones(num_new_arms) * 1000])

    def choose_arm(self, context):
        samples = np.random.randn(self.num_arms) * self.scale * self.loc
        return np.argmax(samples)

    def update(self, context, action, reward):
        self.n[action] += 1
        self.r[action] += reward
        self.scale[action] = 1 / (1 / 1000**2 + self.n[action])
        self.loc[action] = self.scale[action] * self.r[action]
