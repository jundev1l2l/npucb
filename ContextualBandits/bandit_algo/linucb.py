import numpy as np

from util.base_config import BaseConfig


class LinUcbBanditConfig(BaseConfig):
    def set_subconfig_class_hash(self):
        self.subconfig_class_hash = {
            "name": str,
            "alpha": float,
        }


class LinUcbBandit:
    def __init__(self, config, num_arms, dim_context, feature_extractor):
        self.alpha = round(config.alpha, 2)
        self.num_arms = num_arms
        self.dim_context = dim_context
        self.name = f"LinUCB_ALPHA={self.alpha}"
        self.detail = "LinUCB (alpha=" + str(self.alpha)
        self.feature_extractor = feature_extractor

    def init(self):
        self.A = np.array([np.identity(self.dim_context)] * self.num_arms)
        self.A_inv = np.array([np.identity(self.dim_context)] * self.num_arms)
        self.b = np.zeros((self.num_arms, self.dim_context, 1))

    def add_new_arms(self, num_new_arms):
        self.num_arms += num_new_arms
        self.A, self.A_inv = list(map(lambda x: np.concatenate([x, np.array([np.identity(self.dim_context)] * num_new_arms)], axis=0), [self.A, self.A_inv]))
        self.b = np.concatenate([self.b, np.zeros([num_new_arms, self.dim_context, 1])], axis=0)

    def choose_arm(self, context):
        x = np.array([context] * self.num_arms)

        if self.feature_extractor is not None:
            x = self.feature_extractor.encode(x)
            
        x = x.reshape(self.num_arms, self.dim_context, 1)

        theta = self.A_inv @ self.b

        p = np.transpose(theta, (0, 2, 1)) @ x + self.alpha * np.sqrt(
            np.transpose(x, (0, 2, 1)) @ self.A_inv @ x
        )        
        return np.argmax(p)

    def update(self, context, action, reward):
        x = context.reshape((self.dim_context, 1))
        self.A[action] += x @ x.T
        self.b[action] += reward * x
        self.A_inv[action] = np.linalg.inv(self.A[action])
