import numpy as np

from util.base_config import BaseConfig


class LinUcbBanditConfig(BaseConfig):
    def set_subconfig_class_hash(self):
        self.subconfig_class_hash = {
            "name": str,
            "alpha": float,
        }


class LinUcbBandit:
    """
    LinUCB algorithm implementation
    """

    def __init__(self, config, num_arms, dim_context):
        """
        Parameters
        ----------
        alpha : number
            LinUCB parameter
        context: string
            'user' or 'both'(item+user): what to use as a feature vector
        """
        self.alpha = round(config.alpha, 2)
        self.name = f"LinUCB_ALPHA={self.alpha}"
        self.detail = "LinUCB (alpha=" + str(self.alpha)

        self.dim_context = dim_context
        self.A = np.array([np.identity(self.dim_context)] * num_arms)
        self.A_inv = np.array([np.identity(self.dim_context)] * num_arms)
        self.b = np.zeros((num_arms, self.dim_context, 1))
        

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

        A_inv = self.A_inv[pool_idx]
        b = self.b[pool_idx]

        n_pool = len(pool_idx)

        user = np.array([user] * n_pool)
        x = np.hstack((user, features[pool_idx]))
        x = x.reshape(n_pool, self.dim_context, 1)

        theta = A_inv @ b

        p = np.transpose(theta, (0, 2, 1)) @ x + self.alpha * np.sqrt(
            np.transpose(x, (0, 2, 1)) @ A_inv @ x
        )
        return np.argmax(p)

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

        a = pool_idx[displayed]  # displayed article's index
        x = np.hstack((user, features[a]))

        x = x.reshape((self.dim_context, 1))

        self.A[a] += x @ x.T
        self.b[a] += reward * x
        self.A_inv[a] = np.linalg.inv(self.A[a])
