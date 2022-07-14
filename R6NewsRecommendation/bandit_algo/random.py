from util.base_config import BaseConfig
from bandit_algo.egreedy import EgreedyBandit


class RandomBanditConfig(BaseConfig):
    def set_subconfig_class_hash(self):
        self.subconfig_class_hash = {
            "name": str,
        }


class RandomBandit(EgreedyBandit):
    def __init__(self, config, *args, **kwargs):
        config.epsilon = 1.0
        super(RandomBandit, self).__init__(config, *args, **kwargs)
        self.name = "Random"
