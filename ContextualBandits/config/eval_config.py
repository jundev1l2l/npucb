from util.base_config import BaseConfig
from engine import ENGINE_CONFIG_DICT
from data_sampler import DATA_SAMPLER_CONFIG_DICT
from bandit_env import BANDIT_ENV_CONFIG_DICT
from bandit_algo import BANDIT_ALGO_CONFIG_DICT
from feature_extractor import FEATURE_EXTRACTOR_CONFIG_DICT
from evaluator import EVALUATOR_CONFIG_DICT


class EvalConfig(BaseConfig):
    def set_subconfig_class_hash(self):
        self.subconfig_class_hash = {
            "project": str,
            "task": str,
            "default_config": str,
            "exp_config": str,
            "result": str,
            "seed": int,
            "debug": bool,
            "wandb_mode": str,
            "engine": ENGINE_CONFIG_DICT,
            "data_sampler": DATA_SAMPLER_CONFIG_DICT,
            "bandit_env": BANDIT_ENV_CONFIG_DICT,
            "bandit_algo_list": BANDIT_ALGO_CONFIG_DICT,
            "feature_extractor": FEATURE_EXTRACTOR_CONFIG_DICT,
            "evaluator": EVALUATOR_CONFIG_DICT,
        }
