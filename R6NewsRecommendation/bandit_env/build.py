from bandit_env.base_bandit_env import BaseBanditEnv, BaseBanditEnvConfig
from bandit_env.r6_bandit_env import R6BanditEnv, R6BanditEnvConfig


BANDIT_ENV_DICT = {
    "BaseBanditEnv": BaseBanditEnv,
    "R6BanditEnv": R6BanditEnv,
}


BANDIT_ENV_CONFIG_DICT = {
    "BaseBanditEnv": BaseBanditEnvConfig,
    "R6BanditEnv": R6BanditEnvConfig,
}


def build_bandit_env(config, *args, **kwargs):

    BANDIT_ENV = BANDIT_ENV_DICT[config.name]
    
    bandit_env = BANDIT_ENV(config, *args, **kwargs)

    return bandit_env
    