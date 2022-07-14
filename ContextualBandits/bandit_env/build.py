from bandit_env.base_bandit_env import BaseBanditEnv, BaseBanditEnvConfig


BANDIT_ENV_DICT = {
    "BaseBanditEnv": BaseBanditEnv
}


BANDIT_ENV_CONFIG_DICT = {
    "BaseBanditEnv": BaseBanditEnvConfig
}


def build_bandit_env(config, data_sampler, *args, **kwargs):
    if isinstance(data_sampler, list):

        bandit_env_list = []
        for d in data_sampler:
            bandit_env_list.append(build_bandit_env(config, d, *args, **kwargs))
        
        return bandit_env_list
    else:
        BANDIT_ENV = BANDIT_ENV_DICT[config.name]
        
        bandit_env = BANDIT_ENV(config, *args, **kwargs, data_sampler=data_sampler)

        return bandit_env
    