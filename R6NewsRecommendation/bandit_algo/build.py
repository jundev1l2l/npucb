from bandit_algo.random import RandomBandit, RandomBanditConfig
from bandit_algo.egreedy import EgreedyBandit, EgreedyBanditConfig
from bandit_algo.thompsonsampling import ThompsonSamplingBandit, ThompsonSamplingBanditConfig
from bandit_algo.ucb import Ucb1Bandit, Ucb1BanditConfig
from bandit_algo.linucb import LinUcbBandit, LinUcbBanditConfig
from bandit_algo.np_ucb import NPUcbBandit, NPUcbBanditConfig
from bandit_algo.np_ts import NPTSBandit, NPTSBanditConfig


BANDIT_ALGO_DICT = {
    "random": RandomBandit,
    "egreedy": EgreedyBandit,
    "ucb1": Ucb1Bandit,
    "thompsonsampling": ThompsonSamplingBandit,
    "linucb": LinUcbBandit,
    "npucb": NPUcbBandit,
    "npts": NPTSBandit,
}

BANDIT_ALGO_CONFIG_DICT = {
    "random": RandomBanditConfig,
    "egreedy": EgreedyBanditConfig,
    "ucb1": Ucb1BanditConfig,
    "thompsonsampling": ThompsonSamplingBanditConfig,
    "linucb": LinUcbBanditConfig,
    "npucb": NPUcbBanditConfig,
    "npts": NPTSBanditConfig,
}


def build_bandit_algo_list(config_list, *args, **kwargs):
    bandit_algo_baseline = build_bandit_algo(config_list[0], *args, **kwargs)
    bandit_algo_eval_list = [build_bandit_algo(cfg, *args, **kwargs) for cfg in config_list[1:]]

    return bandit_algo_baseline, bandit_algo_eval_list


def build_bandit_algo(config, num_arms, dim_context, rank, debug):
    BANDIT_ALGO = BANDIT_ALGO_DICT[config.name]

    if config.name in ["npucb", "npts",]:
        bandit_algo = BANDIT_ALGO(config, num_arms, dim_context, rank, debug)
    else:
        bandit_algo = BANDIT_ALGO(config, num_arms, dim_context)

    return bandit_algo
