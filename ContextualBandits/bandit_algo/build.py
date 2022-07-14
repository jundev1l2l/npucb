from bandit_algo.random import RandomBandit, RandomBanditConfig
from bandit_algo.egreedy import EgreedyBandit, EgreedyBanditConfig
from bandit_algo.thompsonsampling import ThompsonSamplingBandit, ThompsonSamplingBanditConfig
from bandit_algo.ucb import Ucb1Bandit, Ucb1BanditConfig
from bandit_algo.linucb import LinUcbBandit, LinUcbBanditConfig
from bandit_algo.np_ucb import NPUcbBandit, NPUcbBanditConfig
from bandit_algo.np_ts import NPTSBandit, NPTSBanditConfig
from bandit_algo.np_ucb_online import NPUcbOnlineBandit, NPUcbOnlineBanditConfig
from bandit_algo.np_ts_online import NPTSOnlineBandit, NPTSOnlineBanditConfig
from bandit_algo.neural_ucb import NeuralUcbBandit, NeuralUcbBanditConfig
from bandit_algo.gp_ucb import GPUcbBandit, GPUcbBanditConfig


BANDIT_ALGO_DICT = {
    "random": RandomBandit,
    "egreedy": EgreedyBandit,
    "ucb1": Ucb1Bandit,
    "thompsonsampling": ThompsonSamplingBandit,
    "linucb": LinUcbBandit,
    "npucb": NPUcbBandit,
    "npts": NPTSBandit,
    "npucb_online": NPUcbOnlineBandit,
    "npts_online": NPTSOnlineBandit,
    "neuralucb": NeuralUcbBandit,
    "gpucb": GPUcbBandit,
}

BANDIT_ALGO_CONFIG_DICT = {
    "random": RandomBanditConfig,
    "egreedy": EgreedyBanditConfig,
    "ucb1": Ucb1BanditConfig,
    "thompsonsampling": ThompsonSamplingBanditConfig,
    "linucb": LinUcbBanditConfig,
    "npucb": NPUcbBanditConfig,
    "npts": NPTSBanditConfig,
    "npucb_online": NPUcbOnlineBanditConfig,
    "npts_online": NPTSOnlineBanditConfig,
    "neuralucb": NeuralUcbBanditConfig,
    "gpucb": GPUcbBanditConfig,
}


def build_bandit_algo_list(config_list, *args, **kwargs):
    bandit_algo_baseline = build_bandit_algo(config_list[0], *args, **kwargs)
    bandit_algo_eval_list = [build_bandit_algo(cfg, *args, **kwargs) for cfg in config_list[1:]]

    return bandit_algo_baseline, bandit_algo_eval_list


def build_bandit_algo(config, num_arms, dim_context, feature_extractor, rank, debug):
    BANDIT_ALGO = BANDIT_ALGO_DICT[config.name]

    if config.name in ["npucb", "npts", "npucb_online", "npts_online", "gpucb",]:
        bandit_algo = BANDIT_ALGO(config, num_arms, dim_context, feature_extractor, rank, debug)
    else:
        bandit_algo = BANDIT_ALGO(config, num_arms, dim_context, feature_extractor)

    return bandit_algo
