import os
import random
import matplotlib.pyplot as plt
import numpy as np
import wandb
from time import time
from datetime import timedelta
from tqdm import tqdm

from util.log import MemLogger, TimeLogger
from util.base_config import BaseConfig


class BaseEvaluatorConfig(BaseConfig):
    def set_subconfig_class_hash(self):
        self.subconfig_class_hash = {
            "name": str,
            "num_exp": int,
            "num_steps": int,
        }


class BaseEvaluator:
    def __init__(self, config, bandit_algo_baseline, bandit_algo_eval_list, bandit_env, logger, save_dir, rank, debug):
        self.name = config.name
        self.num_exp = config.num_exp
        self.num_steps = config.num_steps

        self.baseline = bandit_algo_baseline
        self.baseline_name = self.baseline.name

        self.eval_list = bandit_algo_eval_list
        self.eval_name_list = [algo.name for algo in self.eval_list]
        
        self.algo_list = [self.baseline, *self.eval_list]
        self.algo_name_list = [self.baseline_name, *self.eval_name_list]
        self.num_algos = len(self.algo_list)
        
        self.bandit_env = bandit_env
        self.num_arms = self.bandit_env.num_arms

        self.logger = logger
        self.mlogger = MemLogger(logger)
        self.tlogger = TimeLogger(logger)
        self.save_dir = save_dir
        self.rank = rank
        self.debug = debug
        if self.debug:
            self.num_steps = 10

    def eval(self):
        self.summary_metrics = {
            "AvgReward": np.empty([self.num_algos, self.num_exp]),
            "CummReward": np.empty([self.num_algos, self.num_exp]),
            "AvgRegret": np.empty([self.num_algos, self.num_exp]),
            "CummRegret": np.empty([self.num_algos, self.num_exp]),
            "RelAvgReward": np.empty([self.num_algos, self.num_exp]),
            "RelCummReward": np.empty([self.num_algos, self.num_exp]),
            "RelAvgRegret": np.empty([self.num_algos, self.num_exp]),
            "RelCummRegret": np.empty([self.num_algos, self.num_exp]),
        }
        for run_idx in range(self.num_exp):
            self.init_metrics()
            for algo in self.algo_list:
                algo.init()
            self.run_idx = run_idx
            self.run_bandit()
        self.log_summary() if self.rank <=0 else None

    def init_metrics(self):
        self.metrics = {
            "AvgReward": np.zeros([self.num_algos,]),
            "CummReward": np.zeros([self.num_algos,]),
            "AvgRegret": np.zeros([self.num_algos,]),
            "CummRegret": np.zeros([self.num_algos,]),
            "RelAvgReward": np.zeros([self.num_algos,]),
            "RelCummReward": np.zeros([self.num_algos,]),
            "RelAvgRegret": np.zeros([self.num_algos,]),
            "RelCummRegret": np.zeros([self.num_algos,]),
        }
        self.action_freq = np.zeros([self.num_algos, self.num_arms])

    def run_bandit(self):
        context = self.bandit_env.init(self.num_steps)
        terminated = False
        self.curr_step = 0
        
        self.iter_bar = tqdm(range(self.num_steps))
        for _ in self.iter_bar:
            action_list = [algo.choose_arm(context) for algo in self.algo_list]

            context, reward_list, opt_reward, terminated, self.curr_step = self.bandit_env.step(action_list)
            
            for algo_idx, algo in enumerate(self.algo_list):
                algo.update(context, action_list[algo_idx], reward_list[algo_idx])
                self.update_metrics(algo_idx, action_list[algo_idx], reward_list[algo_idx], opt_reward)
                self.log(algo_idx, action_list[algo_idx], reward_list[algo_idx], opt_reward) if self.rank <=0 else None
            if terminated:
                self.summary()
                break
    
    def update_metrics(self, algo_idx, action, reward, opt_reward):
        self.update_action_freq(algo_idx, action)
        self.update_reward(algo_idx, reward)
        self.update_regret(algo_idx, opt_reward - reward)
        
    def update_action_freq(self, algo_idx, action):
        for a in range(self.num_arms):
            self.action_freq[algo_idx][a] *= ((self.curr_step - 1) / self.curr_step)
        self.action_freq[algo_idx][action] += + 1 / self.curr_step

    def update_reward(self, algo_idx, reward):
        self.metrics["CummReward"][algo_idx] += reward
        self.metrics["AvgReward"][algo_idx] = self.metrics["CummReward"][algo_idx] / self.curr_step
        self.metrics["RelCummReward"][algo_idx] = self.metrics["CummReward"][algo_idx] / (self.metrics["CummReward"][0] + 1e-3)
        self.metrics["RelAvgReward"][algo_idx] = self.metrics["AvgReward"][algo_idx] / (self.metrics["AvgReward"][0] + 1e-3)

    def update_regret(self, algo_idx, regret):
        self.metrics["CummRegret"][algo_idx] += regret
        self.metrics["AvgRegret"][algo_idx] = self.metrics["CummRegret"][algo_idx] / self.curr_step
        self.metrics["RelCummRegret"][algo_idx] = self.metrics["CummRegret"][algo_idx] / (self.metrics["CummRegret"][0] + 1e-3)
        self.metrics["RelAvgRegret"][algo_idx] = self.metrics["AvgRegret"][algo_idx] / (self.metrics["AvgRegret"][0] + 1e-3)

    def log(self, algo_idx, action, reward, opt_reward):
        wandb.log(data={f"Step/Run{self.run_idx}": self.curr_step}, step=self.curr_step)
        self.log_single_data(algo_idx, action, reward, opt_reward - reward)
        self.log_optimal_data(opt_reward) if algo_idx == 0 else None
        self.log_metrics(algo_idx)
    
    def log_single_data(self, algo_idx, action, reward, regret):
        algo_name = self.algo_name_list[algo_idx]
        wandb.log(data={
            f"{algo_name}/Action/Run{self.run_idx}": action,
            f"{algo_name}/Reward/Run{self.run_idx}": reward,
            f"{algo_name}/Regret/Run{self.run_idx}": regret,
        }, step=self.curr_step)

    def log_optimal_data(self, opt_reward):
        wandb.log(data={
            f"Optimal/Reward/Run{self.run_idx}": opt_reward,
        }, step=self.curr_step)

    def log_metrics(self, algo_idx):
        algo_name = self.algo_name_list[algo_idx]
        action_freq_logs = dict([[f"{algo_name}/P({action})/Run{self.run_idx}", self.action_freq[algo_idx][action]] for action in range(self.num_arms)])
        wandb.log(data=action_freq_logs, step=self.curr_step)
        for key, value in self.metrics.items():
            wandb.log(data={f"{algo_name}/{key}/Run{self.run_idx}": value[algo_idx]}, step=self.curr_step)

    def summary(self):
        for algo_idx in range(self.num_algos):
            for key in self.summary_metrics.keys():
                self.summary_metrics[key][algo_idx, self.run_idx] = self.metrics[key][algo_idx]

    def log_summary(self):
        for key, value in self.summary_metrics.items():
            for algo_idx in range(self.num_algos):
                algo_name = self.algo_name_list[algo_idx]
                wandb.run.summary[f"{algo_name}/{key}/Mean"] = value[algo_idx].mean(axis=-1)
                wandb.run.summary[f"{algo_name}/{key}/Std"] = value[algo_idx].std(axis=-1)
