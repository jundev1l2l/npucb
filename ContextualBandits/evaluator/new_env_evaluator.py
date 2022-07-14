import os
import numpy as np
import wandb
from tqdm import tqdm
import matplotlib.pyplot as plt

from util.log import MemLogger, TimeLogger
from util.base_config import BaseConfig
from util.misc import get_circum_points


class NewEnvEvaluatorConfig(BaseConfig):
    def set_subconfig_class_hash(self):
        self.subconfig_class_hash = {
            "name": str,
            "num_exp": int,
            "num_steps": int,
            "plot_freq": int,
            "plot_grid_size": int,
            "new_envs_at_steps": list,
        }


class NewEnvEvaluator:
    def __init__(self, config, bandit_algo_baseline, bandit_algo_eval_list, bandit_env, logger, save_dir, rank, debug):
        self.name = config.name
        self.num_exp = config.num_exp
        self.num_steps = config.num_steps
        if hasattr(config, "plot_freq"):
            self.plot_freq = config.plot_freq if config.plot_freq > 0 else self.num_steps
        else:
            self.plot_freq = self.num_steps
        if hasattr(config, "plot_grid_size"):
            self.plot_grid_size = config.plot_grid_size
        else:
            self.plot_grid_size = 50

        self.baseline_algo = bandit_algo_baseline
        self.baseline_algo_name = self.baseline_algo.name

        self.eval_algo_list = bandit_algo_eval_list
        self.eval_algo_name_list = [algo.name for algo in self.eval_algo_list]
        
        self.all_algo_list = [self.baseline_algo, *self.eval_algo_list]
        self.all_algo_name_list = [self.baseline_algo_name, *self.eval_algo_name_list]
        self.num_algos = len(self.all_algo_list)

        self.logger = logger
        self.mlogger = MemLogger(logger)
        self.tlogger = TimeLogger(logger)
        self.save_dir = save_dir
        self.rank = rank
        self.debug = debug
        if self.debug:
            self.num_steps = 10

        # different
        assert isinstance(bandit_env, list), "NewEnvEvaluator needs a list of Bandit Envs for parameter \"bandit_env\""
        self.bandit_env_list = bandit_env
        self.bandit_env = self.bandit_env_list[0]
        self.num_arms = self.bandit_env.num_arms
        
        self.new_envs_at_steps = config.new_envs_at_steps
        self.new_envs = [self.bandit_env_list[env_at_step[0]] for env_at_step in self.new_envs_at_steps]
        self.new_envs_at = [env_at_step[1] for env_at_step in self.new_envs_at_steps]
        

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
            for algo in self.all_algo_list:
                algo.init()
            self.run_idx = run_idx
            self.run_bandit()
        self.log_summary() if self.rank <=0 else None

    def plot(self, algo_idx):
        grid_size = 50
        algo = self.all_algo_list[algo_idx]
        algo_name = self.all_algo_name_list[algo_idx]

        if type(self.bandit_env.data_sampler).__name__ == "WheelDataSampler" and "NPUcb" in type(algo).__name__:  # todo: write codes for other algorithms
            x, y = np.meshgrid(np.linspace(-1, 1, grid_size), np.linspace(-1, 1, grid_size))
            samples = np.array(list(zip(x.reshape(-1), y.reshape(-1))))
            mean, std, ucbs = algo.reward_distribution(samples)

            outer_radius = 0.5 * grid_size
            inner_radius = 0.5 * grid_size * self.bandit_env.data_sampler.delta
            outer_circle = get_circum_points(outer_radius, grid_size ** 2) + [outer_radius, outer_radius]
            inner_circle = get_circum_points(inner_radius, grid_size ** 2) + [outer_radius, outer_radius]
            left_x_axis = np.stack([np.linspace(0, outer_radius - inner_radius, grid_size ** 2), outer_radius * np.ones(grid_size ** 2)], axis=-1)
            right_x_axis = np.stack([np.linspace(outer_radius + inner_radius, grid_size, grid_size ** 2), outer_radius * np.ones(grid_size ** 2)], axis=-1)
            lower_y_axis = np.stack([outer_radius * np.ones(grid_size ** 2), np.linspace(0, outer_radius - inner_radius, grid_size ** 2)], axis=-1)
            upper_y_axis = np.stack([outer_radius * np.ones(grid_size ** 2), np.linspace(outer_radius + inner_radius, grid_size, grid_size ** 2)], axis=-1)

            boundary = [
                outer_circle, 
                inner_circle, 
                left_x_axis,
                right_x_axis,
                lower_y_axis,
                upper_y_axis,
            ]

            for a, (m, s, u) in enumerate(zip(mean, std, ucbs)):
                fig, axes = plt.subplots(1, 3, figsize=(20, 6))
                plt.suptitle(f"Reward Prediction of Arm {a} at Step {self.curr_step}", size=40)
                data_dict = {
                    "Mean": m,
                    "Std": s,
                    "Ucb": u,
                }
                for idx, (name, data) in enumerate(data_dict.items()):
                    axes[idx].set_title(name, size=30)
                    c = axes[idx].pcolor(data.reshape(grid_size, grid_size), cmap="jet", )
                    for b in boundary:
                        axes[idx].plot(b[:, 0], b[:, 1], "k-")
                    plt.sca(axes[idx])
                    num_ticks = 10
                    plt.xticks(np.arange(num_ticks + 1) * int(grid_size / num_ticks), np.linspace(-1, 1, num_ticks + 1).round(1))
                    plt.yticks(np.arange(num_ticks + 1) * int(grid_size / num_ticks), np.linspace(-1, 1, num_ticks + 1).round(1))
                    plt.colorbar(c, ax=axes[idx])

                plot_dir = os.path.join(self.save_dir, "plot")
                os.makedirs(plot_dir, exist_ok=True)
                image_file = os.path.join(plot_dir, f"RewardPrediction_Arm={a}_Step={self.curr_step}.png")
                plt.tight_layout()
                plt.savefig(image_file)
                plt.close()
                wandb.log(data={
                    f"{algo_name}/RewardPrediction/Arm={a}": wandb.Image(image_file),
                }, step=self.curr_step)
    
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
        for epoch, _ in enumerate(self.iter_bar):

            # different
            if self.curr_step in self.new_envs_at[1:]:
                idx = self.new_envs_at.index(self.curr_step)
                self.bandit_env = self.new_envs[idx]
                context = self.bandit_env.init(self.num_steps, total_steps=self.curr_step)
                terminated = False

            action_list = [algo.choose_arm(context) for algo in self.all_algo_list]

            context, reward_list, opt_reward, terminated, self.curr_step = self.bandit_env.step(action_list)
            
            for algo_idx, algo in enumerate(self.all_algo_list):
                algo.update(context, action_list[algo_idx], reward_list[algo_idx])
                self.update_metrics(algo_idx, action_list[algo_idx], reward_list[algo_idx], opt_reward)
                self.log(algo_idx, action_list[algo_idx], reward_list[algo_idx], opt_reward) if self.rank <=0 else None
                if (epoch + 1) % self.plot_freq == 0:
                    self.plot(algo_idx) if self.rank <=0 else None
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
        algo_name = self.all_algo_name_list[algo_idx]
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
        algo_name = self.all_algo_name_list[algo_idx]
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
                algo_name = self.all_algo_name_list[algo_idx]
                wandb.run.summary[f"{algo_name}/{key}/Mean"] = value[algo_idx].mean(axis=-1)
                wandb.run.summary[f"{algo_name}/{key}/Std"] = value[algo_idx].std(axis=-1)
