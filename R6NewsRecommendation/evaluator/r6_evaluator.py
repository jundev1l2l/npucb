import os
import random
import matplotlib.pyplot as plt
import numpy as np
import wandb
from time import time
from datetime import timedelta
from tqdm import tqdm

from util.base_config import BaseConfig


class R6EvaluatorConfig(BaseConfig):
    def set_subconfig_class_hash(self):
        self.subconfig_class_hash = {
            "name": str,
            "num_exp": int,
            "num_steps": int,
            "learn_ratio": float,
            "plot_comparison_shape": str
        }


class R6Evaluator:
    def __init__(self, config, bandit_algo_baseline, bandit_algo_eval_list, bandit_env, logger, save_dir, rank, debug):
        self.name = config.name
        self.num_exp = config.num_exp
        self.num_steps = config.num_steps
        self.learn_ratio = config.learn_ratio
        self.plot_comparison_shape = config.plot_comparison_shape
        self.bandit_algo_baseline = bandit_algo_baseline
        self.bandit_algo_eval_list = bandit_algo_eval_list
        self.bandit_algo_list = [self.bandit_algo_baseline, *self.bandit_algo_eval_list]
        self.bandit_env = bandit_env
        self.logger = logger
        self.save_dir = save_dir
        self.rank = rank
        self.debug = debug
        if self.debug:
            self.num_steps = 1000

    def eval(self):
        """
        run experiment & plot result
        """
        bandit_algo_baseline_name = self.bandit_algo_baseline.name
        bandit_algo_eval_names = [bandit.name for bandit in self.bandit_algo_eval_list]
        
        learn_curves, learn_ctrs, deploy_ctrs = self.compare()

        self.plot_results(
            names=[bandit_algo_baseline_name, *bandit_algo_eval_names],
            learn_curves=learn_curves,
            learn_ctrs=learn_ctrs,
            deploy_ctrs=deploy_ctrs
        )

    def compare(self):
        learn_curves = []
        learn_ctrs = []
        deploy_ctrs = []
        
        """
        baseline policy evaluation
        """
        _learn_curves = []
        _learn_ctrs = np.zeros([1])
        _deploy_ctrs = np.zeros([1])
        for i in range(self.num_exp):
            self.logger.info(
                f"Baseline Policy ({self.bandit_algo_baseline.name}) Evaluation: Exp {i+1}/{self.num_exp}") if self.rank <= 0 else None
            learn, deploy = self.evaluate(self.bandit_algo_baseline)
            _learn_curves.append(learn)
            _learn_ctrs = _learn_ctrs * i / (i + 1) + learn[-1] / (i + 1)
            _deploy_ctrs = _deploy_ctrs * i / (i + 1) + deploy[-1] / (i + 1)
        _learn_curves = self.gather_learn_curves(_learn_curves)
        learn_curves.append(_learn_curves)
        baseline_ctr = _deploy_ctrs.item()  # todo: config 에서 받아오는 것으로 수정 (반복실험 피하기)
        if baseline_ctr == 0:
            baseline_ctr = 1.0
        learn_ctrs.append(_learn_ctrs.item() / baseline_ctr)
        deploy_ctrs.append(_deploy_ctrs.item() / baseline_ctr)
        
        """
        target policy evaluation and comparison
        """
        for model in self.bandit_algo_eval_list:
            _learn_curves = []
            _learn_ctrs = np.zeros([1])
            _deploy_ctrs = np.zeros([1])
            for i in range(self.num_exp):
                self.logger.info(
                    f"Target Policy ({model.name}) Evaluation: Exp {i+1}/{self.num_exp}") if self.rank <= 0 else None
                learn, deploy = self.evaluate(model)
                _learn_curves.append(learn)
                _learn_ctrs = _learn_ctrs * i / (i + 1) + learn[-1] / (i + 1)
                _deploy_ctrs = _deploy_ctrs * i / \
                    (i + 1) + deploy[-1] / (i + 1)
            _learn_curves = self.gather_learn_curves(_learn_curves)
            learn_curves.append(_learn_curves)
            learn_ctrs.append(_learn_ctrs.item() / baseline_ctr)
            deploy_ctrs.append(_deploy_ctrs.item() / baseline_ctr)

        for bandit, learn_ctr, deploy_ctr in zip(self.bandit_algo_list, learn_ctrs, deploy_ctrs):
            if self.rank <= 0:
                wandb.log({f"learn_{bandit.name}": learn_ctr})
                wandb.log({f"deploy_{bandit.name}": deploy_ctr})
                wandb.run.summary[f"learn_{bandit.name}"] = learn_ctr
                wandb.run.summary[f"deploy_{bandit.name}"] = deploy_ctr

        return learn_curves, learn_ctrs, deploy_ctrs

    def evaluate(self, model):
        start_time = time()
        self.G_deploy = 0  # total payoff for the deployment bucket
        self.G_learn = 0  # total payoff for the learning bucket
        self.T_deploy = 1  # counter of valid events for the deployment bucket
        self.T_learn = 0  # counter of valid events for the learning bucket
        self.learn = []
        self.deploy = []

        data = self.bandit_env.init(self.num_steps)
        terminated = False
        self.curr_step = 0

        self.iter_bar = tqdm(range(self.bandit_env.num_steps if self.num_steps < 0 else self.num_steps))
        for _ in self.iter_bar:
            data, terminated, self.curr_step = self.bandit_env.step()
            # action_list = [algo.choose_arm(context) for algo in self.algo_list]

            # context, reward_list, opt_reward, terminated, self.curr_step = self.bandit_env.step(action_list)
            
            # for algo_idx, algo in enumerate(self.algo_list):
            #     algo.update(context, action_list[algo_idx], reward_list[algo_idx])
            #     self.update_metrics(algo_idx, action_list[algo_idx], reward_list[algo_idx], opt_reward)
            #     self.log(algo_idx, action_list[algo_idx], reward_list[algo_idx], opt_reward) if self.rank <=0 else None
            self.evaluate_step(model, data)
            if terminated:
                # self.summary()
                break

        end_time = time()
        time_str = f"{str(timedelta(seconds=round(end_time - start_time, 1))):0>8}"
        ctr = round(self.G_deploy / self.T_deploy, 4)

        if self.rank <=0:
            self.logger.info(
            f"{model.name:<20}{ctr:<20}{time_str}")
            wandb.log({f"ctr_{model.name}": ctr})
            self.logger.info("")

        return self.learn, self.deploy

    def evaluate_step(self, model, data):
        displayed, reward, user, pool_idx, features = data 
        chosen = model.choose_arm(
                self.G_learn + self.G_deploy, user, pool_idx, features)
        if chosen == displayed:
            if random.random() < self.learn_ratio:  # todo: NP에 맞는 온라인 학습 방식으로 바꿔볼 수 있을 것 (ex) 100 라운드마다 학습
                self.G_learn += reward
                self.T_learn += 1
                model.update(displayed, reward, user, pool_idx, features)
                learn_ctr = self.G_learn / self.T_learn
                self.learn.append(learn_ctr)
                if self.rank <=0:
                    wandb.log(data={f"{model.name}_learn": learn_ctr}, step=self.curr_step)
            else:
                self.G_deploy += reward
                self.T_deploy += 1
                deploy_ctr = self.G_deploy / self.T_deploy
                self.deploy.append(deploy_ctr)
                if self.rank <=0:
                    wandb.log(data={f"{model.name}_deploy": deploy_ctr}, step=self.curr_step)

    def plot_results(self, names, learn_curves, learn_ctrs, deploy_ctrs):
        """
        todo:
          - comparison figure
            - shorten model names on x label, and write full name in the text box at the bottom
            - write real value (learn, deploy ctr values) in the text box
        """
        
        os.makedirs(self.save_dir, exist_ok=True)

        plt.title("Learning Curves")
        for name, curve in list(zip(names, learn_curves)):
            plt.plot(curve, label=name)
        plt.xlabel("T")
        plt.ylabel("CTR")
        plt.legend(loc='lower right')
        plot_file = os.path.join(self.save_dir, "learning-curves.png")
        plt.savefig(plot_file)
        self.logger.info("Learning Curves Figure saved at " + plot_file) if self.rank <= 0 else None
        wandb.log({"learning_curve": wandb.Image(plt)}) if self.rank <= 0 else None
        plt.close()

        if self.plot_comparison_shape == "line":
            plt.title("Comparison")
            plt.plot(names, learn_ctrs, label="learn", marker="o")
            plt.plot(names, deploy_ctrs, label="deploy", marker="x")
            plt.ylabel("Relative CTR")
            plt.legend(loc='lower right')
        elif self.plot_comparison_shape == "bar":
            bar_width = 0.35
            plt.title("Comparison")
            plt.bar(x=names, height=learn_ctrs,
                    width=bar_width, color="b", label="learn")
            plt.bar(x=np.arange(len(names)) + bar_width, height=deploy_ctrs,
                    width=bar_width, color="r", label="deploy")
            plt.ylabel("Relative CTR")
            plt.legend(loc='upper left')
        plot_file = os.path.join(self.save_dir, "comparison.png")
        plt.savefig(plot_file)
        self.logger.info("Comparison Figure saved at " + plot_file) if self.rank <= 0 else None
        wandb.log({"comparison": wandb.Image(plt)}) if self.rank <= 0 else None
        plt.close()

    def gather_learn_curves(self, curves):
        len_curve = min([len(curve) for curve in curves])
        curves = [curve[:len_curve] for curve in curves]
        return np.mean(np.array(curves), axis=0)
        