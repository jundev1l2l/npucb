from curses import has_key
import torch
import torch.cuda as c
import logging
import wandb
import yaml
import os
import os.path as osp
import shutil

from time import time
from collections import OrderedDict
from matplotlib import pyplot as plt
from os.path import split, splitext


def dump_config(args_dict, config):
    log_dir = config.result
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, "args.yaml"), "w") as f:
        dump_yaml(args_dict, f)
    with open(os.path.join(log_dir, "config.yaml"), "w") as f:
        dump_yaml(config(), f)
    shutil.copyfile(args_dict["default_config"],
                    os.path.join(log_dir, "default_config.yaml"))
    if config.exp_config != "":
        shutil.copyfile(args_dict["exp_config"],
                        os.path.join(log_dir, "exp_config.yaml"))


def dump_yaml(data, file):
    class MyDumper(yaml.SafeDumper):
        # HACK: insert blank lines between top-level objects
        # inspired by https://stackoverflow.com/a/44284819/3786245
        def write_line_break(self, data=None):
            super().write_line_break(data)
            if len(self.indents) == 1:
                super().write_line_break()
    yaml.dump(data, file, Dumper=MyDumper, sort_keys=False)


def get_logger(task, log_dir, debug):
    log_file = f"debug.log" if debug else f"{task}.log"
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger()
    for hdlr in logger.handlers[:]:
        logger.removeHandler(hdlr)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(os.path.join(log_dir, log_file))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def init_wandb(project, log_dir, config_dict, mode):
    exp_group_name = log_dir.split("result/")[1].split("/")
    exp_group = "/".join(exp_group_name[:-1])
    exp_name = str(exp_group_name[-1])
    wandb.init(
        project=project,
        group=exp_group,
        name=exp_name,
        config=config_dict,
        dir=log_dir,
        mode=mode,
    )


class ValueLogger:
    def __init__(self, logger):
        self.logger = logger
        self.history = {}

    def reset(self):
        self.history = {}

    def save(self, name, values):
        self.history[name] = values

    def diff(self, baseline, proposed):
        val_baseline = self.history[baseline]
        val_proposed = self.history[proposed]
        val_diff = []
        for base, prop in list(zip(val_baseline, val_proposed)):
            val_diff.append(prop - base)
        val_diff = list(
            map(lambda x: round(x.abs().max().item(), 8), val_diff))
        self.logger.info(f"[VALUE]")
        self.logger.info(val_diff)
        self.logger.info("")


class TimeLogger:
    def __init__(self, logger):
        self.curr = 0
        self.logs = []
        self.logger = logger
        self.history = {}

    def reset(self):
        self.curr = time()

    def log(self, name):
        self.logs.append(f"{name}: + {(time() - self.curr)} s, "
                         f"total {time()} s")
        self.save(name)
        self.reset()

    def flush(self):
        self.logger.info("[TIME]")
        for log in self.logs:
            self.logger.info(log)
        self.logger.info("")
        self.logs = []

    def save(self, name):
        if name in self.history.keys():
            self.history[name].append(time() - self.curr)
        else:
            self.history[name] = [time() - self.curr]

    def summary(self, previous_log=False):
        if not previous_log:
            self.logs = []
        summary = {}
        for key, val in self.history.items():
            summary[key] = {}
            summary[key]["sum"] = sum(val)
            summary[key]["len"] = len(val)
            summary[key]["mean"] = sum(val) / len(val)
            self.logs.append(
                f"{key}: Sum {sum(val)}, Len {len(val)}, Mean {sum(val) / len(val)}")
        self.flush()
        return summary


class MemLogger:
    def __init__(self, logger):
        self.curr = 0
        self.logs = []
        self.logger = logger
        self.history = {}

    def reset(self):
        self.curr = c.memory_allocated()

    def log(self, name):
        self.logs.append(f"{name}: + {(c.memory_allocated() - self.curr) / 1024 / 1024} MiB, "
                         f"total {c.memory_allocated()/1024/1024} MiB")
        self.save(name)
        self.reset()

    def flush(self):
        self.logger.info("[MEMORY]")
        for log in self.logs:
            self.logger.info(log)
        self.logger.info("")
        self.logs = []

    def save(self, name):
        if name in self.history.keys():
            self.history[name].append(c.memory_allocated() - self.curr)
        else:
            self.history[name] = [c.memory_allocated() - self.curr]

    def summary(self, previous_log=False):
        if not previous_log:
            self.logs = []
        summary = {}
        for key, val in self.history.items():
            summary[key] = {}
            summary[key]["sum"] = sum(val)
            summary[key]["len"] = len(val)
            summary[key]["mean"] = sum(val) / len(val)
            self.logs.append(
                f"{key}: Sum {sum(val)}, Len {len(val)}, Mean {sum(val) / len(val)}")
        self.flush()
        return summary


class RunningAverage(object):
    def __init__(self, *keys):
        self.sum = OrderedDict()
        self.cnt = OrderedDict()
        self.clock = time()
        for key in keys:
            self.sum[key] = 0
            self.cnt[key] = 0

    def update(self, key, val):
        if isinstance(val, torch.Tensor):
            val = val.item()
        if self.sum.get(key, None) is None:
            self.sum[key] = val
            self.cnt[key] = 1
        else:
            self.sum[key] = self.sum[key] + val
            self.cnt[key] += 1

    def reset(self):
        for key in self.sum.keys():
            self.sum[key] = 0
            self.cnt[key] = 0
        self.clock = time()

    def clear(self):
        self.sum = OrderedDict()
        self.cnt = OrderedDict()
        self.clock = time()

    def keys(self):
        return self.sum.keys()

    def get(self, key):
        assert(self.sum.get(key, None) is not None)
        return self.sum[key] / self.cnt[key]

    def info(self, show_et=True):
        line = ''
        for key in self.sum.keys():
            val = self.sum[key] / self.cnt[key]
            if type(val) == float:
                line += f'{key} {val:.4f} '
            else:
                line += f'{key} {val} +-'.format(key, val)
        if show_et:
            line += f'({time()-self.clock:.3f} secs)'
        return line


def get_log(fileroot):
    step = []
    loss = []
    train_time = []
    eval_time = []
    ctxll = []
    tarll = []
    file = open(fileroot, "r")
    lines = file.readlines()
    for line in lines:
        # training step
        if "step" in line:
            linesplit = line.split(" ")
            step += [int(linesplit[3])]
            _loss = linesplit[-3]
            loss += [100 if _loss == "nan" else float(_loss)]
            train_time += [float(linesplit[-2][1:])]
        # evaluation step
        elif "ctx_ll" in line:
            linesplit = line.split(" ")
            ctxll += [float(linesplit[-5])]
            tarll += [float(linesplit[-3])]
            eval_time += [float(linesplit[-2][1:])]

    return step, loss, None, ctxll, tarll


def plot_log(fileroot, x_begin=None, x_end=None):
    plt.clf()  # clear current figure

    step, loss, stepll, ctxll, tarll = get_log(fileroot)
    step = list(map(int, step))
    loss = list(map(float, loss))
    ctxll = list(map(float, ctxll))
    tarll = list(map(float, tarll))
    stepll = list(map(int, stepll)) if stepll else None

    if x_begin is None:
        x_begin = 0
    if x_end is None:
        x_end = step[-1]

    print_freq = 1 if len(step) == 1 else step[1] - step[0]

    plt.plot(step[x_begin//print_freq:x_end//print_freq],
             loss[x_begin//print_freq:x_end//print_freq])
    plt.xlabel('step')
    plt.ylabel('loss')

    dir, file = split(fileroot)
    filename = splitext(file)[0]
    plt.savefig(dir + "/" + filename + f"-{x_begin}-{x_end}.png")


def plot_freq_cov():
    """
    # backbone.yaml
    root:
      "../../scratch/banp/junhyun/1d/"
    backbone:
      cnp:
        "reproduce/"
      canp:
        "reproduce/"
      ...
    """
    with open(osp.join("model_paths.yaml")) as f:
        model_paths = yaml.safe_load(f)
    root = model_paths["root"]
    model_paths = model_paths["backbone"]
    x_base = torch.linspace(-2, 2, 500).unsqueeze(-1)

    for kernel in ["rbf", "periodic", "matern"]:
        plt.clf()
        for model, path in model_paths.items():
            freq_cov = torch.load(
                osp.join(root, model, path, f"freq_cov_{kernel}.pt"))
            # plt.scatter(x_base.cpu(), freq_cov.cpu(), s=3, alpha=1.0, label=model+f"-{freq_cov.mean():0.2f}")
            plt.scatter(x_base.cpu(), freq_cov.cpu(), s=3, alpha=1.0,
                        label=f"{model}-{freq_cov.mean():0.2f}")
            plt.ylim([0, 1])

        # backbone = "_".join(model_paths.keys())
        models = "all"
        plt.legend()
        plt.title(f"Frequentist Coverage - {kernel}")
        if not osp.exists(osp.join(root, "plot", "freq_cov", models)):
            os.makedirs(osp.join(root, "plot", "freq_cov", models))
        plt.savefig(osp.join(root, "plot", "freq_cov",
                    models, f"freq_cov_{kernel}.jpg"))


if __name__ == "__main__":
    plot_freq_cov()
