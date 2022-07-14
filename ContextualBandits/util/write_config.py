import itertools
import os
import json
from argparse import Namespace
import sys

sys.path.insert(0, "/home/junhyun/projects/npucb/ContextualBandits")

from util.log import dump_yaml


def get_config_dict_from_args(args: Namespace) -> dict:
    args_dict = vars(args)
    config_dict = get_config_dict(args_dict.keys(), args_dict.values())
    
    return config_dict


def get_expid(names, values):
    exp_name = ""
    
    for name, value in zip(names, values):
        if name == "":
            continue
        elif name == "LR":
            exp_name += f"{name}={value:.0e}_"
        else:
            exp_name += f"{name}={value}_"
    exp_name = exp_name[:-1]

    return exp_name


def parse_key(key):
    return key.split(".")


def get_config_dict(keys, values):
    config_dict = {}
    for key, value in zip(keys, values):
        key = parse_key(key)
        if len(key) == 1:
            config_dict[key[0]] = value
        else:
            for i, k in enumerate(key):    
                if i == 0:
                    if k not in config_dict.keys():
                        config_dict[k] = {}
                    cfg = config_dict[k]
                elif 0 < i < (len(key) - 1):
                    cfg[k] = {}
                    cfg = cfg[k]
                elif i == (len(key) - 1):
                    cfg[k] = value
    return config_dict


def write_config(setting_list, config_path, result_path):
    os.makedirs(config_path, exist_ok=True)
    print(f"Following Experiment Configs Generated at {config_path}:\n")

    for setting in setting_list:
        names = [x[0] for x in setting]
        keys = [x[1] for x in setting]
        value_list = [x[2] for x in setting]

        for values in itertools.product(*value_list):
            values = list(values)
            expid = get_expid(names, values)
            config_dict = get_config_dict(keys, values)
            config_dict["result"] = os.path.join(result_path, expid)
            
            with open(os.path.join(config_path, expid + ".yaml"), "w") as f:
                dump_yaml(config_dict, f)
            print(expid + ".yaml")
    print()


if __name__ == "__main__":
    from util.write_script import write_script

    no_setting_exp_list = []

    from script.train.expid_list import task, exp_group_list
    for exp_group in exp_group_list:
        setting_file = f"script/{task}/{exp_group}/setting.json"
        if os.path.exists(setting_file):
            with open(setting_file, "r") as f:
                setting_list = json.load(f)
            config_path = f"script/{task}/{exp_group}/exp_config"
            result_path = f"/data/ContextualBandits/result/{task}/{exp_group}"
            write_config(setting_list, config_path, result_path)
        else:
            no_setting_exp_list.append((task, exp_group))
        script_path = f"script/{task}/{exp_group}"
        write_script(task, script_path)

    from script.eval.expid_list import task, exp_group_list
    for exp_group in exp_group_list:
        setting_file = f"script/{task}/{exp_group}/setting.json"
        if os.path.exists(setting_file):
            with open(setting_file, "r") as f:
                setting_list = json.load(f)
            config_path = f"script/{task}/{exp_group}/exp_config"
            result_path = f"/data/ContextualBandits/result/{task}/{exp_group}"
            write_config(setting_list, config_path, result_path)
        else:
            no_setting_exp_list.append((task, exp_group))
        script_path = f"script/{task}/{exp_group}"
        write_script(task, script_path)

    print()
    for (task, exp_group) in no_setting_exp_list:
        print(f"Experiment \"{task}/{exp_group}\" has manual settings (no setting.json in directory)")
