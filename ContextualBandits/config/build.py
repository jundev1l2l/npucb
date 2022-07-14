import json
import yaml
from copy import deepcopy

from config.train_config import TrainConfig
from config.eval_config import EvalConfig


CONFIG_DICT = {
    "train": TrainConfig,
    "eval": EvalConfig
}


def build_config(task, default_config, exp_config):

    CONFIG = CONFIG_DICT[task]

    config_dict = get_config_dict(default_config)
    if exp_config != "":
        exp_config_dict = get_config_dict(exp_config)
        config_dict = merge_config_dict(config_dict, exp_config_dict)
    config_dict = clean_config_dict(config_dict)

    config = CONFIG(config_dict)

    return config


def get_config_dict(file_or_dict):  # todo: List[Dict] 인 경우 (ex. Bandit List, Model List) 추가하기

    if isinstance(file_or_dict, dict):
            config_dict = file_or_dict
    else:
        if file_or_dict.endswith(".json"):
            with open(file_or_dict, "r") as f:
                config_dict = json.load(f)
        elif file_or_dict.endswith(".yaml"):
            with open(file_or_dict, "r") as f:
                config_dict = yaml.safe_load(f)

    return config_dict


def merge_config_dict(old, new):

    if (old is not None) and (type(old) != type(new)):
        raise TypeError(f"""Type Error when updating config_dict\n
        Old Dict: \n
            {old} \n
        New Dict: \n
            {new}\n
        """)
    elif isinstance(new, dict):
        merged = old
        for key in new.keys():
            if key in old.keys():
                merged[key] = merge_config_dict(old[key], new[key])
            else:
                merged[key] = new[key]
        return merged
    else:
        return new


def clean_config_dict(config_dict):

    cleaned_dict = deepcopy(config_dict)
    for key, value in config_dict.items():
        if isinstance(value, dict):
            cleaned_dict[key] = clean_config_dict(value)
        elif value is None:
            cleaned_dict.pop(key, None)
        else:
            continue

    return cleaned_dict
