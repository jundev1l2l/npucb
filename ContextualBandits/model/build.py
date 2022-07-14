import os
import torch

from model.src.mlp import MLPModel, MLPModelConfig
from model.src.np import NPModel, NPModelConfig
from model.src.anp import ANPModel, ANPModelConfig
from model.src.bnp import BNPModel, BNPModelConfig
from model.src.banp import BANPModel, BANPModelConfig

from util.custom_ddp import CustomDDP
from util.load import remove_module_from_dict


MODEL_DICT = {
    "MLP": MLPModel,
    "NP": NPModel,
    "ANP": ANPModel,
    "BNP": BNPModel,
    "BANP": BANPModel,
}

MODEL_CONFIG_DICT = {
    "MLP": MLPModelConfig,
    "NP": NPModelConfig,
    "ANP": ANPModelConfig,
    "BNP": BNPModelConfig,
    "BANP": BANPModelConfig,
}


def build_model(config, rank, debug):

    MODEL = MODEL_DICT[config.name]
    
    config_dict = config()
    config_dict.pop("name", None)
    config_dict.pop("ckpt", None)
    for key in config_dict.keys():
        if key not in config.subconfig_class_hash.keys():
            config_dict.pop(key, None)

    model = MODEL(**config_dict)

    if hasattr(config, "ckpt"):
        if debug:
            ckpt_path_list = config.ckpt.split("/")
            ckpt_path_list.insert(-1, "debug")
            ckpt_file = "/".join(ckpt_path_list)
        else:
            ckpt_file = config.ckpt
        model_dict = torch.load(ckpt_file, map_location="cuda" if rank >= 0 else "cpu")
        model_dict = remove_module_from_dict(model_dict)
        model.load_state_dict(model_dict)

    model.eval()

    if rank >= 0:
        model = model.cuda(rank)
        model = CustomDDP(
            model,
            device_ids=[rank],
            find_unused_parameters=True
        )

    return model


if __name__ == "__main__":
    os.chdir("../")
    model = build_model("np")
    print(model)
    
