from typing import OrderedDict


def remove_module_from_dict(model_dict):
    model_dict_without_module = OrderedDict()
    for key, value in model_dict.items():
        key_split = key.split(".")
        if key_split[0] == "module":
            key_split.pop(0)
        key = ".".join(key_split)
        model_dict_without_module[key] = value

    return model_dict_without_module
