import json
import yaml

from copy import deepcopy


class BaseConfig:
    def __init__(self, file_or_dict):
        self.set_subconfig_class_hash()
        self.set(file_or_dict)

    def set_subconfig_class_hash(self):
        self.subconfig_class_hash = {}

    def __getitem__(self, key):
        return getattr(self, key)
    
    def __setitem__(self, key, value):
        self.set_config(key, value)

    def __call__(self):
        config_dict = deepcopy(vars(self))
        config_dict.pop("subconfig_class_hash")
        for key, value in config_dict.items():
            if isinstance(value, list) and (self.subconfig_class_hash[key] != list):
                config_dict[key] = [v() for v in value]
            if "Config" in type(value).__name__:
                config_dict[key] = value()
        return config_dict

    def __str__(self):
        return yaml.dump(self.__call__(), sort_keys=False)

    def set(self, file_or_dict):
        if isinstance(file_or_dict, dict):
            config_dict = file_or_dict
        else:
            if file_or_dict.endswith(".json"):
                with open(file_or_dict, "r") as f:
                    config_dict = json.load(f)
            elif file_or_dict.endswith(".yaml"):
                with open(file_or_dict, "r") as f:
                    config_dict = yaml.safe_load(f)
        for key, value in config_dict.items():
            key = key.lower()
            self.set_config(key, value)

    def set_config(self, key, value):
        key = key.lower()
        try:
            subconfig_class = self.subconfig_class_hash[key]
        except:
            raise KeyError(f"Key \"{key}\" not in subconfig_class_hash of {type(self)}")
        if isinstance(subconfig_class, dict):
            if isinstance(value, list):
                setattr(self, key, [subconfig_class[v["name"]](v) for v in value])
            else:
                setattr(self, key, subconfig_class[value["name"]](value))
        else:
            if isinstance(value, list) and (subconfig_class != list):
                setattr(self, key, [subconfig_class(v) for v in value])
            else:
                if key in self.subconfig_class_hash.keys():
                    try:
                        setattr(self, key, subconfig_class(value))
                    except TypeError:
                        expected_type = subconfig_class.__name__
                        raise TypeError(f"Config Error in {type(self).__name__}\n\nKey: {key}\n\nValue: {value}\n\nCurrent Type: {type(value).__name__}\n\nExpected Type: {expected_type}")
                else:
                    print(f"Warning: Key \"{key}\" is not defined in {type(self).__name__}")
                    if isinstance(value, dict):
                        setattr(self, key, BaseConfig(value))
                    else:
                        setattr(self, key, value)
    
    def update(self, file_or_dict):
        if isinstance(file_or_dict, dict):
            config_dict = file_or_dict
        else:
            if file_or_dict.endswith(".json"):
                with open(file_or_dict, "r") as f:
                    config_dict = json.load(f)
            elif file_or_dict.endswith(".yaml"):
                with open(file_or_dict, "r") as f:
                    config_dict = yaml.safe_load(f)
        self.update_config(config_dict)
    
    def update_config(self, config_dict):
        for key, value in config_dict.items():
            if key in vars(self).keys():
                if isinstance(value, dict):
                    getattr(self, key).update_config(value)
                else:
                    self.set_config(key, value)
            else:
                self.set_config(key, value)


if __name__ == "__main__":
    config = BaseConfig("default_train.yaml")
    print(config)
    config.update("exp_train.yaml")
    print("config updated\n")
    print(config)
