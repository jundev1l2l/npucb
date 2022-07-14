from util.base_config import BaseConfig
from engine import ENGINE_CONFIG_DICT
from data_sampler import DATA_SAMPLER_CONFIG_DICT
from dataset import DATASET_CONFIG_DICT
from dataloader import DATALOADER_CONFIG_DICT
from model import MODEL_CONFIG_DICT
from trainer import TRAINER_CONFIG_DICT
from feature_extractor import FEATURE_EXTRACTOR_CONFIG_DICT


class TrainConfig(BaseConfig):
    def set_subconfig_class_hash(self):
        self.subconfig_class_hash = {
            "project": str,
            "task": str,
            "default_config": str,
            "exp_config": str,
            "result": str,
            "debug": bool,
            "seed": int,
            "wandb_mode": str,
            "engine": ENGINE_CONFIG_DICT,
            "data_sampler": DATA_SAMPLER_CONFIG_DICT,
            "dataset": DATASET_CONFIG_DICT,
            "dataloader": DATALOADER_CONFIG_DICT,
            "model": MODEL_CONFIG_DICT,
            "feature_extractor": FEATURE_EXTRACTOR_CONFIG_DICT,
            "trainer": TRAINER_CONFIG_DICT,
        }
