from trainer.base_trainer import BaseTrainer, BaseTrainerConfig
from trainer.np_trainer import NPTrainer, NPTrainerConfig
from trainer.feature_extractor_trainer import FeatureExtractorTrainer, FeatureExtractorTrainerConfig


TRAINER_DICT = {
        "BaseTrainer": BaseTrainer,
        "NPTrainer": NPTrainer,
        "FeatureExtractorTrainer": FeatureExtractorTrainer,
}

TRAINER_CONFIG_DICT = {
        "BaseTrainer": BaseTrainerConfig,
        "NPTrainer": NPTrainerConfig,
        "FeatureExtractorTrainer": FeatureExtractorTrainerConfig,
}


def build_trainer(config, *args, **kwargs):
    trainer = TRAINER_DICT[config.name](config=config, *args, **kwargs)

    return trainer
