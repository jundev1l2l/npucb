from trainer.base_trainer import BaseTrainer, BaseTrainerConfig
from trainer.np_trainer import NPTrainer, NPTrainerConfig


TRAINER_DICT = {
        "BaseTrainer": BaseTrainer,
        "NPTrainer": NPTrainer,
}

TRAINER_CONFIG_DICT = {
        "BaseTrainer": BaseTrainerConfig,
        "NPTrainer": NPTrainerConfig,
}


def build_trainer(config, *args, **kwargs):
    trainer = TRAINER_DICT[config.name](config=config, *args, **kwargs)

    return trainer
