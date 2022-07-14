from dataset.torch_dataset import TorchDataset, TorchDatasetConfig
from dataset.base_dataset import BaseDataset, BaseDatasetConfig
from dataset.np_dataset import NPDataset, NPDatasetConfig
from dataset.np_random_dataset import NPRandomDataset, NPRandomDatasetConfig


DATASET_DICT = {
    "TorchDataset": TorchDataset,
    "BaseDataset": BaseDataset,
    "NPDataset": NPDataset,
    "NPRandomDataset": NPRandomDataset,
}


DATASET_CONFIG_DICT = {
    "TorchDataset": TorchDatasetConfig,
    "BaseDataset": BaseDatasetConfig,
    "NPDataset": NPDatasetConfig,
    "NPRandomDataset": NPRandomDatasetConfig,
}


def build_dataset(config, *args, **kwargs):

    DATASET = DATASET_DICT[config.name]
    
    mode_list = ["train", "val"]
    dataset_dict = {}
    for mode in mode_list:
        dataset = DATASET(config, *args, **kwargs)
        dataset_dict[mode] = dataset

    return dataset_dict
