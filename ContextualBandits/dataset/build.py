from dataset.torch_dataset import TorchDataset, TorchDatasetConfig
from dataset.base_dataset import BaseDataset, BaseDatasetConfig
from dataset.np_dataset import NPDataset, NPDatasetConfig
from dataset.np_random_dataset import NPRandomDataset, NPRandomDatasetConfig
from dataset.gp_random_dataset import GPRandomDataset, GPRandomDatasetConfig


DATASET_DICT = {
    "TorchDataset": TorchDataset,
    "BaseDataset": BaseDataset,
    "NPDataset": NPDataset,
    "NPRandomDataset": NPRandomDataset,
    "GPRandomDataset": GPRandomDataset,
}


DATASET_CONFIG_DICT = {
    "TorchDataset": TorchDatasetConfig,
    "BaseDataset": BaseDatasetConfig,
    "NPDataset": NPDatasetConfig,
    "NPRandomDataset": NPRandomDatasetConfig,
    "GPRandomDataset": GPRandomDatasetConfig,
}


def build_dataset(config, *args, **kwargs):
    
    DATASET = DATASET_DICT[config.name]

    mode_list = ["train", "val"]
    dataset_dict = {}
    for mode in mode_list:
        dataset = DATASET(config, *args, **kwargs)
        dataset_dict[mode] = dataset
    
    return dataset_dict
