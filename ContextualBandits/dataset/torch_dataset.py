from torch.utils.data import Dataset as TorchDataset
from util.base_config import BaseConfig


class TorchDatasetConfig(BaseConfig):
    def set_subconfig_class_hash(self):
        self.subconfig_class_hash = {
            "name": str,
        }
