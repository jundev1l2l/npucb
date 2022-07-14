from torch.utils.data import DataLoader as TorchDataloader
from util.base_config import BaseConfig


class TorchDataloaderConfig(BaseConfig):
    def set_subconfig_class_hash(self):
        self.subconfig_class_hash = {
            "name": str,
            "batch_size": int,
            "num_workers": int,
        }
