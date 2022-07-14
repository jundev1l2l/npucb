import torch
from attrdict import AttrDict

from util.base_config import BaseConfig
from trainer.base_trainer import BaseTrainer


class NPTrainerConfig(BaseConfig):
    def set_subconfig_class_hash(self):
        self.subconfig_class_hash = {
            "name": str,
            "lr": float,
            "num_epochs": int,
            "loss": str,
            "clip_loss": float,
            "val_freq": int,
            "save_freq": int
        }


class NPTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super(NPTrainer, self).__init__( *args, **kwargs)

    def get_batch_dict(self, batch):
        xc, yc, xt, yt = self.check_dimension(list(map(lambda x: x.cuda(), batch)))
        x = torch.cat([xc, xt], dim=1)
        y = torch.cat([yc, yt], dim=1)
        batch_dict = AttrDict({"xc": xc, "yc": yc, "xt": xt, "yt": yt, "x": x, "y": y})

        return batch_dict

    def check_dimension(self, val_list):
        result = []
        for val in val_list:
            result.append(val.unsqueeze(dim=1) if val.ndim < 3 else val)
        return result
        