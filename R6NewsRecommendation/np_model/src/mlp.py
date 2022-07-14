import torch.nn as nn
from torch.nn import MSELoss, BCELoss
from attrdict import AttrDict

from util.base_config import BaseConfig


class MLPModelConfig(BaseConfig):
    def set_subconfig_class_hash(self):
        self.subconfig_class_hash = {
            "name": str,
            "size" : str,
            "path": str,
            "ckpt": str
        }


def make_layers(dx, dy, dh):
    layers = [nn.Linear(dx, dh[0]), nn.ReLU()]
    for i in range(0, len(dh) - 1):
        layers.append(nn.Linear(dh[i], dh[i + 1]))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(dh[-1], dy))
    layers = nn.ModuleList(layers)
    return layers


class MLPModel(nn.Module):
    def __init__(self,
                 dim_x,
                 dim_y,
                 dim_hid,
                 ):
        super(MLPModel, self).__init__()
        self.name = "MLP"
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_hid = dim_hid
        self.layers = make_layers(dim_x, dim_y, dim_hid)

    def predict(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def forward(self, batch, loss="l2", clip_loss=.0):
        x, y = batch.x, batch.y
        preds = self.predict(x).squeeze(dim=0)
        outs = AttrDict()
        outs.ys = preds
        if loss == "l2":
            outs.loss = MSELoss()(preds, y.squeeze(dim=0))
        elif loss == "bce":
            preds = nn.Sigmoid()(preds)
            outs.ys = preds
            outs.loss = BCELoss()(preds, y.squeeze(dim=0))
        else:
            raise KeyError(f"MLP model do not have {self.loss} loss. It only has L2, BCE loss.")

        return outs