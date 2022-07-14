import torch.nn as nn
from torch.nn import MSELoss, BCELoss
from attrdict import AttrDict

from util.base_config import BaseConfig


class MLPModelConfig(BaseConfig):
    def set_subconfig_class_hash(self):
        self.subconfig_class_hash = {
            "name": str,
            "dim_x" : int,
            "dim_y": int,
            "dim_hid": list,
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
        print(f"Number of Parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad)}")

    def predict(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def forward(self, batch, loss="mse", clip_loss=.0):
        x, y = batch.x, batch.y
        preds = self.predict(x).squeeze(dim=0)
        outs = AttrDict()
        outs.ys = preds
        if loss == "mse":
            outs.loss = MSELoss()(preds, y.squeeze(dim=0))
        elif loss == "bce":
            preds = nn.Sigmoid()(preds)
            outs.ys = preds
            outs.loss = BCELoss()(preds, y.squeeze(dim=0))
        else:
            raise KeyError(f"MLP model do not have {loss.upper()} loss. It only has MSE, BCE loss.")

        return outs
