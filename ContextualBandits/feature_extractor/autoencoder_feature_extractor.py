import torch
import torch.nn as nn
from attrdict import AttrDict

from util.base_config import BaseConfig


class AutoencoderFeatureExtractorConfig(BaseConfig):
    
    def set_subconfig_class_hash(self):
        self.subconfig_class_hash = {
            "name": str,
            "dim_x": int,
            "dim_encoder_list": list,
            "dim_decoder_list": list,
            "dim_encoded": int,
            "activation_type": str,
            "ckpt": str,
        }


class AutoencoderFeatureExtractor(nn.Module):
    def __init__(self, config):
        super(AutoencoderFeatureExtractor,self).__init__()
        self.name = config.name
        self.dim_x = config.dim_x
        self.dim_encoder_list = config.dim_encoder_list
        self.dim_decoder_list = config.dim_decoder_list if hasattr(config, "dim_decoder_list") else self.dim_encoder_list[::-1]
        self.dim_encoded = config.dim_encoded
        self.activation_type = config.activation_type
        self.activation_fn = {"relu": nn.ReLU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid, "identity": nn.Identity}.get(self.activation_type)
        self.encoder = self.get_layer(self.dim_x, self.dim_encoder_list, self.dim_encoded)
        self.decoder = self.get_layer(self.dim_encoded, self.dim_decoder_list, self.dim_x)
    
    def get_layer(self, dim_in, dim_hidden_list, dim_out):

        layers = [nn.Linear(dim_in, dim_hidden_list[0]), self.activation_fn()]
        if len(dim_hidden_list) > 1:
            for i in range(len(dim_hidden_list) - 1):
                layers.append(nn.Linear(dim_hidden_list[i], dim_hidden_list[i + 1]))
                layers.append(self.activation_fn())
        layers.append(nn.Linear(dim_hidden_list[-1], dim_out))
        
        return nn.Sequential(*layers)

    def forward(self, batch, loss, clip_loss=None):
        
        input = batch.x
        encoded = self.encoder(input)
        output = self.decoder(encoded)
        loss = self.get_loss(input, output, loss)
        
        outs = AttrDict()
        outs.input = input
        outs.output = output
        outs.encoded = encoded
        outs.loss = loss

        return outs

    def get_loss(self, input, output, loss):

        loss_fn = {"mse": nn.MSELoss, "l1": nn.L1Loss}.get(loss)(reduce="mean")
        loss = loss_fn(input, output)

        return loss

    def encode(self, x):
        if type(x) != torch.Tensor:
            return self.encoder(torch.tensor(x, dtype=next(self.parameters()).dtype).to(next(self.parameters()).device)).cpu().numpy()
        else:
            return self.encoder(x)
