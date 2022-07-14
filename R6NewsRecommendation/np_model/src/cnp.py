import torch
import torch.nn as nn
from attrdict import AttrDict

from util.metric import *
from model.src.modules import PoolingEncoder, Decoder


class CNP(nn.Module):
    def __init__(self,
            dim_x=1,
            dim_y=1,
            dim_hid=128,
            enc_pre_depth=4,
            enc_post_depth=2,
            dec_depth=3):

        super().__init__()

        self.enc1 = PoolingEncoder(
                dim_x=dim_x,
                dim_y=dim_y,
                dim_hid=dim_hid,
                pre_depth=enc_pre_depth,
                post_depth=enc_post_depth)

        self.enc2 = PoolingEncoder(
                dim_x=dim_x,
                dim_y=dim_y,
                dim_hid=dim_hid,
                pre_depth=enc_pre_depth,
                post_depth=enc_post_depth)

        self.dec = Decoder(
                dim_x=dim_x,
                dim_y=dim_y,
                dim_enc=2*dim_hid,
                dim_hid=dim_hid,
                depth=dec_depth)

    def predict(self, xc, yc, xt, num_samples=None):
        encoded = torch.cat([self.enc1(xc, yc), self.enc2(xc, yc)], -1)  # [B,2Eh]
        encoded = torch.stack([encoded]*xt.shape[-2], -2)  # [B,N,2Eh]
        return self.dec(encoded, xt)  # Normal([B,N,Dy=5])

    def forward(self, batch, num_samples=None, reduce_ll=True):
        outs = AttrDict()
        num_ctx = batch.xc.shape[-2]
        py = self.predict(batch.xc, batch.yc, batch.x)  # Normal([B,N,1])
        ll = py.log_prob(batch.y).sum(-1)  # [B,N]

        if self.training:
            outs.loss = -ll.mean()
        else:
            num_ctx = batch.xc.shape[-2]  # Nc
            if reduce_ll:
                outs.ctx_loss = ll[...,:num_ctx].mean()  # [1,]
                outs.tar_loss = ll[...,num_ctx:].mean()  # [1,]
            else:
                outs.ctx_loss = ll[...,:num_ctx]  # [B,Nc]
                outs.tar_loss = ll[...,num_ctx:]  # [B,Nt]

        Nc = num_ctx
        mu_hat = py.mean
        sigma_hat = py.scale
        mu_hat_c = mu_hat[:, :Nc, :]  # [B,Nc,Dy]
        mu_hat_t = mu_hat[:, Nc:, :]  # [B,Nt,Dy]
        sigma_hat_c = sigma_hat[:, :Nc, :]  # [B,Nc,Dy]
        sigma_hat_t = sigma_hat[:, Nc:, :]  # [B,Nct,Dy]
        y = batch.y
        y_c = y[:, :Nc, :]  # [B,Nc,Dy]
        y_t = y[:, Nc:, :]  # [B,Nt,Dy]
        outs.ctx_ll = compute_nll(mu_hat_c, sigma_hat_c, y_c).mean()
        outs.tar_ll = compute_nll(mu_hat_t, sigma_hat_t, y_t).mean()

        print("In Model: input size", batch.xc.size(),
              "output size", outs.ctx_loss.size())

        return outs
        # {"loss": [1,]} while training
        # {"ctx_ll": [1,], "tar_ll": [1,]} while evaluating (if reduce_ll = True)
