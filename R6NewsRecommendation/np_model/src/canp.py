import torch
import torch.nn as nn
from attrdict import AttrDict

from util.metric import *
from model.src.modules import CrossAttnEncoder, Decoder, PoolingEncoder


class CANP(nn.Module):
    def __init__(self,
            dim_x=1,
            dim_y=1,
            dim_hid=128,
            enc_v_depth=4,
            enc_qk_depth=2,
            enc_pre_depth=4,
            enc_post_depth=2,
            dec_depth=3):

        super().__init__()

        self.enc1 = CrossAttnEncoder(
                dim_x=dim_x,
                dim_y=dim_y,
                dim_hid=dim_hid,
                v_depth=enc_v_depth,
                qk_depth=enc_qk_depth)

        self.enc2 = PoolingEncoder(
                dim_x=dim_x,
                dim_y=dim_y,
                dim_hid=dim_hid,
                self_attn=True,
                pre_depth=enc_pre_depth,
                post_depth=enc_post_depth)

        self.dec = Decoder(
                dim_x=dim_x,
                dim_y=dim_y,
                dim_enc=2*dim_hid,
                dim_hid=dim_hid,
                depth=dec_depth)

    def predict(self, xc, yc, xt, num_samples=None):
        theta1 = self.enc1(xc, yc, xt)  # [B,Nt,Eh]
        theta2 = self.enc2(xc, yc)  # [B,Eh]
        encoded = torch.cat([theta1,
            torch.stack([theta2]*xt.shape[-2], -2)], -1)  # [B,Nt,2Eh]
        return self.dec(encoded, xt)

    def forward(self, batch, num_samples=None, reduce_ll=True):
        outs = AttrDict()
        num_ctx = batch.xc.shape[-2]
        py = self.predict(batch.xc, batch.yc, batch.x)
        ll = py.log_prob(batch.y).sum(-1)

        if self.training:
            outs.loss = - ll.mean()
        else:
            num_ctx = batch.xc.shape[-2]
            if reduce_ll:
                outs.ctx_loss = ll[...,:num_ctx].mean()
                outs.tar_loss = ll[...,num_ctx:].mean()
            else:
                outs.ctx_loss = ll[...,:num_ctx]
                outs.tar_loss = ll[...,num_ctx:]

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

        return outs
