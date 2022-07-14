import torch
import torch.nn as nn
from torch.distributions import kl_divergence
from attrdict import AttrDict

from util.misc import stack, logmeanexp
from util.metric import *
from util.base_config import BaseConfig
from model.src.modules import CrossAttnEncoder, PoolingEncoder, Decoder


class ANPModelConfig(BaseConfig):
    def set_subconfig_class_hash(self):
        self.subconfig_class_hash = {
            "name": str,
            "dim_x": int,
            "dim_y": int,
            "dim_hid": int,
            "enc_v_depth": int,
            "enc_qk_depth": int,
            "enc_pre_depth": int,
            "enc_post_depth": int,
            "dec_depth": int,
            "ckpt": str
        }


class ANPModel(nn.Module):
    def __init__(self,
            dim_x=1,
            dim_y=1,
            dim_hid=128,
            dim_lat=128,
            enc_v_depth=4,
            enc_qk_depth=2,
            enc_pre_depth=4,
            enc_post_depth=2,
            dec_depth=3):

        super().__init__()
        self.name = "ANP"

        self.denc = CrossAttnEncoder(
                dim_x=dim_x,
                dim_y=dim_y,
                dim_hid=dim_hid,
                v_depth=enc_v_depth,
                qk_depth=enc_qk_depth)

        self.lenc = PoolingEncoder(
                dim_x=dim_x,
                dim_y=dim_y,
                dim_hid=dim_hid,
                dim_lat=dim_lat,
                self_attn=True,
                pre_depth=enc_pre_depth,
                post_depth=enc_post_depth)

        self.dec = Decoder(
                dim_x=dim_x,
                dim_y=dim_y,
                dim_enc=dim_hid+dim_lat,
                dim_hid=dim_hid,
                depth=dec_depth)

    def predict(self, xc, yc, xt, z=None, num_samples=None):
        theta = stack(self.denc(xc, yc, xt), num_samples)
        if z is None:
            pz = self.lenc(xc, yc)
            z = pz.rsample() if num_samples is None \
                    else pz.rsample([num_samples])
        z = stack(z, xt.shape[-2], -2)
        encoded = torch.cat([theta, z], -1)
        return self.dec(encoded, stack(xt, num_samples))

    def forward(self, batch, num_samples=1, reduce_ll=True, loss="nll", clip_loss=.0):
        outs = AttrDict()
        num_ctx = batch.xc.shape[-2]
        if self.training:
            pz = self.lenc(batch.xc, batch.yc)
            qz = self.lenc(batch.x, batch.y)
            z = qz.rsample() if num_samples is None else \
                    qz.rsample([num_samples])
            py = self.predict(batch.xc, batch.yc, batch.x,
                    z=z, num_samples=num_samples)

            if num_samples > 1:
                # K * B * N
                recon = py.log_prob(stack(batch.y, num_samples)).sum(-1)
                # K * B
                log_qz = qz.log_prob(z).sum(-1)
                log_pz = pz.log_prob(z).sum(-1)

                # K * B
                log_w = recon.sum(-1) + log_pz - log_qz

                outs.loss = -logmeanexp(log_w).mean() / batch.x.shape[-2]
            else:
                outs.recon = py.log_prob(batch.y).sum(-1).mean()
                outs.kld = kl_divergence(qz, pz).sum(-1).mean()
                outs.loss = -outs.recon + outs.kld / batch.x.shape[-2]

        else:
            py = self.predict(batch.xc, batch.yc, batch.x, num_samples=num_samples)
            if num_samples is None:
                ll = py.log_prob(batch.y).sum(-1)
            else:
                y = torch.stack([batch.y]*num_samples)
                if reduce_ll:
                    ll = logmeanexp(py.log_prob(y).sum(-1))
                else:
                    ll = py.log_prob(y).sum(-1)

            if reduce_ll:
                outs.ctx_loss = - ll[...,:num_ctx].mean()
                outs.tar_loss = - ll[...,num_ctx:].mean()
                outs.loss = outs.ctx_loss + outs.tar_loss
            else:
                outs.ctx_loss = - ll[...,:num_ctx]
                outs.tar_loss = - ll[...,num_ctx:]
                outs.loss = outs.ctx_loss + outs.tar_loss

        Nc = num_ctx
        mu_hat = py.mean
        sigma_hat = py.scale
        mu_hat_c = mu_hat[:, :, :Nc, :]  # [Ns,B,Nc,Dy]
        mu_hat_t = mu_hat[:, :, Nc:, :]  # [Ns,B,Nt,Dy]
        sigma_hat_c = sigma_hat[:, :, :Nc, :]  # [Ns,B,Nc,Dy]
        sigma_hat_t = sigma_hat[:, :, Nc:, :]  # [Ns,B,Nct,Dy]
        y = batch.y
        y_c = y[:, :Nc, :]  # [B,Nc,Dy]
        y_t = y[:, Nc:, :]  # [B,Nt,Dy]

        outs.ctx_ll = compute_nll(mu_hat_c, sigma_hat_c, y_c).mean()
        outs.tar_ll = compute_nll(mu_hat_t, sigma_hat_t, y_t).mean()

        return outs
    