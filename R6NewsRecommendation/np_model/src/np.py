import torch
import torch.nn as nn
from torch.distributions import kl_divergence
from attrdict import AttrDict

from util.misc import stack, logmeanexp
from util.metric import *
from util.base_config import BaseConfig
from model.src.modules import PoolingEncoder, Decoder


class NPModelConfig(BaseConfig):
    def set_subconfig_class_hash(self):
        self.subconfig_class_hash = {
            "name": str,
            "dim_x": int,
            "dim_y": int,
            "dim_hid": int,
            "enc_pre_depth": int,
            "enc_post_depth": int,
            "dec_depth": int,
            "ckpt": str
        }


class NPModel(nn.Module):
    def __init__(self,
            dim_x=1,
            dim_y=1,
            dim_hid=128,
            dim_lat=128,
            enc_pre_depth=4,
            enc_post_depth=2,
            dec_depth=3):

        super().__init__()
        self.name = "NP"

        self.denc = PoolingEncoder(
                dim_x=dim_x,
                dim_y=dim_y,
                dim_hid=dim_hid,
                pre_depth=enc_pre_depth,
                post_depth=enc_post_depth)

        self.lenc = PoolingEncoder(
                dim_x=dim_x,
                dim_y=dim_y,
                dim_hid=dim_hid,
                dim_lat=dim_lat,
                pre_depth=enc_pre_depth,
                post_depth=enc_post_depth)

        self.dec = Decoder(
                dim_x=dim_x,
                dim_y=dim_y,
                dim_enc=dim_hid+dim_lat,
                dim_hid=dim_hid,
                depth=dec_depth)

    def predict(self, xc, yc, xt, z=None, num_samples=None, num_bs=None):
        theta = stack(self.denc(xc, yc), num_samples)
        if z is None:
            pz = self.lenc(xc, yc)
            z = pz.rsample() if num_samples is None \
                    else pz.rsample([num_samples])
        encoded = torch.cat([theta, z], -1)
        encoded = stack(encoded, xt.shape[-2], -2)
        dist = self.dec(encoded, stack(xt, num_samples))
        outs = AttrDict()
        outs.loc = dist.loc
        outs.scale = dist.scale
        return outs

    def forward(self, batch, num_samples=1, reduce_ll=True, loss="nll", clip_loss=.0):
        outs = AttrDict()
        Ns = num_samples
        Nc = batch.xc.size(1)
        if "w" in batch.keys():
            if batch.w is None:
                mask = torch.ones([Ns, *batch.y.shape], dtype=torch.float32).to(batch.x.device)
            else:
                mask = stack(batch.w, (Ns or 1), 0).to(batch.x.device)  # [Ns,B,N,Dy]
        else:
            mask = torch.ones([Ns, *batch.y.shape], dtype=torch.float32).to(batch.x.device)
        mask_c = mask[:, :, :Nc, :]
        mask_t = mask[:, :, Nc:, :]

        if self.training:
            pz = self.lenc(batch.xc, batch.yc)
            qz = self.lenc(batch.x, batch.y)
            z = qz.rsample() if num_samples is None else qz.rsample([num_samples])
            preds = self.predict(batch.xc, batch.yc, batch.x,
                                z=z, num_samples=num_samples)
            py = Normal(preds.loc, preds.scale)

            if num_samples > 1:
                # K * B * N
                # recon = py.log_prob(stack(batch.y, num_samples)).sum(-1)  # [Ns, B, N]
                recon = (py.log_prob(stack(batch.y, num_samples)) * mask).sum(-1)  # [Ns, B, N]
                # K * B
                log_qz = qz.log_prob(z).sum(-1)
                log_pz = pz.log_prob(z).sum(-1)

                # K * B
                log_w = recon.sum(-1) + log_pz - log_qz  # [Ns, B]

                outs.loss = -logmeanexp(log_w).mean() / batch.x.shape[-2]
            else:
                outs.recon = (py.log_prob(batch.y) * mask).sum(-1).mean()
                outs.kld = kl_divergence(qz, pz).sum(-1).mean()
                outs.loss = -outs.recon + outs.kld / batch.x.shape[-2]

        else:
            preds = self.predict(batch.xc, batch.yc, batch.x, num_samples=num_samples)
            py = Normal(preds.loc, preds.scale)
            if num_samples is None:
                ll = (py.log_prob(batch.y) * mask).sum(-1)
            else:
                y = torch.stack([batch.y]*num_samples)
                if reduce_ll:
                    ll = logmeanexp((py.log_prob(y) * mask).sum(-1))
                else:
                    ll = (py.log_prob(y) * mask).sum(-1)
            if reduce_ll:
                outs.ctx_loss = ll[...,:Nc].mean()
                outs.tar_loss = ll[...,Nc:].mean()
                outs.loss = outs.ctx_loss + outs.tar_loss
            else:
                outs.ctx_loss = ll[...,:Nc]
                outs.tar_loss = ll[...,Nc:]
                outs.loss = outs.ctx_loss + outs.tar_loss

        mu_hat = py.mean
        sigma_hat = py.scale
        mu_hat_c, mu_hat_t = mu_hat[...,:Nc, :], mu_hat[..., Nc:, :]
        sigma_hat_c, sigma_hat_t = sigma_hat[..., :Nc, :], sigma_hat[..., Nc:, :]
        weight_c, weight_t = mask[:, :, :Nc, :], mask[:, :, Nc:, :]
        outs.ctx_ll = compute_nll(mu_hat_c, sigma_hat_c, batch.yc, None, 1e-3, mask_c).mean()
        outs.tar_ll = compute_nll(mu_hat_t, sigma_hat_t, batch.yt, None, 1e-3, mask_t).mean()

        return outs
