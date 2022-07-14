import torch
import torch.nn as nn
from attrdict import AttrDict

from util.misc import stack, logmeanexp
from util.sampling import sample_with_replacement as SWR
from util.metric import *
from util.base_config import BaseConfig
from model.src.canp import CANP


class BANPModelConfig(BaseConfig):
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
        

class BANPModel(CANP):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "BANP"
        self.dec.add_ctx(2*kwargs['dim_hid'])

    def encode(self, xc, yc, xt, mask=None):
        theta1 = self.enc1(xc, yc, xt)  # [Ns,B,N,Eh]
        theta2 = self.enc2(xc, yc)  # [Ns,B,Eh]
        encoded = torch.cat([theta1,
            torch.stack([theta2]*xt.shape[-2], -2)], -1)
        return encoded  # [Ns,B,N,2Eh]

    def predict(self, xc, yc, xt, num_samples=None, return_base=False, loss="nll", clip_loss=.0):
        with torch.no_grad():
            bxc, byc = SWR(xc, yc, num_samples=num_samples, reduce=False)  # [Ns,B,Nc,D]
            sxc, syc = stack(xc, num_samples), stack(yc, num_samples)  # [Ns,B,Nc,D]

            encoded = self.encode(bxc, byc, sxc)  # [Ns,B,Nc,2Eh]
            py_res = self.dec(encoded, sxc)  # [Ns,B,Nc,D]

            mu, sigma = py_res.mean, py_res.scale  # [Ns,B,Nc,D]
            res = SWR((syc - mu)/sigma).detach()  # [Ns,B,Nc,D]
            res = (res - res.mean(-2, keepdim=True))

            bxc = sxc
            byc = mu + sigma * res

        encoded_base = self.encode(xc, yc, xt)  # [B,Nt,D]

        sxt = stack(xt, num_samples)  # [Ns,B,Nt,D]
        encoded_bs = self.encode(bxc, byc, sxt)  # [B,Nt,2Eh]

        py = self.dec(stack(encoded_base, num_samples),  # [Ns,B,Nt,D]
                sxt, ctx=encoded_bs)

        if self.training or return_base:
            py_base = self.dec(encoded_base, xt)  # [B,Nc,D]
            return py_base, py
        else:
            return py

    def forward(self, batch, num_samples=1, reduce_ll=True, loss="nll", clip_loss=.0):
        outs = AttrDict()
        num_ctx = batch.xc.shape[-2]
        weight = None
        # weight = torch.stack([batch.w] * (num_samples or 1), 0).to(batch.x.device)  # [Ns,B,N,Dy]

        def compute_ll(py, y, w=None):
            if w is None:
                w = torch.ones(y.shape, dtype=torch.float32).to(batch.x.device)
            ll = (py.log_prob(y) * w).sum(-1)
            if ll.dim() == 3 and reduce_ll:
                ll = logmeanexp(ll)
            return ll

        if self.training:
            py_base, py = self.predict(batch.xc, batch.yc, batch.x,
                    num_samples=num_samples)

            outs.ll_base = compute_ll(py_base, batch.y, weight).mean()
            outs.ll = compute_ll(py, batch.y, weight).mean()
            outs.loss = - outs.ll_base - outs.ll
        else:
            py = self.predict(batch.xc, batch.yc, batch.x,
                    num_samples=num_samples)
            ll = compute_ll(py, batch.y, weight)
            if reduce_ll:
                outs.ctx_loss = ll[...,:num_ctx].mean()
                outs.tar_loss = ll[...,num_ctx:].mean()
                outs.loss = outs.ctx_loss + outs.tar_loss
            else:
                outs.ctx_loss = ll[...,:num_ctx]
                outs.tar_loss = ll[...,num_ctx:]
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
