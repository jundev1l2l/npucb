import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from model.src.attention import MultiHeadAttn, SelfAttn


__all__ = ['PoolingEncoder', 'CrossAttnEncoder', 'Decoder']


def build_mlp(dim_in, dim_hid, dim_out, depth):
    modules = [nn.Linear(dim_in, dim_hid), nn.ReLU(True)]
    for _ in range(depth-2):
        modules.append(nn.Linear(dim_hid, dim_hid))
        modules.append(nn.ReLU(True))
    modules.append(nn.Linear(dim_hid, dim_out))
    return nn.Sequential(*modules)


class PoolingEncoder(nn.Module):
    """
    structure: net_pre -> aggregation -> net_post. 논문 그림과 달리 구현체에서는 encoder에 aggregation이 포함되어 있음.

    dim_lat: NP에서 latent variable z의 차원
    self_attn: ANP, BANP에서 attention 사용
    """
    def __init__(self, dim_x=1, dim_y=1,
            dim_hid=128, dim_lat=None, self_attn=False,
            pre_depth=4, post_depth=2):
        super().__init__()

        self.use_lat = dim_lat is not None

        self.net_pre = build_mlp(dim_x+dim_y, dim_hid, dim_hid, pre_depth) \
                if not self_attn else \
                nn.Sequential(
                        build_mlp(dim_x+dim_y, dim_hid, dim_hid, pre_depth-2),
                        nn.ReLU(True),
                        SelfAttn(dim_hid, dim_hid))

        self.net_post = build_mlp(dim_hid, dim_hid,
                2*dim_lat if self.use_lat else dim_hid,
                post_depth)

    def forward(self, xc, yc, mask=None):
        out = self.net_pre(torch.cat([xc, yc], -1))  # [B,N,Eh]
        if mask is None:
            out = out.mean(-2)  # [B,Eh]
        else:
            mask = mask.to(xc.device)
            out = (out * mask.unsqueeze(-1)).sum(-2) / \
                    (mask.sum(-1, keepdim=True).detach() + 1e-5)
        if self.use_lat:
            mu, sigma = self.net_post(out).chunk(2, -1)
            sigma = 0.1 + 0.9 * torch.sigmoid(sigma)
            return Normal(mu, sigma)
        else:
            return self.net_post(out)  # [B,Eh]


class CrossAttnEncoder(nn.Module):
    def __init__(self, dim_x=1, dim_y=1, dim_hid=128,
            dim_lat=None, self_attn=True,
            v_depth=4, qk_depth=2):
        super().__init__()
        self.use_lat = dim_lat is not None

        dim_v = dim_x + dim_y

        if not self_attn:
            self.net_v = build_mlp(dim_v, dim_hid, dim_hid, v_depth)
        else:
            self.net_v = build_mlp(dim_v, dim_hid, dim_hid, v_depth-2)
            self.self_attn = SelfAttn(dim_hid, dim_hid)

        self.net_qk = build_mlp(dim_x, dim_hid, dim_hid, qk_depth)

        self.attn = MultiHeadAttn(dim_hid, dim_hid, dim_hid,
                2*dim_lat if self.use_lat else dim_hid)

    def forward(self, xc, yc, xt, w=None, mask=None):
        q, k = self.net_qk(xt), self.net_qk(xc)

        v = self.net_v(torch.cat([xc, yc], -1))

        if hasattr(self, 'self_attn'):
            v = self.self_attn(v, mask=mask)

        out = self.attn(q, k, v, mask=mask)
        if self.use_lat:
            mu, sigma = out.chunk(2, -1)
            sigma = 0.1 + 0.9 * torch.sigmoid(sigma)
            return Normal(mu, sigma)
        else:
            return out


class Decoder(nn.Module):
    def __init__(self, dim_x=1, dim_y=1,
            dim_enc=128, dim_hid=128, depth=3):
        super().__init__()
        self.fc = nn.Linear(dim_x+dim_enc, dim_hid)
        self.dim_hid = dim_hid

        modules = [nn.ReLU(True)]
        for _ in range(depth-2):
            modules.append(nn.Linear(dim_hid, dim_hid))
            modules.append(nn.ReLU(True))
        modules.append(nn.Linear(dim_hid, 2*dim_y))
        self.mlp = nn.Sequential(*modules)

    def add_ctx(self, dim_ctx):
        self.dim_ctx = dim_ctx
        self.fc_ctx = nn.Linear(dim_ctx, self.dim_hid, bias=False)

    def forward(self, encoded, x, ctx=None):
        """
        encoded: [B,(Nbs,)Nt,2Eh]
        x: [B,(Nbs,)Nt,Dx]
        """
        packed = torch.cat([encoded, x], -1)  # [B,(Nbs,)Nt,2Eh+Dx]
        hid = self.fc(packed)  # [B,(Nbs,)Nt,Dh]
        if ctx is not None:
            hid = hid + self.fc_ctx(ctx)  # [B,(Nbs,)Nt,Dh]
        out = self.mlp(hid)  # [B,(Nbs,)Nt,2Dy]
        
        mu, sigma = out.chunk(2, -1)  # [B,Nt,Dy] each
        sigma = 0.1 + 0.9 * F.softplus(sigma)
        # sigma = F.softplus(sigma)
        return Normal(mu, sigma)  # Normal([B,Nt,Dy])
        