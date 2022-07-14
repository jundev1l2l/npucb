import os
import math
import random
import torch
import numpy as np

from importlib.machinery import SourceFileLoader
from time import time
from datetime import timedelta


def find_free_port():
    """ https://stackoverflow.com/questions/1365265/on-localhost-how-do-i-pick-a-free-port-number """
    import socket
    from contextlib import closing

    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return str(s.getsockname()[1])


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def wallclock(fn):
    start_time = time()
    fn()
    end_time = time()
    time_str = f"{str(timedelta(seconds=end_time-start_time)):0>8}"
    return time_str


def to_cuda(x_list):
    if isinstance(x_list, list):
        return list(map(lambda x: x.cuda(), x_list))
    else:
        return x_list.detach().cpu()


def to_cpu(x_list):
    if isinstance(x_list, list):
        return list(map(lambda x: x.detach().cpu(), x_list))
    else:
        return x_list.detach().cpu()


def gen_load_func(parser, func):
    def load(args, cmdline):
        sub_args, cmdline = parser.parse_known_args(cmdline)
        for k, v in sub_args.__dict__.items():
            args.__dict__[k] = v
        return func(**sub_args.__dict__), cmdline
    return load


def load_module(filename):
    module_name = os.path.splitext(os.path.basename(filename))[0]
    return SourceFileLoader(module_name, filename).load_module()
    # <module "module_name" from "filename">
    #
    # ex.
    # <module "cnp" from "backbone/cnp.py">


def logmeanexp(x, dim=0):
    return x.logsumexp(dim) - math.log(x.shape[dim])


def stack(x, num_samples=None, dim=0):
    return x if num_samples is None \
            else torch.stack([x]*num_samples, dim=dim)


def hrminsec(duration):
    hours, left = duration // 3600, duration % 3600
    mins, secs = left // 60, left % 60
    return f"{hours}hrs {mins}mins {secs}secs"


def one_hot(x, num):  # [B,N] -> [B,N,num]
    B, N = x.shape
    _x = torch.zeros([B, N, num], dtype=torch.float32)
    for b in range(B):
        for n in range(N):
            i = x[b, n]
            _x[b, n, i] = 1.0
    return _x


def get_circum_points(r, n=100):
    return np.array([(math.cos(2 * math.pi / n * x) * r, math.sin(2 * math.pi / n * x) * r) for x in range(0, n + 1)])
