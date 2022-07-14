import os
import sys
import torch.multiprocessing as mp

sys.path.insert(0, "/home/junhyun/projects/npucb/R6NewsRecommendation")

from config import get_argument, build_config
from worker import build_worker
from util.write_config import get_config_dict_from_args
from util.log import dump_config


def main():

    args = get_argument()
    args_dict = get_config_dict_from_args(args)
    
    config = build_config(args.task, args.default_config, args.exp_config)
    config.update(args_dict)
    if config.debug:
        config.result = os.path.join(config.result, "debug")
    dump_config(args_dict, config)
    
    worker = build_worker(args.task)

    if config.engine.gpu[0] < 0:
            worker(-1, config)
    else:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config.engine.gpu)[1:-1]
        os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
        os.environ["WANDB_START_METHOD"] = "thread"  # wandb.init(), mp.spawn() 충돌 방지
        ngpus_per_node = len(config.engine.gpu)
        mp.spawn(worker, nprocs=ngpus_per_node, args=(config,))


if __name__ == "__main__":
    main()
