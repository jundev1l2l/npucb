import torch
import torch.distributed as dist

from util.base_config import BaseConfig


class DDPEngineConfig(BaseConfig):
    def set_subconfig_class_hash(self):
        self.subconfig_class_hash = {
            "name": str,
            "gpu": list,
            "init_method": str,
            "backend": str,
        }


def init_ddp_engine(config, rank):
    gpu_groups = config.gpu
    ngpus_per_node = len(gpu_groups)

    print("Use GPU: {}".format(gpu_groups[rank]))
    print(f"Init Method: {config.init_method}")
    print("")
    dist.init_process_group(backend=config.backend, init_method=config.init_method,
                            world_size=ngpus_per_node, rank=rank)

    torch.cuda.set_device(rank)


def cleanup_engien():
    dist.destroy_process_group()
