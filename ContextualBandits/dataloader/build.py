from torch.utils.data.distributed import DistributedSampler

from dataloader.torch_dataloader import TorchDataloader, TorchDataloaderConfig


DATALOADER_DICT = {
    "TorchDataloader": TorchDataloader,
}


DATALOADER_CONFIG_DICT = {
    "TorchDataloader": TorchDataloaderConfig,
}


def build_dataloader(config, dataset_dict, use_distributed):
    
    DATALOADER = DATALOADER_DICT[config.name]

    dataloader_dict = {}
    for mode, dataset in dataset_dict.items():
        if dataset.name in ["NPDataset", "NPRandomDataset", "GPRandomDataset",]:
            dataset.batch_size = config.batch_size
        sampler = DistributedSampler(dataset) if use_distributed else None
        dataloader = DATALOADER(
            batch_size=config.batch_size,
            num_workers=config.num_workers if hasattr(config, "num_workers") else 0,
            dataset=dataset, 
            sampler=sampler,
            drop_last=True,
        )
        dataloader_dict[mode] = dataloader
        
    return dataloader_dict
