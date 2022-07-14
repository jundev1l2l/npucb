import wandb

from config.train_config import TrainConfig
from util.log import get_logger, init_wandb
from util.misc import set_seed
from engine import build_engine
from data_sampler import build_data_sampler
from dataset import build_dataset
from dataloader import build_dataloader
from model import build_model
from trainer import build_trainer


def train_worker(rank: int, config: TrainConfig):
    
    # build logger
    logger = get_logger(
        task=config.task,
        log_dir=config.result,
        debug=config.debug
    )
    init_wandb(
        project=config.project,
        log_dir=config.result,
        config_dict=config(),
        mode=config.wandb_mode,
    ) if rank <= 0 else None
    
    # build engine
    if rank >= 0:
        logger.info("Build Engine...") if rank <= 0 else None
        logger.info("") if rank <= 0 else None
        build_engine(config.engine, rank)
    logger.info("") if rank <= 0 else None
    logger.info("=" * 30) if rank <= 0 else None
    logger.info("Start News Recommendation Bandit Experiment: Training") if rank <= 0 else None
    logger.info("") if rank <= 0 else None

    # build experiment
    set_seed(config.seed)
    
    logger.info("Build Model...") if rank <= 0 else None
    logger.info("") if rank <= 0 else None
    model = build_model(
        config=config.model,
        rank=rank,
        debug=config.debug,
    )

    logger.info("Build Data Sampler...") if rank <= 0 else None
    logger.info("") if rank <= 0 else None
    data_sampler = build_data_sampler(
        config=config.data_sampler,
        logger=logger,
    )
    
    logger.info("Build Dataset...") if rank <= 0 else None
    logger.info("") if rank <= 0 else None
    dataset_dict = build_dataset(
        config=config.dataset,
        data_sampler=data_sampler,
        debug=config.debug,
    )

    logger.info("Build Dataloader...") if rank <= 0 else None
    logger.info("") if rank <= 0 else None
    dataloader_dict = build_dataloader(
        config=config.dataloader,
        dataset_dict=dataset_dict,
        use_distributed=(rank >= 0),
    )

    logger.info("Build Trainer...") if rank <= 0 else None
    logger.info("") if rank <= 0 else None
    trainer = build_trainer(
        config=config.trainer,
        model=model,
        dataloader_dict=dataloader_dict,
        logger=logger,
        save_dir=config.result,
        rank=rank,
        debug=config.debug,
    )

    # run experiment
    logger.info("Start Training...") if rank <= 0 else None
    logger.info("") if rank <= 0 else None
    
    trainer.train()

    logger.info(f"Training Finished") if rank <= 0 else None
    logger.info(f"Results Saved at {config.result}") if rank <= 0 else None
    if config.wandb_mode == "online":
        logger.info(f"Results Uploaded at https://wandb.ai/{wandb.run.entity}/{wandb.run.project}/runs/{wandb.run.id}") if rank <= 0 else None
    wandb.finish() if rank <= 0 else None
