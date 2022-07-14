import wandb

from config.eval_config import EvalConfig
from util.log import get_logger, init_wandb
from util.misc import set_seed
from engine import build_engine
from data_sampler import build_data_sampler
from bandit_env import build_bandit_env
from bandit_algo import build_bandit_algo_list
from feature_extractor import build_feature_extractor
from evaluator import build_evaluator


def eval_worker(rank: int, config: EvalConfig):

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
        build_engine(config.engine, rank)  # todo: make this method returns Engine instance "engine", and use engine.initiate()
    
    logger.info("") if rank <= 0 else None
    logger.info("=" * 30) if rank <= 0 else None
    logger.info("Start Contextual Bandit Benchmarks Experiment: Evaluation") if rank <= 0 else None
    logger.info("") if rank <= 0 else None

    # build experiment
    set_seed(config.seed + 1)

    logger.info("Build Data Sampler...") if rank <= 0 else None
    logger.info("") if rank <= 0 else None
    data_sampler = build_data_sampler(
        config=config.data_sampler,
    )

    logger.info("Build Bandit Env...") if rank <= 0 else None
    logger.info("") if rank <= 0 else None
    bandit_env = build_bandit_env(
        config=config.bandit_env,
        data_sampler=data_sampler,
        logger=logger,
    )

    if hasattr(config, "feature_extractor"):
        logger.info("Build Feature Extractor...") if rank <= 0 else None
        logger.info("") if rank <= 0 else None
        feature_extractor = build_feature_extractor(
            config=config.feature_extractor,
            rank=rank,
            debug=config.debug,
        )
    else:
        feature_extractor = None
    
    logger.info("Build Bandit Algos...") if rank <= 0 else None
    logger.info("") if rank <= 0 else None
    bandit_algo_baseline, bandit_algo_eval_list = build_bandit_algo_list(
        config_list=config.bandit_algo_list,
        num_arms=bandit_env[0].num_arms if isinstance(bandit_env, list) else bandit_env.num_arms,
        dim_context=bandit_env[0].dim_context if isinstance(bandit_env, list) else bandit_env.dim_context,
        feature_extractor=feature_extractor,
        rank=rank,
        debug=config.debug,
    )
    
    logger.info("Build Evaluator...") if rank <= 0 else None
    logger.info("") if rank <= 0 else None
    evaluator = build_evaluator(
        config=config.evaluator,
        bandit_algo_baseline=bandit_algo_baseline,
        bandit_algo_eval_list=bandit_algo_eval_list,
        bandit_env=bandit_env,
        logger=logger,
        save_dir=config.result,
        rank=rank,
        debug=config.debug,
    )

    # run experiment
    logger.info("Start Evaluation...") if rank <= 0 else None
    logger.info("") if rank <= 0 else None
    
    evaluator.eval()

    logger.info("") if rank <= 0 else None
    logger.info(f"Evaluation Finished") if rank <= 0 else None
    logger.info(f"Results Saved at {config.result}") if rank <= 0 else None
    if config.wandb_mode == "online":
        logger.info(f"Results Uploaded at https://wandb.ai/{wandb.run.entity}/{wandb.run.project}/runs/{wandb.run.id}") if rank <= 0 else None
    wandb.finish() if rank <= 0 else None
