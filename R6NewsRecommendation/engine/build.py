from engine.ddp_engine import init_ddp_engine, DDPEngineConfig


ENGINE_DICT = {
    "DDPEngine": init_ddp_engine,
}

ENGINE_CONFIG_DICT = {
    "DDPEngine": DDPEngineConfig,
}


def build_engine(
    config: DDPEngineConfig,
    rank: int,
) -> None:

    ENGINE = ENGINE_DICT[config.name]
    
    ENGINE(config,rank)
