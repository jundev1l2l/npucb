from data_sampler.mushroom_data_sampler import MushroomDataSampler, MushroomDataSamplerConfig
from data_sampler.stock_data_sampler import StockDataSampler, StockDataSamplerConfig
from data_sampler.statlog_data_sampler import StatlogDataSampler, StatlogDataSamplerConfig
from data_sampler.jester_data_sampler import JesterDataSampler, JesterDataSamplerConfig
from data_sampler.covertype_data_sampler import CovertypeDataSampler, CovertypeDataSamplerConfig
from data_sampler.adult_data_sampler import AdultDataSampler, AdultDataSamplerConfig
from data_sampler.census_data_sampler import CensusDataSampler, CensusDataSamplerConfig
from data_sampler.wheel_data_sampler import WheelDataSampler, WheelDataSamplerConfig
from data_sampler.gp_data_sampler import GPDataSampler, GPDataSamplerConfig


DATA_SAMPLER_DICT = {
    "MushroomDataSampler": MushroomDataSampler,
    "StockDataSampler": StockDataSampler,
    "StatlogDataSampler": StatlogDataSampler,
    "JesterDataSampler": JesterDataSampler,
    "CovertypeDataSampler": CovertypeDataSampler,
    "AdultDataSampler": AdultDataSampler,
    "CensusDataSampler": CensusDataSampler,
    "WheelDataSampler": WheelDataSampler,
    "GPDataSampler": GPDataSampler,
}


DATA_SAMPLER_CONFIG_DICT = {
    "MushroomDataSampler": MushroomDataSamplerConfig,
    "StockDataSampler": StockDataSamplerConfig,
    "StatlogDataSampler": StatlogDataSamplerConfig,
    "JesterDataSampler": JesterDataSamplerConfig,
    "CovertypeDataSampler": CovertypeDataSamplerConfig,
    "AdultDataSampler": AdultDataSamplerConfig,
    "CensusDataSampler": CensusDataSamplerConfig,
    "WheelDataSampler": WheelDataSamplerConfig,
    "GPDataSampler": GPDataSamplerConfig,
}


def build_data_sampler(config):

    if isinstance(config, list):
        config_list = config
        data_sampler_list = []

        for temp_config in config_list:
            data_sampler_list.append(build_data_sampler(temp_config))
        
        return data_sampler_list

    elif config.name == "WheelDataSampler" and isinstance(config.delta, list):
        delta_list = config.delta
        data_sampler_list = []

        for delta in delta_list:
            temp_config = config
            temp_config.delta = delta
            data_sampler_list.append(build_data_sampler(temp_config))
        
        return data_sampler_list

    else:
        DATA_SAMPLER = DATA_SAMPLER_DICT[config.name]

        data_sampler = DATA_SAMPLER(config)

        return data_sampler
