from data_sampler.r6_data import R6DataSampler, R6DataSamplerConfig


DATA_SAMPLER_DICT = {
    "R6DataSampler": R6DataSampler,
}


DATA_SAMPLER_CONFIG_DICT = {
    "R6DataSampler": R6DataSamplerConfig,
}


def build_data_sampler(config, *args, **kwargs):

    DATA_SAMPLER = DATA_SAMPLER_DICT[config.name]

    data_sampler = DATA_SAMPLER(config, *args, **kwargs)

    return data_sampler
