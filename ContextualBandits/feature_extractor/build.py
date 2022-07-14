import torch

from feature_extractor.autoencoder_feature_extractor import AutoencoderFeatureExtractor, AutoencoderFeatureExtractorConfig
from util.custom_ddp import CustomDDP

from util.load import remove_module_from_dict


FEATURE_EXTRACTOR_DICT = {
    "AutoencoderFeatureExtractor": AutoencoderFeatureExtractor,
}

FEATURE_EXTRACTOR_CONFIG_DICT = {
    "AutoencoderFeatureExtractor": AutoencoderFeatureExtractorConfig,
}


def build_feature_extractor(config, rank, debug):
    
    FEATURE_EXTRACTOR = FEATURE_EXTRACTOR_DICT[config.name]
    feature_extractor = FEATURE_EXTRACTOR(config)

    if hasattr(config, "ckpt"):
        if debug:
            ckpt_path_list = config.ckpt.split("/")
            ckpt_path_list.insert(-1, "debug")
            ckpt_file = "/".join(ckpt_path_list)
        else:
            ckpt_file = config.ckpt
        feature_extractor_dict = torch.load(ckpt_file, map_location="cuda" if rank >= 0 else "cpu")
        feature_extractor_dict = remove_module_from_dict(feature_extractor_dict)
        feature_extractor.load_state_dict(feature_extractor_dict)

    feature_extractor.eval()

    if rank >= 0:
        feature_extractor = feature_extractor.cuda(rank)
        feature_extractor = CustomDDP(
            feature_extractor,
            device_ids=[rank],
            find_unused_parameters=True
        )

    return feature_extractor
