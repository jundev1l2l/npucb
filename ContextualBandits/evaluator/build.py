from evaluator.base_evaluator import BaseEvaluator, BaseEvaluatorConfig
from evaluator.new_arm_evaluator import NewArmEvaluator, NewArmEvaluatorConfig
from evaluator.new_env_evaluator import NewEnvEvaluator, NewEnvEvaluatorConfig


EVALUATOR_DICT = {
        "BaseEvaluator": BaseEvaluator,
        "NewArmEvaluator": NewArmEvaluator,
        "NewEnvEvaluator": NewEnvEvaluator,
}

EVALUATOR_CONFIG_DICT = {
        "BaseEvaluator": BaseEvaluatorConfig,
        "NewArmEvaluator": NewArmEvaluatorConfig,
        "NewEnvEvaluator": NewEnvEvaluatorConfig,
}


def build_evaluator(config, *args, **kwargs):
    EVALUATOR = EVALUATOR_DICT[config.name]
    
    evaluator = EVALUATOR(
        config=config,
        *args,
        **kwargs,
    )

    return evaluator
