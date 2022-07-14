from evaluator.base_evaluator import BaseEvaluator, BaseEvaluatorConfig
from evaluator.r6_evaluator import R6Evaluator, R6EvaluatorConfig


EVALUATOR_DICT = {
        "BaseEvaluator": BaseEvaluator,
        "R6Evaluator": R6Evaluator,        
}

EVALUATOR_CONFIG_DICT = {
        "BaseEvaluator": BaseEvaluatorConfig,
        "R6Evaluator": R6EvaluatorConfig,
}


def build_evaluator(config, *args, **kwargs):
    EVALUATOR = EVALUATOR_DICT[config.name]
    
    evaluator = EVALUATOR(
        config=config,
        *args,
        **kwargs,
    )

    return evaluator
