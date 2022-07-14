# UCB Algorithm with Neural Processes (NP-UCB) for Contextual Bandits

This repository contains the codes for my master thesis paper, *"Empirical Study on Contextual Bandit Algorithms with Neural Processes"*.
All the works are done when I am a graduate student of Graduate School of Artificial Intelligence (AIGS), Ulsan National Institue of Science and Technology (UNIST), and a member of Learning Intelligence Machine Laboratory (LIM Lab).

`ContextualBandits/` contains the wheel bandit experiment, while `R6NewsRecommendation/` contains news recommendation experiment.
Each experiment needs two configuration files.
One of them sets the default configuration for a group of experiments, while the other overrides the default configuration to apply different conditions for each experiments in a same group.
`script/train/example/` and `script/eval/example/` contains example experiments for training and evaluation, respectively.
`script/official` contains all the experiments reported in my thesis.

In detail, `script/official/{experiment_group}/` contains `default.yaml` and `exp_config/{experiment_name}.yaml`.

At `ContextualBandits/` or `R6NewsRecommendation/`, run experiment with 
`python main.py --project {project_name_for_logging_in_wandb} --task {train_or_eval} --default_config script/official/{experiment_group}/default.yaml --exp_config script/official/{experiment_group}/exp_config/{experiment_name}.yaml {--any_other_args...}`.
If you want to override some of the configurations when running the code, give them as arguments.

