from argparse import ArgumentParser, Namespace


def get_argument() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument("--project", type=str, default="R6NewsRecommendation")
    parser.add_argument("--task", type=str, choices=["train", "eval",], required=True)

    # config
    parser.add_argument("--default_config", type=str, required=True)
    parser.add_argument("--exp_config", type=str, default="")

    # general
    parser.add_argument("--result", type=str)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--wandb_mode", type=str)
    parser.add_argument("--seed", type=int)

    # engine
    parser.add_argument("--engine.name", type=str)
    parser.add_argument("--engine.gpu", type=int, nargs="+")
    parser.add_argument("--engine.init_method", type=str)
    parser.add_argument("--engine.backend", type=str)

    # trainer
    parser.add_argument("--trainer.name", type=str,
                        choices=["Trainer", "NPTrainer"])
    parser.add_argument("--trainer.lr", type=float)
    parser.add_argument("--trainer.num_epochs", type=int)
    parser.add_argument("--trainer.loss", type=str,
                        choices=["nll", "l2", "bce"])
    parser.add_argument("--trainer.clip_loss", type=float)
    parser.add_argument("--trainer.val_freq", type=int)
    parser.add_argument("--trainer.save_freq", type=int)

    # evaluator
    parser.add_argument("--evaluator.name", type=str)
    parser.add_argument("--evaluator.n_exp", type=int)
    parser.add_argument("--evaluator.learn_ratio", type=float)
    parser.add_argument("--evaluator.plot_comparison_shape", type=str)

    # model
    parser.add_argument("--model.name", type=str, choices=[
                        "mlp", "np", "anp", "bnp", "banp",])
    parser.add_argument("--model.size", type=str, choices=["base", "large",])

    # data_sampler
    parser.add_argument("--data_sampler.day", type=int, nargs="+")
    parser.add_argument("--data_sampler.reward_balance", type=float,
                        help="Ratio of Split Data with Reward 1")

    # dataset
    parser.add_argument("--dataset.train.day", nargs="+", type=int)
    parser.add_argument("--dataset.val.day", nargs="+", type=int)
    parser.add_argument("--dataset.n_ctx", type=int)
    parser.add_argument("--dataset.n_tar", type=int)
    parser.add_argument("--dataset.path", type=str)

    # dataloader
    parser.add_argument("--dataloader.name", type=str)
    parser.add_argument("--dataloader.batch_size", type=int)

    args = parser.parse_args()

    given_args = {}
    for key, value in vars(args).items():
        if value is not None:
            given_args[key] = value

    return Namespace(**given_args)



if __name__ == "__main__":
    from util.base_config import BaseConfig
    args = get_argument()
    config = BaseConfig(args.__dict__.pop("default_config"))
    print(config)
    config.update(args.__dict__.pop("exp_config"))
    print("config updated\n")
    print(config)
    config.update(args.__dict__)
    print("config updated\n")
    print(config)
