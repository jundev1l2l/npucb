import os
import json
import pprint


if __name__ == "__main__":
    exp_list = [
        "attn/all_logit",
        "attn/sum_logit",
        "attn/sum_qk",
        "train/banp/balance",
        "train/banp/nll",
        "train/banp/size",
        "train/np/balance",
        "train/np/nll",
        "train/np/size",
    ]

    print("Following Exp Configs Migration Completed:\n")
    for exp in exp_list:
        script_file = f"_bash_script/{exp}/settings.json"
        target_file = f"experiment/script/{exp}/setting.json"
        os.makedirs(f"experiment/script/{exp}", exist_ok=True)
        with open(script_file, "r") as f:
            config_dict = json.load(f)
        config = config_dict["settings"]
        with open(target_file, "w") as f:
            pprint.pprint(config, f, indent=2, compact=False)  # should replace ' to " in json files
        print(target_file)
