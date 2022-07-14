import os
import itertools
from shutil import copyfile

model_list = ["banp"]
exp_list = ["balance", "nll", "size"]
save_dir = "/Users/junhyun/data/R6/result/train/"
file_list = ["train_curve.png", "both_curves.png"]
for model, exp, file in itertools.product(model_list, exp_list, file_list):
    os.makedirs(os.path.join(save_dir, f"summary_{exp}"), exist_ok=True)
    path = os.path.join(save_dir, model, exp)
    settings = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
    for setting in settings:
        exp_name = f"{model.upper()}-{setting}-{file}"
        if os.path.isfile(os.path.join(path, setting, file)):
            copyfile(os.path.join(path, setting, file), os.path.join(save_dir, f"summary_{exp}", exp_name))
