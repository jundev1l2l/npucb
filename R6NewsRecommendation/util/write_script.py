import os
import sys
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "vit"))


def write_script(task, config_path):
    default_config = os.path.join(config_path, "default_config.yaml")
    exp_config_path = os.path.join(config_path, "exp_config")

    if os.path.exists(exp_config_path):
        exp_config_list = os.listdir(exp_config_path)
        script = ""
        for file in exp_config_list:
            exp_config = os.path.join(exp_config_path, file)
            script += f"python main.py \\\n"
            script += f"--task {task} \\\n"
            script += f"--default_config {default_config} \\\n"
            script += f"--exp_config {exp_config} \\\n"
            script += f"--engine.gpu $* ;\n"
            script += f"\n"
        num_exp = len(exp_config_list)
    else:
        script = f"python main.py \\\n"
        script += f"--task {task} \\\n"
        script += f"--default_config {default_config} \\\n"
        script += f"--engine.gpu $* ;\n"
        script += f"\n"
        num_exp = 1

    script_path = os.path.join(config_path, "script.sh")
    with open(script_path, "w") as f:
        f.write(script)
    print(f"Script written at {script_path}. (# of exp: {num_exp})")
    print("-" * 100)


if __name__ == "__main__":
    from script.train.expid_list import task, exp_group_list
    for exp_group in exp_group_list:
        script_path = f"script/{task}/{exp_group}"
        write_script(task, script_path)

    from script.eval.expid_list import task, exp_group_list
    for exp_group in exp_group_list:
        script_path = f"script/{task}/{exp_group}"
        write_script(task, script_path)
