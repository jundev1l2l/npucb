import os

num_split = 4
start = 4
task = "attn"
model_list = [
    "all_logit",
    "sum_logit",
    "sum_qk"
]
shell_list = []
for model in model_list:
    sh_files = []
    for dir in sorted(os.listdir(f"/Users/junhyun/projects/npucb/script/{task}/{model}")):
        if os.path.splitext(dir)[1] == ".sh":
            sh_files.append(f"sh script/{task}/{model}/" + dir)
    shell_list.append(sh_files)
shell_list = sum(shell_list, [])
num_exp = len(shell_list)
buffer = []
exp_per_split = num_exp // num_split
for i, shell in enumerate(shell_list):
    gpu = i // exp_per_split + start
    shell = shell + f" {gpu}"
    buffer.append(shell)
    if len(buffer) == exp_per_split:
        print("; ".join(buffer))
        buffer = []
print(f"Total Exp #: {len(shell_list)}")
