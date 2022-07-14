with open("util/latex/delta=0.50_reward_prediction_BANP.txt", "r") as f:
    lines = f.readlines()

for arm in [0,1,2,3,4]:
    for step in [4000, 8000, 12000, 16000, 20000]:
        new_lines = []
        for line in lines:
            if line.startswith("\\label"):
                new_lines.append(line.strip("}\n") + f"{step}_{arm}" + "}\n")
            elif line.startswith("\\includegraphics"):
                new_lines.append(line.strip(".png}\n") + f"{step}_{arm}" + ".png}\n")
            else:
                new_lines.append(line)
        print("".join(new_lines))
