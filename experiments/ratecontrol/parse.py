import os
import json
import glob

def load(exp_str):
    alg = exp_str.split("_")[0]

    exp_str_pattern = "~/ray_results/" + alg + "/" + exp_str + "*"
    path = glob.glob(os.path.expanduser(exp_str_pattern))
    if len(path) != 1:
        raise ValueError("ambiguous path")
    path = path[0]
    res = path + "/result.json"
    return res

if __name__ == "__main__":

    exp_list = [
        "FILL_HERE",
    ]

    multi_data = []
    data_lengths = []

    for i, exp in enumerate(exp_list):
        res = load(exp)

        multi_data.append([])
        timesteps = []

        with open(res) as f:
            for l in f.readlines():
                data = json.loads(l)
                ts, rew = data["timesteps_total"], data["episode_reward_mean"]
                # ts, rew = data["timesteps_total"], data["custom_metrics"]["queue_mean"]
                timesteps.append(ts)
                multi_data[i].append(rew)

        data_lengths.append(len(timesteps))
    
    length = min(data_lengths)

    for j in range(length):
        print(timesteps[j], end=",")
        for i, _ in enumerate(exp_list):
            print(multi_data[i][j], end=",")
        print()