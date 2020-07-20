import os
import glob
import json
import ray
# from ray.rllib import rollout
import myrollout as rollout

import ns3_rnn_model
import ns3_rnn_centralized
import ns3_rnn_gnn

def load_checkpoint(exp_str, pick="best"):

    alg = exp_str.split("_")[0]

    exp_str_pattern = "~/ray_results/" + alg + "/" + exp_str + "*"
    path = glob.glob(os.path.expanduser(exp_str_pattern))
    if len(path) != 1:
        print("ambiguous path")
        exit()
    path = path[0]
    chp_pattern = path + "/checkpoint*"
    checkpoint_files = glob.glob(chp_pattern)
    if len(checkpoint_files) == 0:
        print("no checkpoint file")
        exit()
    elif len(checkpoint_files) <= 2:
        r = [int(f.split('/')[-1].split('_')[-1]) for f in checkpoint_files]
        if pick == "best":
            # select the checkpoint with the best reward
            chpno = min(r)
        elif pick == "last":
            # select the last checkpoint
            chpno = max(r)
        else:
            print("invalid pick option")
            exit()
    else:
        print("many checkpoints")
        exit()
    
    checkpoint = path+"/checkpoint_"+str(chpno)+"/checkpoint-"+str(chpno)

    return alg, checkpoint

if __name__ == "__main__":

    exp_str = "FILL_HERE"

    alg, checkpoint = load_checkpoint(exp_str, pick="last")

    parser = rollout.create_parser()
    args = parser.parse_args(["--run=" + alg, "--no-render", checkpoint])

    simTime = 20
    stepTime = 0.005
    nSteps_per_episode = simTime / stepTime
    iterations = 1
    args.steps = nSteps_per_episode * iterations
    args.config = {
        "num_workers": 0,
        # "num_gpus_per_worker": 1,
        # "log_level": "DEBUG",
        "env_config": {
            "simTime": simTime,
            "debug": True,
            # "topology": "complex",  # uncomment for transfer
            "randomFlow": False,
            "randomIntensity": False,
            "intensity": 1.0,
            "seed": 0,
        },
        "multiagent": {
            # do not train from the checkpoint, only inference
            "policies_to_train": ["nothing"]
        }
    }

    ray.init()

    # single experiment
    _, args.checkpoint = load_checkpoint(exp_str, pick="best")
    rollout.run(args, parser)

    # multi seed experiment
    # rollout.run(args, parser)
    # for i in range(10):
    #     args.config["env_config"]["seed"] = i + 1
    #     rollout.run(args, parser)

    # intensity experiment
    # intensity = 0.1
    # for i in range(10):
    #     args.config["env_config"]["intensity"] = intensity
    #     _, args.checkpoint = load_checkpoint(exp_str, pick="best")
    #     rollout.run(args, parser)
    #     intensity += 0.1
