import os
import glob
import json
import ray
# from ray.rllib import rollout
import myrollout as rollout

import ns3_rnn_model

if __name__ == "__main__":

    # FC-3
    # exp_str = "PPO_ns3_multiagent_env_40db3a12"
    # FC-6
    # exp_str = "PPO_ns3_multiagent_env_4237dad2"
    # FC-9
    # exp_str = "PPO_ns3_multiagent_env_4456bcf2"
    # FC-12
    exp_str = "PPO_ns3_multiagent_env_4b0b8136"
    # FIM
    # exp_str = "PPO_ns3_multiagent_env_6e1d89ce"

    exp_str_pattern = "~/ray_results/PPO/" + exp_str + "*"
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
        # select the checkpoint with the best reward
        r = [int(f.split('/')[-1].split('_')[-1]) for f in checkpoint_files]
        chpno = min(r)
    else:
        print("many checkpoints")
        exit()
    
    checkpoint = path+"/checkpoint_"+str(chpno)+"/checkpoint-"+str(chpno)

    with open(path + "/params.json") as f:
        params = json.loads(f.read())
        topo_inherit = params["env_config"]["topology"]

    parser = rollout.create_parser()
    args = parser.parse_args(["--run=PPO", "--no-render", checkpoint])

    simTime = 20
    stepTime = 0.005
    nSteps_per_episode = simTime / stepTime
    iterations = 1
    args.steps = nSteps_per_episode * iterations
    args.config = {
        "num_workers": 0,
        # "num_gpus_per_worker": 1,
        "env_config": {
            "simTime": simTime,
            "debug": False,
            "topology": topo_inherit,  # replace to test generalization
            "randomFlow": False,
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
    rollout.run(args, parser)

    # multi seed experiment
    # for i in range(3):
    #     args.config["env_config"]["seed"] = i + 1
    #     rollout.run(args, parser)

    # intensity experiment
    # intensity = 0.1
    # for i in range(10):
    #     args.config["env_config"]["intensity"] = intensity
    #     rollout.run(args, parser)
    #     intensity += 0.1