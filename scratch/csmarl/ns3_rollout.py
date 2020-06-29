import os
import glob
import json
import ray
# from ray.rllib import rollout
import myrollout as rollout

import ns3_rnn_model
import ns3_rnn_centralized

def load_checkpoint(exp_str, pick="best"):

    # FC-3
    # exp_str = "PPO_ns3_multiagent_env_40db3a12"
    # FC-6
    # exp_str = "PPO_ns3_multiagent_env_4237dad2"
    # FC-9
    # exp_str = "PPO_ns3_multiagent_env_4456bcf2"
    # FC-12
    # exp_str = "PPO_ns3_multiagent_env_4b0b8136"
    # FIM
    # exp_str = "PPO_ns3_multiagent_env_6e1d89ce"
    # FIM-RC, ns3_rnn_model, lr=5e-4
    # exp_str = "PPO_ns3_multiagent_env_8a8cb833"
    # FIM-RC, ns3_rnn_centralized, lr=5e-4
    # exp_str = "CCPPOTrainer_ns3_multiagent_env_7b4731bd"
    # FIM-RC, ns3_rnn_centralized, lr=5e-5
    # exp_str = "CCPPOTrainer_ns3_multiagent_env_7b4731be"
    # FIM-RC, ns3_rnn_centralized, queue+utility, lr=5e-5
    # FIM-RC, ns3_rnn_centralized, queue+utility, lr=5e-4
    # exp_str = "CCPPOTrainer_ns3_multiagent_env_a2f8d38a"

    # FIM-RC, ns3_rnn_model, utility, lr=5e-4
    # exp_str = "PPO_ns3_multiagent_env_80ffd84e"

    # FIM-RC, PPO, utility, not using qlen for state, lr=5e-4
    # exp_str = "PPO_ns3_multiagent_env_e63749ee"

    # FIM-RC, CCPPO, utility, not using qlen, lr=5e-4
    # exp_str = "CCPPOTrainer_ns3_multiagent_env_344cb16a"

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

    # with open(path + "/params.json") as f:
    #     params = json.loads(f.read())
    #     topo_inherit = params["env_config"]["topology"]

    # exp_str = "PPO_ns3_multiagent_env_a9f2aac4"
    # randomIntensity=true
    # exp_str = "PPO_ns3_multiagent_env_678248d6"
    # exp_str = "PPO_ns3_multiagent_env_2243e67d"

    # exp_str = "PPO_ns3_multiagent_env_c5915e5a"

    # exp_str = "PPO_ns3_multiagent_env_2f1a1b76"

    # exp_str = "PPO_ns3_multiagent_env_14caf1bf"
    # exp_str = "PPO_ns3_multiagent_env_bd15beee"

    # exp_str = "PPO_ns3_multiagent_env_49d5c152"
    # exp_str = "PPO_ns3_multiagent_env_1ffb6178"
    exp_str = "PPO_ns3_multiagent_env_e6620174"

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
        "env_config": {
            "simTime": simTime,
            "debug": True,
            # "topology": "complex",  # uncomment for generalization
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
    # _, args.checkpoint = load_checkpoint(exp_str, pick="best")
    # rollout.run(args, parser)

    # multi seed experiment
    # rollout.run(args, parser)
    # for i in range(10):
    #     args.config["env_config"]["seed"] = i + 1
    #     rollout.run(args, parser)

    # intensity experiment
    intensity = 0.1
    for i in range(10):
        args.config["env_config"]["intensity"] = intensity
        _, args.checkpoint = load_checkpoint(exp_str, pick="best")
        rollout.run(args, parser)
        intensity += 0.1
