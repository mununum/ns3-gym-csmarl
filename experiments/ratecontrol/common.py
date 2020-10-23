import GPUtil

from ray.rllib.utils import merge_dicts
from ns3_multiagent_env import MyCallbacks

def config(args):

    env_config = {
        "topology": args.topology
    }

    gpus = GPUtil.getGPUs()
    NUM_GPUS = len(gpus)
    assert NUM_GPUS > 0
    # NUM_GPUS = 2
    num_workers = 8
    num_gpus_per_worker = NUM_GPUS / num_workers
    if env_config["topology"] == "fim":
        timesteps_total = 5e7
    elif env_config["topology"] == "random":
        timesteps_total = 5e8
    else:
        timesteps_total = 3e8

    config = {
        "env": "ns3_multiagent_env",
        "batch_mode": "complete_episodes",
        "log_level": "WARN",
        "num_workers": num_workers,
        "num_gpus_per_worker": num_gpus_per_worker,
        "sgd_minibatch_size": 2000,
        "lr_schedule": [[0, 5e-5], [timesteps_total, 0.0]],
        "env_config": env_config,
        "callbacks": MyCallbacks,
    }

    if env_config["topology"] == "random":
        # add evaluation config
        num_eval_workers = 4
        num_gpus_per_worker = NUM_GPUS / (num_workers + num_eval_workers)
        eval_config = {
            "evaluation_interval": 1,
            "evaluation_num_workers": num_eval_workers,
            "evaluation_num_episodes": num_eval_workers,
            "evaluation_config": {
                "env_config": {"topology": "complex"},
                "explore": False,
            },
            "num_gpus_per_worker": num_gpus_per_worker,
        }
        config = merge_dicts(config, eval_config)

    if args.debug:
        env_config["simTime"] = 1
        config["log_level"] = "DEBUG"
        config["num_workers"] = 0

    stop = {
        "timesteps_total": timesteps_total,
    }

    return config, stop