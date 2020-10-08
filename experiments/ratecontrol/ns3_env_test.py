import ray
from ray import tune

import ns3_multiagent_env

if __name__ == "__main__":
    
    ray.init(local_mode=True)

    env_config = {
        "debug": True,
        "topology": "fim",
    }

    config = {
        "env": "ns3_multiagent_env",
        "batch_mode": "complete_episodes",
        "log_level": "DEBUG",
        "num_workers": 0,
        "env_config": env_config,
    }

    stop = {
        "training_iteration": 10,
    }

    tune.run("PPO", config=config, stop=stop)