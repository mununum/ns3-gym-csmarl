import os
import ray
from ray.rllib import rollout

# import ns3_multiagent_env
import ns3_rnn_model

if __name__ == "__main__":
    parser = rollout.create_parser()

    checkpoint = ""  # complex, random flow
    # checkpoint = "~/ray_results/PPO/PPO_ns3_multiagent_env_69e97584_2020-04-15_23-57-236gqgmiy1/checkpoint_1000/checkpoint-1000"  # complex, fixed flow
    # checkpoint = "~/ray_results/PPO/PPO_ns3_multiagent_env_29858a2e_2020-04-15_14-51-33unb49qu6/checkpoint_490/checkpoint-490"  # FIM
    checkpoint = "~/ray_results/PPO/PPO_ns3_multiagent_env_e62c78f4_2020-04-18_00-12-4895pljofk/checkpoint_510/checkpoint-510"  # FC
    checkpoint = os.path.expanduser(checkpoint)
    args = parser.parse_args(["--run=PPO", "--no-render", checkpoint])

    simTime = 60
    stepTime = 0.02
    nSteps_per_episode = simTime / stepTime
    args.steps = nSteps_per_episode * 10
    args.config = {
        "num_workers": 0,
        # "num_gpus_per_worker": 1,
        "env_config": {
            "simTime": simTime,
            # "debug": True,
            "topology": "complex",
            "fixedFlow": True,
        },
        "multiagent": {
            "policies_to_train": ["nothing"]
        }
    }

    rollout.run(args, parser)