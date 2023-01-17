import gym
import gym_stag_hunt
from ray import tune
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from gym_stag_hunt.envs.pettingzoo.hunt import raw_env

if __name__ == "__main__":
    def env_creator(args):
        return PettingZooEnv(raw_env(**args))

    tune.register_env("StagHunt-Hunt-PZ-v0", env_creator)

    model = tune.run(
        "DQN",
        name="stag_hunt",
        stop={"episodes_total": 10000},
        checkpoint_freq=100,
        checkpoint_at_end=True,
        config={
            "horizon": 100,
            "framework": "tf2",
            # Environment specific
            "env": "StagHunt-Hunt-PZ-v0",
            # General
            "num_workers": 2,
            # Method specific
            "multiagent": {
                "policies": {"player_0", "player_1"},
                "policy_mapping_fn": (lambda agent_id, episode, **kwargs: agent_id),
                "policies_to_train": ["player_0", "player_1"]
            },
            # Env Specific
            "env_config": {
                "obs_type": "coords",
                "forage_reward": 1.0,
                "stag_reward": 5.0,
                "stag_follows": True,
                "mauling_punishment": -.5,
                "enable_multiagent": True,
            }
        }
    )