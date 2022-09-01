import os, sys
import pytest
import numpy as np
import pybullet as p

from pathlib import Path

import gym
from stable_baselines3.common.utils import set_random_seed

# todo: write test that ensures reward = zero if pose = target pose
path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, path)

from somo_rl.utils import parse_config
from somo_rl.utils.import_environment import import_env


def test_PenSpinner_reward():

    # initialize the environment
    env_name = "PenSpinner"
    run_config_path = Path(os.path.dirname(__file__)) / f"../tests/assets/test_configs/{env_name}.yaml"
    run_config = parse_config.validate_config(run_config_path)

    # overwrite reward flags to only use object pose in reward
    run_config["reward_flags"] = {"orientation": -1, "position": -1}
    env_id = run_config["env_id"]

    # prepare env
    import_env(env_id)
    env = gym.make(env_id, run_config=run_config, render=False)
    set_random_seed(run_config["seed"])
    env.seed(run_config["seed"])
    env.reset()

    # take 10 random steps
    for _ in range(10):

        # sample a random action
        action = env.action_space.sample()

        # overwrite target positions with object positions
        box_pos, box_or_quat = p.getBasePositionAndOrientation(env.object_id)

        env.env.target_position, env.env.target_orientation = np.array(
            box_pos
        ), np.array(box_or_quat)

        assert (
            env.env.get_reward(action) == 0
        ), f"reward wasn't 0, even tho target and object pose match"

        # take random action
        env.step(action)
    env.close()


if __name__ == "__main__":
    test_PenSpinner_reward()
