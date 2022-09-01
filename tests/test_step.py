import os
import sys
import pytest

import gym
from stable_baselines3.common.utils import set_random_seed

from pathlib import Path

# todo: add this path addition to the scripts that are run from within somo_rl
path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, path)

from somo_rl.utils import parse_config
from somo_rl.utils.import_environment import import_env


def somogym_step_tester(
    env_name,
    render=False,
    debug=False,
    total_env_steps=5,
):
    run_config_path = Path(os.path.dirname(__file__)) / f"../tests/assets/test_configs/{env_name}.yaml"
    run_config = parse_config.validate_config(run_config_path)
    env_id = run_config["env_id"]

    # prepare env
    import_env(env_id)
    env = gym.make(env_id, run_config=run_config, render=render, debug=debug)
    set_random_seed(run_config["seed"])
    env.seed(run_config["seed"])
    env.reset()

    # run env for total_env_steps steps
    for _ in range(total_env_steps):
        env.step(env.action_space.sample())  # take a random action

    # make sure seeding works correctly for this env
    # seed once, reset, and take a step
    set_random_seed(run_config["seed"])
    env.seed(run_config["seed"])
    env.reset()
    action_a = env.action_space.sample()
    step_result_a = env.step(action_a)  # take a random action
    # seed and reset again and take another step
    set_random_seed(run_config["seed"])
    env.seed(run_config["seed"])
    env.reset()
    action_b = env.action_space.sample()
    step_result_b = env.step(action_b)  # take a random action

    # compare results
    assert (
        step_result_a[0] == step_result_b[0]
    ).all(), f"seeding does not work correctly for env {env_name}: observations are inconsistent"
    assert (
        step_result_a[1] == step_result_b[1]
    ), f"seeding does not work correctly for env {env_name}: rewards are inconsistent"
    assert (
        step_result_a[2] == step_result_b[2]
    ), f"seeding does not work correctly for env {env_name}: done flags are inconsistent"
    assert (
        step_result_a[3] == step_result_b[3]
    ), f"seeding does not work correctly for env {env_name}: info entries are inconsistent"

    # finally, close the env
    env.close()
    

# ANTIPODAL GRIPPER
def test_AntipodalGripper_step():
    somogym_step_tester("AntipodalGripper")
@pytest.mark.gui
def test_AntipodalGripper_step_gui():
    somogym_step_tester("AntipodalGripper", render=True, debug=True, total_env_steps=100)

# IN-HAND MANIPULATION
def test_InHandManipulation_step():
    somogym_step_tester("InHandManipulation")
@pytest.mark.gui
def test_InHandManipulation_step_gui():
    somogym_step_tester("InHandManipulation", render=True, debug=True, total_env_steps=100)

# IN-HAND MANIPULATION INVERTED
def test_InHandManipulationInverted_step():
    somogym_step_tester("InHandManipulationInverted")
@pytest.mark.gui
def test_InHandManipulationInverted_step_gui():
    somogym_step_tester("InHandManipulationInverted", render=True, debug=True, total_env_steps=100)

# PEN SPINNER
def test_PenSpinner_step():
    somogym_step_tester("PenSpinner")
@pytest.mark.gui
def test_PenSpinner_step_gui():
    somogym_step_tester("PenSpinner", render=True, debug=True, total_env_steps=100)

# PEN SPINNER FAR
def test_PenSpinnerFar_step():
    somogym_step_tester("PenSpinnerFar")
@pytest.mark.gui
def test_PenSpinnerFar_step_gui():
    somogym_step_tester("PenSpinnerFar", render=True, debug=True, total_env_steps=100)

# PLANAR BLOCK PUSHING
def test_PlanarBlockPushing_step():
    somogym_step_tester("PlanarBlockPushing")
@pytest.mark.gui
def test_PlanarBlockPushing_step_gui():
    somogym_step_tester("PlanarBlockPushing", render=True, debug=True, total_env_steps=100)

# PLANAR REACHING
def test_PlanarReaching_step():
    somogym_step_tester("PlanarReaching")
@pytest.mark.gui
def test_PlanarReaching_step_gui():
    somogym_step_tester("PlanarReaching", render=True, debug=True, total_env_steps=100)

# SNAKE LOCOMOTION CONTINUOUS (CONTINUOUS LOCOSNAKE DOES NOT WORK YET)
# def test_SnakeLocomotionContinuous_step():
#     somogym_step_tester("SnakeLocomotionContinuous")
# @pytest.mark.gui
# def test_SnakeLocomotionContinuous_step_gui():
#     somogym_step_tester("SnakeLocomotionContinuous", render=True, debug=True, total_env_steps=100)

# SNAKE LOCOMOTION DISCRETE
def test_SnakeLocomotionDiscrete_step():
    somogym_step_tester("SnakeLocomotionDiscrete")
@pytest.mark.gui
def test_SnakeLocomotionDiscrete_step_gui():
    somogym_step_tester("SnakeLocomotionDiscrete", render=True, debug=True, total_env_steps=100)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "gui":
        test_AntipodalGripper_step_gui()
        test_InHandManipulation_step_gui()
        test_InHandManipulationInverted_step_gui()
        test_PenSpinner_step_gui()
        test_PenSpinnerFar_step_gui()
        test_PlanarBlockPushing_step_gui()
        test_PlanarReaching_step_gui()
        # test_SnakeLocomotionContinuous_step_gui()
        test_SnakeLocomotionDiscrete_step_gui()
    else:
        test_AntipodalGripper_step()
        test_InHandManipulation_step()
        test_InHandManipulationInverted_step()
        test_PenSpinner_step()
        test_PenSpinnerFar_step()
        test_PlanarBlockPushing_step()
        test_PlanarReaching_step()
        # test_SnakeLocomotionContinuous_step()
        test_SnakeLocomotionDiscrete_step()
