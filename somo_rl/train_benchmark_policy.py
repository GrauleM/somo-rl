import os
import sys
import argparse
from pathlib import Path
path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, path)
from copy import deepcopy

from user_settings import EXPERIMENT_ABS_PATH
from somo_rl.train_policy import run as run_training
from somo_rl.utils import parse_config


def run_benchmark(
    exp_abs_path,
    exp_name,
    run_group_name,
    run_name,
    render=False,
    debug=False,
    overwrite=False,
    note="",
    run_config_abs_path=None,
):

    run_dir = Path(exp_abs_path) / exp_name / run_group_name / run_name

    if run_config_abs_path is None:
        run_config_file = run_dir / "run_config.yaml"
    else:
        run_config_file = Path(run_config_abs_path)

    run_config = parse_config.construct_benchmark_config(run_config_file)

    if not overwrite:
        if "run_started" in run_config.keys() and run_config["run_started"]:
            print(f"CRITICAL WARNING: Run {self.run_ID} already started/complete. Edit 'run_started' config field to allow overwrite.")
            return 1

        with open(run_config_file, "a") as config_file:
            config_file.write("\nrun_started: True")

    if not run_config:
        print('ERROR: Run "' + run_name + '" invalid run config')
        return 1

    run_training(
        exp_abs_path=exp_abs_path,
        exp_name=exp_name,
        run_group_name=run_group_name,
        run_name=run_name,
        render=render,
        debug=debug,
        note=note,
        run_config_input=run_config
    )

    env_name = run_config["env_id"].split('-')[0]
    print(f"\nSUCCESS: Finished training policy using {env_name} benchmark! I hope it worked well!\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training run argument parser")
    parser.add_argument(
        "-e",
        "--exp_name",
        help="Experiment name",
        required=True,
        default=None,
    )
    parser.add_argument(
        "-g",
        "--run_group_name",
        help="Run-group name",
        required=True,
        default=None,
    )
    parser.add_argument(
        "-r",
        "--run_name",
        help="Run Name",
        required=True,
        default=None,
    )
    parser.add_argument(
        "-v",
        "--render",
        help="Render the env of one of the threads",
        action="store_true",
    )
    parser.add_argument(
        "-d", "--debug", help="Display SoMo-RL Debugging Dashboard", action="store_true"
    )
    parser.add_argument('-dl','--debug_list', nargs='+', help='List of debugger components to show in panel (space separated). Choose from reward_components, observations, actions, applied_torques', required=False, default=[])

    parser.add_argument(
        "-o",
        "--overwrite",
        help="Allow overwrite of data of previous experiment",
        action="store_true",
    )
    parser.add_argument(
        "--note", help="Note to keep track of purpose of run", default=""
    )
    argument = parser.parse_args()

    debug = argument.debug 
    if len(argument.debug_list) > 0:
        debug = deepcopy(argument.debug_list)
        
    run_benchmark(
        EXPERIMENT_ABS_PATH,
        argument.exp_name,
        argument.run_group_name,
        argument.run_name,
        argument.render,
        debug,
        argument.overwrite,
        argument.note,
    )
