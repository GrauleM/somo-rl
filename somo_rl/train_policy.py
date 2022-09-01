import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from copy import deepcopy

import argparse
import numpy as np

from pathlib import Path

from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_util import Monitor
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)

path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, path)

from somo_rl.utils.import_environment import import_env
from somo_rl.utils import parse_config, construct_policy_model

from user_settings import EXPERIMENT_ABS_PATH
import gym


def create_note(run_dir, run_ID, start_datetime, note):
    try:
        git_commit_hash = str(
            subprocess.check_output(["git", "rev-parse", "HEAD"])
        ).strip()[2:-3]
    except:
        git_commit_hash = "unavailable"

    with open(run_dir / "info.txt", "w") as info_file:
        info_file.write("_____________________________________\n")
        info_file.write("TRAINING RUN INFO:\n")
        info_file.write(f"- Run ID: {run_ID}\n")
        info_file.write("- PID: " + str(os.getpid()) + "\n")
        info_file.write("- Start Datetime: " + start_datetime + "\n")
        info_file.write("- Git Commit Hash: " + git_commit_hash + "\n")

        info_file.write("_____________________________________\n")
        info_file.write("NOTES ON EXPERIMENT:\n")
        info_file.write("- " + note + "\n")


def make_env(
    env_id,
    run_config,
    max_episode_steps,
    rank=0,
    run_ID=None,
    monitoring_dir=None,
    render=False,
    debug=False,
    is_eval_env=False,
):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param rank: (int) index of the subprocess
    :param monitoring_dir: (str) directory to store monitor files
    """

    def _init():
        try:
            seed = run_config["seed"] + rank
        except:
            seed = rank
        if is_eval_env:  # set outside reasonable range of # env ranks
            seed += 100
        set_random_seed(seed)
        print(f"Set seed to {seed}.")

        import_env(run_config["env_id"])
        env = gym.make(
            env_id,
            run_config=run_config,
            run_ID=run_ID,
            render=render,
            debug=debug,
        )
        env._max_episode_steps = max_episode_steps
        env.seed(seed)
        if monitoring_dir is not None and not is_eval_env:
            log_file = Path(monitoring_dir) / str(rank)
            reward_keywords = tuple([])
            if run_config["reward_flags"]:
                reward_keywords = tuple(run_config["reward_flags"].keys())
            env = Monitor(env, str(log_file), info_keywords=reward_keywords)
        return env

    return _init


def log_on_complete(
    start_time, run_ID, eval_dir, run_dir, run_config, success=True
):
    # Log training run results
    end_time = time.time()
    end_datetime = datetime.now().strftime("%m/%d/%Y %H:%M:%S")
    training_duration = end_time - start_time

    print(f"Training Duration: {training_duration}")

    try:
        best_eval_reward = np.max(
            np.mean(
                dict(np.load(Path(eval_dir) / "evaluations.npz"))["results"], axis=1
            )
        )
    except:
        best_eval_reward = None

    with open(run_dir / "info.txt", "a") as info_file:
        eval_reward_message = (
            "No eval callbacks generated."
            if best_eval_reward is None
            else best_eval_reward
        )
        info_file.write("_____________________________________\n")
        info_file.write("RESULTS:\n")
        info_file.write("- Success: " + str(success) + "\n")
        info_file.write("- Best Eval Reward: " + str(eval_reward_message) + "\n")
        info_file.write("- End Datetime: " + end_datetime + "\n")
        info_file.write(
            "- Training Duration (sec): " + str(int(np.round(training_duration))) + "\n"
        )
        info_file.write(
            "- Training Timesteps: " + str(run_config["training_timesteps"]) + "\n"
        )

    if success:
        print(
            f"\nSUCCESS! Experiment {run_ID} is done training!\n"
        )
    else:
        print(
            f"\nERROR! Experiment {run_ID} failed on following exception:\n"
        )


def run(
    exp_abs_path,
    exp_name,
    run_group_name,
    run_name,
    render=False,
    debug=False,
    overwrite=False,
    note="",
    run_config_input=None,
    expert_dir_abs_path=None,
):

    run_ID = [exp_name, run_group_name, run_name]

    run_dir = Path(exp_abs_path)
    for subdivision in run_ID:
        run_dir = run_dir / subdivision

    # run_config_input is a run_config dictionary
    if isinstance(run_config_input, dict):
        run_config = deepcopy(run_config_input)
    # run_config_input wasn't given or is a path
    else:
        # use the run_config found in the run directory
        if run_config_input is None:
            run_config_file = run_dir / "run_config.yaml"
        # run_config_input is a path to the run_config.yaml file
        else:
            run_config_file = Path(run_config_input)

        run_config = parse_config.validate_config(run_config_file)

        if not overwrite:
            if "run_started" in run_config.keys() and run_config["run_started"]:
                print(f"CRITICAL WARNING: Run {run_ID} already started/complete. Edit 'run_started' config field to allow overwrite.")
                return 1

            with open(run_config_file, "a") as config_file:
                config_file.write("\nrun_started: True")

    if not run_config:
        print('ERROR: Run "' + run_name + '" invalid run config')
        return 1

    import_env(run_config["env_id"])

    if expert_dir_abs_path:
        run_config["expert_dir_abs_path"] = expert_dir_abs_path

    # Set up logging and results directories
    monitoring_dir = run_dir / "monitoring"
    models_dir = run_dir / "models"
    checkpoints_dir = run_dir / "callbacks" / "checkpoints"
    eval_dir = run_dir / "callbacks" / "eval_results"
    results_dir = run_dir / "results"

    dirs = [monitoring_dir, models_dir, checkpoints_dir, eval_dir, results_dir]

    for dir_path in dirs:
        shutil.rmtree(dir_path, ignore_errors=True)
        os.makedirs(dir_path)

    tensorboard_log = None
    if "tensorboard_log" in run_config.keys():
        tensorboard_log = run_config["tensorboard_log"]

    env_id = run_config["env_id"]

    # Number of processes to use
    if "num_threads" in run_config:
        num_threads = run_config["num_threads"]
    else:
        num_threads = 1

    start_time = time.time()
    start_datetime = datetime.now().strftime("%m/%d/%Y %H:%M:%S")

    create_note(run_dir, run_ID, start_datetime, note)

    # Create the vectorized environment
    train_env = SubprocVecEnv(
        [
            make_env(
                env_id=env_id,
                run_config=run_config,
                max_episode_steps=run_config["max_episode_steps"],
                rank=i,
                run_ID=run_ID,
                monitoring_dir=monitoring_dir,
                render=(render and i == 0),
                debug=debug if i == 0 else False,
            )
            for i in range(num_threads)
        ],
        start_method="forkserver",
    )

    # separate evaluation env
    eval_env = SubprocVecEnv(
        [
            make_env(
                env_id=env_id,
                run_config=run_config,
                max_episode_steps=run_config["max_episode_steps"],
                run_ID = deepcopy(run_ID).append("EVAL_ENV")
            )
        ]
    )

    # create callbacks
    eval_callback = EvalCallback(
        eval_env,
        n_eval_episodes=run_config["eval_cb"]["n_eval_episodes"],
        best_model_save_path=models_dir,
        log_path=eval_dir,
        eval_freq=run_config["eval_cb"]["eval_freq"],
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=run_config["checkpoint_cb"]["save_freq"],
        save_path=checkpoints_dir,
    )
    callback = CallbackList([eval_callback, checkpoint_callback])

    policy_kwargs = {}
    if "policy_kwargs" in run_config:
        policy_kwargs = deepcopy(run_config["policy_kwargs"])

    model = construct_policy_model.construct_policy_model(
        run_config["alg"],
        run_config["policy"],
        train_env,
        policy_kwargs=policy_kwargs,
        tensorboard_log=tensorboard_log,
        verbose=1
    )

    # Run Training
    try:
        model.learn(total_timesteps=run_config["training_timesteps"], callback=callback)
    except Exception as e:
        log_on_complete(
            start_time, run_ID, eval_dir, run_dir, run_config, success=False
        )
        raise e

    log_on_complete(start_time, run_ID, eval_dir, run_dir, run_config)

    model.save(models_dir / "final_model")
    train_env.close()

    return 0


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

    run(
        EXPERIMENT_ABS_PATH,
        argument.exp_name,
        argument.run_group_name,
        argument.run_name,
        argument.render,
        debug,
        argument.overwrite,
        argument.note,
    )
