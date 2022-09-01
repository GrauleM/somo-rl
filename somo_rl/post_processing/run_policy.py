import os
import platform
import shutil
import sys
import time
from datetime import datetime
import argparse
from copy import deepcopy

path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, path)

from somo_rl.utils.import_environment import import_env
from somo_rl.utils import parse_config, construct_policy_model

from user_settings import EXPERIMENT_ABS_PATH

import csv
import gym
import json
import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
from pathlib import Path
import pandas as pd


def extract_step_reward(prev_total_rewards, cur_total_rewards):
    step_rewards = {}
    for key in prev_total_rewards:
        step_rewards[key] = cur_total_rewards[key] - prev_total_rewards[key]
    return step_rewards


class Policy_rollout:
    def __init__(self, exp_abs_path, run_ID, debug=False):
        self.run_ID = run_ID
        self.run_dir = Path(exp_abs_path)
        for subdivision in self.run_ID:
            self.run_dir = self.run_dir / subdivision
        run_config_file = self.run_dir / "run_config.yaml"

        self.run_config = parse_config.validate_config(run_config_file)

        # check if it's a benchmark config
        if "action_time" not in self.run_config:
            self.run_config = parse_config.construct_benchmark_config(run_config_file)

        if not self.run_config:
            raise (Exception, "ERROR: Invalid run config")

        import_env(self.run_config["env_id"])

        self.models_dir = self.run_dir / "models"
        self.checkpoints_callbacks_dir = self.run_dir / "callbacks" / "checkpoints"
        self.select_checkpoints_dir = self.models_dir / "select_checkpoints"

        self.results_dir = self.run_dir / "results"
        self.processed_data_dir = self.results_dir / "processed_data"
        self.videos_dir = self.results_dir / "render_vids"

        now = datetime.now()
        self.datetime = now.strftime("%d-%m-%y_%H-%M")

        for dir_path in [self.processed_data_dir, self.videos_dir]:
            os.makedirs(dir_path, exist_ok=True)

        self.env = gym.make(
            self.run_config["env_id"],
            run_config=self.run_config,
            run_ID=run_ID,
            debug=debug,
        )


    def set_model(self, model, from_callbacks=False):
        if model == "final_model":
            model_file = self.models_dir / "final_model"
        elif model == "best_model":
            model_file = self.models_dir / "best_model"
        elif model == "latest_checkpoint":
            checkpoints_dir = self.checkpoints_callbacks_dir if from_callbacks else self.select_checkpoints_dir
            checkpoint_files = list(
                filter(
                    lambda file: file.endswith("_steps.zip"),
                    os.listdir(checkpoints_dir),
                )
            )
            latest_num = max(
                map(lambda file: int(file.split("_")[-2]), checkpoint_files)
            )
            model_file = checkpoints_dir / (
                "rl_model_" + str(latest_num) + "_steps"
            )
        elif isinstance(model, int):
            checkpoints_dir = self.checkpoints_callbacks_dir if from_callbacks else self.select_checkpoints_dir
            checkpoint_files = list(
                filter(
                    lambda file: file.endswith("_steps.zip"),
                    os.listdir(checkpoints_dir),
                )
            )
            model_file = checkpoints_dir / (
                "rl_model_" + str(model) + "_steps"
            )
            if model_file not in checkpoint_files:
                print("ERROR: invalid checkpoint number.")
                sys.exit(1)
        elif os.path.exists(model):
            model_file = model
        else:
            print("ERROR: invalid input model.")
            sys.exit(1)

        alg = construct_policy_model.ALGS[self.run_config["alg"]]
        self.model = alg.load(model_file)
        return model_file


    def run_rollout(
        self, model="best_model", from_callbacks=False, num_steps=None, run_render=True, save_vid=False, zero_action=False, seed=None, record_data=True
    ):
        self.env_seed = seed if isinstance(seed, int) else np.random.randint(1000)
        self.results_ID = self.datetime + f"_s{self.env_seed}"
        self.env.seed(self.env_seed)
        print(f"\n\nSet seed to {self.env_seed}!")

        _model_file = self.set_model(model)

        if not isinstance(num_steps, int):
            num_steps = int(self.run_config["max_episode_steps"])

        obs = self.env.reset(run_render=run_render)
        prev_total_rewards = self.env.reset_reward_component_info

        self.actions = [None] * num_steps
        self.applied_torques = [None] * num_steps
        self.observations = [None] * num_steps
        self.reward_info = [None] * num_steps
        self.rewards = [None] * num_steps

        if run_render and save_vid:
            vid_filename = Path(
                os.path.abspath(
                    self.videos_dir / (self.results_ID + "_render_vid.mp4")
                )
            )
            logIDvideo = p.startStateLogging(
                p.STATE_LOGGING_VIDEO_MP4, str(vid_filename)
            )

        # actions_df = pd.read_pickle("../../hardware_transfer/IHM-SAC-10.pkl").iloc[: , :8]

        total_reward = 0
        for i in range(num_steps):
            action, _states = self.model.predict(obs) #, deterministic=True)
            # action = list(action[:4]) + [0, 0] + list(action[6:])
        # for i, action_line in actions_df.iterrows():
        #     action = action_line.to_numpy()
            if zero_action:
                action *= 0
            obs, rewards, _dones, info = self.env.step(action)

            if record_data:
                print("  * REWARD: " + str(rewards))
                total_reward += rewards

                self.actions[i] = deepcopy(action)
                self.applied_torques[i] = deepcopy(self.env.applied_torque)
                self.observations[i] = deepcopy(obs)
                self.reward_info[i] = extract_step_reward(prev_total_rewards, info)
                self.rewards[i] = rewards

                prev_total_rewards = deepcopy(info)

            if run_render:
                self.env.render()

        succeeded = self.env.check_success()
        print(f"\nTotal Reward over Rollout: {np.round(total_reward, 3)}")
        print(f"Succeeded: {succeeded}\n\n")

        if run_render and save_vid:
            p.stopStateLogging(logIDvideo)

        return succeeded


    def run_and_save_rollout(
        self,
        model="best_model",
        from_callbacks=False, 
        num_steps=None,
        run_render=True,
        save_vid=False,
        zero_action=False,
        seed=None
    ):

        self.run_rollout(model, from_callbacks, num_steps, run_render, save_vid, zero_action, seed=seed)

        data_dir_ID = self.processed_data_dir / self.results_ID
        os.makedirs(data_dir_ID)

        # make dataframes
        reward_df = pd.DataFrame(self.rewards, columns=["step_reward"])
        reward_components_df = pd.concat([reward_df, pd.DataFrame(self.reward_info)], axis=1)
        observations_df = pd.DataFrame(self.observations)
        raw_actions_df = pd.DataFrame(self.actions, columns=[f"action_{i}" for i in range(len(self.actions[0]))])
        applied_torques_df = pd.DataFrame(self.applied_torques, columns=[f"applied_{i}" for i in range(len(self.actions[0]))])
        actions_df = pd.concat([raw_actions_df, applied_torques_df], axis=1)

        # save dataframes
        reward_components_df.to_pickle(data_dir_ID / "reward_components.pkl")
        observations_df.to_pickle(data_dir_ID / "observations.pkl")
        actions_df.to_pickle(data_dir_ID / "actions.pkl")

    
    def calculate_success_rate(self, model="best_model", num_steps=None, num_runs=10):
        success_rate_dir = self.results_dir / "success_rate"
        os.makedirs(success_rate_dir, exist_ok=True)

        results = np.zeros(num_runs)

        for i in range(num_runs):
            results[i] = self.run_rollout(model=model, from_callbacks=False, num_steps=num_steps, run_render=False, save_vid=False, zero_action=False, record_data=False)
        
        success_rate = np.mean(results) * 100
        print(f"Finished {num_runs} rollouts on {self.run_ID}!")
        print(f"SUCCESS COUNT: {np.sum(results)}")
        print(f"SUCCESS RATE: {np.round(success_rate, 2)}%")

        results_dict = {
            "run_ID": self.run_ID,
            "datetime": self.datetime,
            "num_rollouts": num_runs,
            "success_count": np.sum(results),
            "success_rate": success_rate
        }

        f = open(success_rate_dir / f"results_{self.datetime}.json", "w")
        json.dump(results_dict, f)
        f.close()

        return results_dict



def run(exp_name, run_group_name, run_name, exp_abs_path=EXPERIMENT_ABS_PATH, debug=False, seed=None, save=False, model="best_model", from_callbacks=False, num_steps=None, run_render=True, save_vid=False, zero_action=False):
    run_ID = [exp_name, run_group_name, run_name]
    policy = Policy_rollout(
        exp_abs_path=exp_abs_path, run_ID=run_ID, debug=debug
    )
    rollout_func = policy.run_and_save_rollout if save else policy.run_rollout

    print("\n\nRunning policy rollout!")
    # print(f"Results ID: {policy.results_ID}\n\n")

    rollout_func(model=model, from_callbacks=from_callbacks, num_steps=num_steps, run_render=run_render, save_vid=save_vid, zero_action=zero_action, seed=seed)

    print("\n\nFinished running policy rollout!")
    print(f"Results ID: {policy.results_ID}\n\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Policy rollout run")
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
        "--exp_abs_path",
        help="Experiment directory absolute path",
        required=False,
        default=EXPERIMENT_ABS_PATH,
    )
    parser.add_argument(
        "-d", "--debug", help="Display SoMo-RL Debugging Dashboard", action="store_true"
    )
    parser.add_argument(
        "--seed",
        help="Random seed",
        required=False,
        default=None
    )
    parser.add_argument(
        "-s",
        "--save",
        help="Record data over rollout",
        action="store_true",
    )
    parser.add_argument(
        "-m",
        "--model",
        help="Model name, path, or checkpoint number",
        required=False,
        default="best_model",
    )
    parser.add_argument(
        "--not_from_callbacks",
        help="Look at 'select checkpoints' directory for checkpoints rather than at callbacks",
        action="store_true",
    )
    parser.add_argument(
        "--num_steps",
        help="Number of steps to run for",
        required=False,
        default=None
    )
    parser.add_argument(
        "-v",
        "--render",
        help="Render the env",
        action="store_true",
    )
    parser.add_argument(
        "-sv",
        "--save_vid",
        help="Record render video over rollout",
        action="store_true",
    )
    parser.add_argument(
        "--zero_action",
        help="Set action to 0 for each step",
        action="store_true",
    )
    arg = parser.parse_args()
    seed = int(arg.seed) if arg.seed is not None else None
    num_steps = int(arg.num_steps) if arg.num_steps is not None else None
    run(exp_name=arg.exp_name, run_group_name=arg.run_group_name, run_name=arg.run_name, exp_abs_path=arg.exp_abs_path, debug=arg.debug, seed=seed, save=arg.save, model=arg.model,from_callbacks=not arg.not_from_callbacks, num_steps=num_steps, run_render=arg.render, save_vid=arg.save_vid, zero_action=arg.zero_action)