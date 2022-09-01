import jsonschema
import os
import sys
import yaml
from pathlib import Path
from copy import deepcopy

from somo_rl.utils.construct_policy_model import ALGS, POLICIES
from somo_rl.utils.import_environment import import_env


REQUIRED_ARGS = ["env_id", "seed", "alg", "policy", "training_timesteps", "eval_cb", "checkpoint_cb", "max_episode_steps"]
required_args_str = str(REQUIRED_ARGS).replace("'", '')

schema = f"""
type: object
required: {required_args_str}
properties:
    env_id:
        type: string
    seed:
        type: number
    alg:
        type: string
    policy:
        type: string
    training_timesteps:
        type: number
    eval_cb:
        required: [n_eval_episodes, eval_freq]
        properties:
            n_eval_episodes:
                type: number
            eval_freq:
                type: number
    checkpoint_cb:
        required: [save_freq]
        properties:
            save_freq:
                type: number
    action_time:
        type: number
    bullet_time_step:
        type: number
    max_torque_rate:
        type: number 
    torque_multiplier:
        oneOf:
          - type: number
          - type: array
    max_episode_steps:
        type: number
    run_started:
        type: boolean
"""

def validate_config(
    config_file_abs_path,
):
    with open(config_file_abs_path, "r") as config_file:
        try:
            run_config = yaml.safe_load(config_file)
        except yaml.YAMLError as exc:
            print(exc)
            return {}

    try:
        jsonschema.validate(run_config, yaml.load(schema))
    except jsonschema.exceptions.ValidationError as exp:
        print("RUN CONFIG ERROR: " + str(exp))
        return {}

    if run_config["alg"] not in ALGS:
        print("TRAINING CONFIG ERROR: Invalid RL algorithm.")
        return {}

    if run_config["policy"] not in POLICIES:
        print("TRAINING CONFIG ERROR: Invalid RL policy.")
        return {}

    return run_config


def construct_benchmark_config(config_file_abs_path): 
    run_config = validate_config(config_file_abs_path)
    import_env(run_config["env_id"])
    from environments.utils.parse_benchmark_config import load
    benchmark_config = deepcopy(load(run_config["env_id"]))
    for param in benchmark_config:
        run_config[param] = deepcopy(benchmark_config[param])
    return run_config
