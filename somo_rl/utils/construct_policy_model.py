import numpy as np

from stable_baselines3.ppo import MlpPolicy as PPOMlpPolicy
from stable_baselines3.sac import MlpPolicy as SACMlpPolicy
from stable_baselines3.td3 import MlpPolicy as TD3MlpPolicy
from stable_baselines3 import PPO, SAC, TD3

from stable_baselines3.common.noise import NormalActionNoise


ALGS = {
    "PPO": PPO,
    "SAC": SAC,
    "TD3": TD3,
}

POLICIES = {
    "PPOMlpPolicy": PPOMlpPolicy,
    "SACMlpPolicy": SACMlpPolicy,
    "TD3MlpPolicy": TD3MlpPolicy,
}


def construct_policy_model(alg, policy, env, **kwargs):
    if alg == "TD3":
        # these are the default settings used in the stable baselines documentation
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)
        )
        model = ALGS[alg](POLICIES[policy], env, action_noise=action_noise, **kwargs)
    else:
        model = ALGS[alg](POLICIES[policy], env, **kwargs)
    return model
