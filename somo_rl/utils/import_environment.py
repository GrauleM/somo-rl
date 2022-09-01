import sys
import os

# TODO: import repo through pip instead
path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../somogym"))
print(path)
sys.path.insert(0, path)

# All environments within the experiment architecture (in this case block_pusher_experiments) must be enumerated here
# This is allows easy imports of environment modules during training and postprocessing based on config

def import_env(env_id):
    import environments
    env_name = env_id.split("-")[0]

    if env_name == "PlanarReaching":
        import environments.PlanarReaching

    elif env_name == "PlanarReachingObstacle":
        import environments.PlanarReachingObstacle

    elif env_name == "InHandManipulation":
        import environments.InHandManipulation

    elif env_name == "InHandManipulationInverted":
        import environments.InHandManipulationInverted

    elif env_name == "PlanarBlockPushing":
        import environments.PlanarBlockPushing

    elif env_name == "AntipodalGripper":
        import environments.AntipodalGripper

    elif env_name == "PenSpinner":
        import environments.PenSpinner

    elif env_name == "PenSpinnerFar":
        import environments.PenSpinnerFar

    elif env_name == "SnakeLocomotionContinuous":
        import environments.SnakeLocomotionContinuous

    elif env_name == "SnakeLocomotionDiscrete":
        import environments.SnakeLocomotionDiscrete

    elif env_name[-6:] == "Cloner":
        if env_name == "InHandManipulationCloner":
            import environments.SomoBehaviorCloning.InHandManipulationCloner
        
        elif env_name == "PenSpinnerCloner":
            import environments.SomoBehaviorCloning.PenSpinnerCloner

        elif env_name == "PlanarBlockPushingCloner":
            import environments.SomoBehaviorCloning.PlanarBlockPushingCloner

        elif env_name == "SnakeLocomotionDiscreteCloner":
            import environments.SomoBehaviorCloning.SnakeLocomotionDiscreteCloner

    else:
        raise Exception(f"CRITICAL ERROR: Invalid environment '{env_name}' listed in config.")
        sys.exit(1)
