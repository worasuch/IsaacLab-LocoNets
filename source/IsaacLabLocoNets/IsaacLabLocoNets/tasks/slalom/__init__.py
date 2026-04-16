"""Locomotion environments with velocity-tracking commands.

These environments are based on the `legged_gym` environments provided by Rudin et al.

Reference:
    https://github.com/leggedrobotics/legged_gym
"""

import gymnasium as gym

from . import agents
##
# Register Gym environments.
##


gym.register(
    id='Slalom',
    entry_point=f"{__name__}.slalom_env:SlalomLocomotionTask",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.slalom_env_cfg:SlalomEnvCfg",
        "es_cfg_entry_point": f"{agents.__name__}:es_cfg.yaml",
    },
)