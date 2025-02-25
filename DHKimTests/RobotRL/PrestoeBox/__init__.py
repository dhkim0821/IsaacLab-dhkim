import os
import gymnasium as gym
# from DHKimTests.RobotRL.BoxLift import agents, ik_abs_env_cfg, ik_rel_env_cfg, joint_pos_env_cfg
from . import agents
from .locomanip_env_cfg import prestoe_boxlift_env_cfg

gym.register(
    id="Prestoe-Lift-RL-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": prestoe_boxlift_env_cfg.PrestoeLiftEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PrestoeLift_PPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="Prestoe-Lift-RL-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": prestoe_boxlift_env_cfg.PrestoeLiftEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PrestoeLift_PPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)
