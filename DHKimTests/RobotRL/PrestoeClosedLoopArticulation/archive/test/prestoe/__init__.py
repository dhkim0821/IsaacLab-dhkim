import gymnasium as gym

from . import agents

from .prestoe_env_cfg import PrestoeEnvCfg
# from scripts.prestoe.rsl_rl_ppo_cfg_prestoe import Prestoe_PPORunnerCfg


gym.register(
    id="Prestoe-RL-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": PrestoeEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Prestoe_PPORunnerCfg",
    },
)