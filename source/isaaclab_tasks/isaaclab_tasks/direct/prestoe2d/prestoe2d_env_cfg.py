# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# from isaaclab_assets.robots.cartpole import CARTPOLE_CFG

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

import sys
import os

# sys.path.append("/home/daros/Repositories/IsaacLab-dhkim/DHKimTests/")
sys.path.append("/home/daros/Repositories/IsaacLab-dhkim/")

from DHKimTests.PresToe2D.Prestoe2D import Prestoe2D_CFG

@configclass
class Prestoe2dEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 5.0
    # - spaces definition
    # action: 6 actuated joints + 3 inequality constraint (lambda) + 1 equality constraints (mu)
    action_space = 6 + 3 + 1
    # action_space = 6
    observation_space = 18
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # robot(s)
    # robot_cfg: ArticulationCfg = CARTPOLE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    robot_cfg = Prestoe2D_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=500, env_spacing=2.0, replicate_physics=True)
    # scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=30, env_spacing=2.0, replicate_physics=True)

    # custom parameters/scales
    joint_dof_name: str = "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle"
    # - action scale
    action_scale = 50.0  # [Nm]
    # - reward scales
    rew_scale_alive = 1.0
    rew_scale_terminated = -5.0
    rew_scale_xvel = 2.0
    rew_scale_height_maintenance = 0.5
    rew_scale_jvel = -0.05
    # rew_scale_imitation = -0.5
    rew_scale_imitation = 0.0

    # - reset states/conditions
    # initial_pitch_range = [-0.3, 0.3]
    initial_hip_range = [-0.3, 0.3]  
    initial_knee_range = [0, 0.5]
    initial_ankle_range = [-0.3, 0.3]
    min_zpos = 0.75
    max_jpos = 1.2  # max joint position
    
@configclass
class Prestoe2dEnvCfg_PLAY(Prestoe2dEnvCfg):
    """Configuration for the Prestoe2D environment in play mode."""
    # - scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4, env_spacing=2.0, replicate_physics=True)
    # - simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=1)