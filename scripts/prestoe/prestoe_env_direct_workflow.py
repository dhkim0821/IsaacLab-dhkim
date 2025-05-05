import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform

from prestoe_config import PRESTOE_CFG


class PresToeEnvCfg(DirectRLEnvCfg):

    decimation = 1 # same dt for simulation and policy
    episode_length_s = 5.0 # 5 seconds long episode
    action_scale = 100.0

    action_space = len(PRESTOE_CFG.actuators) 
    observation_space = len(PRESTOE_CFG.init_state.joint_pos)

    state_space = 0 # I am not 100 sure what this is doing... I am guessing the buffer size for training nets?
    
    # 120 hz
    sim = SimulationCfg(dt = 1 / 120, render_interval=decimation) # same delta time for both sim and pol
    robot_cfg = PRESTOE_CFG.replace(prim_path="/World/envs/env_.*/Robot") # type: ignore


    # actuators for the pulley
    R_AL_pulley_pitch_name = "R_AL_pulley_pitch"
    R_toe_pulley_pitch_name = "R_toe_pulley_pitch"
    R_AR_pulley_pitch_name = "R_AR_pulley_pitch"
    L_AL_pulley_pitch_name = "R_AL_pulley_pitch"
    L_toe_pulley_pitch_name = "R_toe_pulley_pitch"
    L_AR_pulley_pitch_name = "R_AR_pulley_pitch"


    #TODO: do actuators path stuff for arms and body thing


    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # define the termination flags
    max_pretoe_pos = None

    # define the reward stuff 
    # TODO: Look into chanining the values of rewards to make it better
    rew_scale_alive = 1.0
    rew_scale_terminated = -2.0

class PrestoeEnv(DirectRLEnv):

    pass


@torch.jit.script
def compute_rewards( 
    rew_scale_alive: float,
    rew_scale_terminated: float,
    prestoe_pos: torch.Tensor,
    prestoe_vel: torch.Tensor,
    reset_terminated: torch.Tensor,
):
    pass 

