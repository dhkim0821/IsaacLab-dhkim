# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg, CurriculumCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
import math

##
# Pre-defined configs
##
from Vivo.Vivo_Assets.vivo import VIVO_CFG  # isort: skip





# ------------------------------------------------------------
# Curriculum Configuration for Command Range in X and Y axes
# This configuration is used in both `train_rsl` and `play_rsl`
# within the Isaac Lab framework to define progressive command
# ranges for the base velocity in locomotion tasks.
#
# When using `play_rsl`, make sure to set `range_start` to
# the maximum curriculum range you trained on to ensure proper
# behavior. The curriculum progression only happens during training.
# ------------------------------------------------------------

@configclass
class CurriculumCfgWithCommandRange(CurriculumCfg):
    # Curriculum for X-axis linear velocity (forward/backward motion)
    command_range_x = CurrTerm(
        func="isaaclab_tasks.manager_based.locomotion.velocity.mdp.curriculums:command_range_curriculum",
        params={
            "command_name": "base_velocity",  # The target command being modified
            "attribute": "ranges.lin_vel_x",  # Attribute within the environment to be changed
            "range_start": (-1.0, 1.0),       # Initial value range at curriculum step 0
            "range_end": (-3.0, 3.0),         # Final value range after all steps (here it’s constant)
            
            # num_steps determines how many curriculum steps (transitions) are applied
            # Example:
            #   range_start: (-1.0, 1.0)
            #   range_end: (-3.0, 3.0)
            #   num_steps: 4
            # The curriculum progresses as:
            #   Step 0: (-1.0, 1.0)
            #   Step 1: (-1.5, 1.5)
            #   Step 2: (-2.0, 2.0)
            #   Step 3: (-2.5, 2.5)
            #   Step 4: (-3.0, 3.0)
            "num_steps": 2,                   # Number of curriculum steps (excluding the initial one)
            "increment_every": 2000,          # Number of training iterations before advancing one step
        },
    )

    # Curriculum for Y-axis linear velocity (lateral motion)
    command_range_y = CurrTerm(
        func="isaaclab_tasks.manager_based.locomotion.velocity.mdp.curriculums:command_range_curriculum",
        params={
            "command_name": "base_velocity",
            "attribute": "ranges.lin_vel_y",
            "range_start": (-1.0, 1.0),       # Start with a narrower range for lateral commands
            "range_end": (-2.0, 2.0),         # Gradually widen to allow more dynamic movement
            "num_steps": 2,                   # Again, two curriculum steps to reach final range
            "increment_every": 2000,          # Applied every 2000 training iterations
        },
    )

    
@configclass
class VivoRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    # Use the new curriculum config class
    curriculum: CurriculumCfgWithCommandRange = CurriculumCfgWithCommandRange()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.robot = VIVO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        
        #self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/body"
        self.scene.height_scanner = None
        self.scene.depth_camera.prim_path = "{ENV_REGEX_NS}/Robot/body/RSD455/Camera_Pseudo_Depth"

        # scale down the terrains because the robot is small
        self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.1)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01

        # reduce action scale
        self.actions.joint_pos.scale = 0.25

        # set the initial command range for curriculum start
        #self.commands.base_velocity.ranges.lin_vel_x = (2.9, 3.0)
        #self.commands.base_velocity.ranges.lin_vel_y = (1.9, 2.0)
        #self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        #self.commands.base_velocity.ranges.heading = (-math.pi, math.pi)

        # event
        self.events.push_robot = None
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 3.0)
        self.events.add_base_mass.params["asset_cfg"].body_names = "body"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "body"
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }

        # rewards
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_shank"
        self.rewards.feet_air_time.weight = 0.01
        self.rewards.undesired_contacts = None
        self.rewards.dof_torques_l2.weight = -0.0002
        self.rewards.track_lin_vel_xy_exp.weight = 1.75
        self.rewards.track_ang_vel_z_exp.weight = 0.75
        self.rewards.dof_acc_l2.weight = -2.5e-7

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = "body"

@configclass
class VivoRoughEnvCfg_PLAY(VivoRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None