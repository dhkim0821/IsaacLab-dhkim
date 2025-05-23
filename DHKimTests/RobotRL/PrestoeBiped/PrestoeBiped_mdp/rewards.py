# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_rotate_inverse, yaw_quat
from isaaclab.assets import Articulation, RigidObject

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    toe_ids = env.scene['contact_forces'].find_bodies(["L_toe_link", "R_toe_link"])[0]
    air_time_toe = contact_sensor.data.current_air_time[:, toe_ids]
    contact_time_toe = contact_sensor.data.current_contact_time[:, toe_ids]
    in_contact_toe = contact_time_toe > 0
    
    
    air_time_heel = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time_heel = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact_heel = contact_time_heel > 0.0
    
    in_contact_leftFoot   = torch.logical_or(in_contact_toe[:,0], in_contact_heel[:,0])
    in_contact_rightFoot  = torch.logical_or(in_contact_toe[:,1], in_contact_heel[:,1])
    in_contact            = torch.stack((in_contact_leftFoot, in_contact_rightFoot), dim = 1)
    
    contact_time_leftFoot  = torch.min(contact_time_toe[:,0] , contact_time_heel[:,0])
    contact_time_rightFoot = torch.min(contact_time_toe[:,1] , contact_time_heel[:,1])
    contact_time           = torch.stack((contact_time_leftFoot, contact_time_rightFoot), dim = 1)
    
    air_time_leftFoot      = torch.min(air_time_toe[:,0] ,  air_time_heel[:,0])
    ait_time_rightFoot     = torch.min(air_time_toe[:,1] ,  air_time_heel[:,1])
    air_time               = torch.stack((air_time_leftFoot, ait_time_rightFoot) , dim = 1)
    
    
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward

def feet_contact(env, command_name: str, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward contact matching with the phase
    """

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    cycle = 0.8
    offset = 0.5
    heel_offset = 0.05
    if hasattr(env, "episode_length_buf"):
        phase = env.episode_length_buf * env.step_dt % cycle / cycle
    else:
        phase = torch.zeros(env.num_envs, device=env.device)
    phase_left_toe = phase
    phase_right_toe = (phase + offset) % 1.0
    phase_left_heel = (phase + heel_offset) % 1.0
    phase_right_heel = (phase + offset + heel_offset) % 1.0
    heel_phase = torch.cat([phase_left_heel.unsqueeze(1), phase_right_heel.unsqueeze(1)], dim=-1)
    toe_phase = torch.cat([phase_left_toe.unsqueeze(1), phase_right_toe.unsqueeze(1)], dim=-1)
    res = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)

    # compute the reward
    toe_ids = env.scene['contact_forces'].find_bodies(["L_toe_link", "R_toe_link"])[0]
    for i in range(2):
        is_stance_toe = toe_phase[:, i] < 0.55
        contact_toe = contact_sensor.data.net_forces_w[:, toe_ids[i], 2] > 1

        is_stance_heel = heel_phase[:, i] < 0.55
        contact_heel = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids[i], 2] > 1

        res += ~(contact_toe ^ is_stance_toe) 
        res += ~(contact_heel ^ is_stance_heel)
    res *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return res


def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_rotate_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)

def energy_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint torques on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    #torch.sum(torch.square(self.torques * self.dof_vel), dim=1)
    return torch.sum(torch.square(asset.data.applied_torque * asset.data.joint_vel), dim=1)
