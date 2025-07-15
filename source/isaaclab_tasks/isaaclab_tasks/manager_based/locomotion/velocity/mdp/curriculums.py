# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def terrain_levels_vel(
    env: "ManagerBasedRLEnv", env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Curriculum based on the distance the robot walked when commanded to move at a desired velocity.

    This term is used to increase the difficulty of the terrain when the robot walks far enough and decrease the
    difficulty when the robot walks less than half of the distance required by the commanded velocity.

    .. note::
        It is only possible to use this term with the terrain type ``generator``. For further information
        on different terrain types, check the :class:`isaaclab.terrains.TerrainImporter` class.

    Returns:
        The mean terrain level for the given environment ids.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")
    # compute the distance the robot walked
    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    # robots that walked far enough progress to harder terrains
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    # robots that walked less than half of their required distance go to simpler terrains
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up
    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)
    # return the mean terrain level
    return torch.mean(terrain.terrain_levels.float())


def command_range_curriculum(
    env: "ManagerBasedRLEnv", 
    env_ids: Sequence[int] = None, 

    #These are example values, aren't actually used
    command_name: str = "base_velocity", 
    attribute: str = "ranges.lin_vel_x",
    range_start: tuple = (-1.0, 1.0), 
    range_end: tuple = (-3.0, 3.0), 
    num_steps: int = 5,
    increment_every: int = 2000,
    **kwargs
):
    """
    Gradually increases the command range from range_start to range_end over num_steps.
    Increments after every 'increment_every' learning iterations (calls to this function).
    Works by directly setting attributes like env.command_manager.cfg.base_velocity.ranges.lin_vel_x.
    """
    if not hasattr(env, "_command_range_progress"):
        env._command_range_progress = {}
    if attribute not in env._command_range_progress:
        env._command_range_progress[attribute] = 0

    if not hasattr(env, "_command_range_call_counter"):
        env._command_range_call_counter = {}
    if attribute not in env._command_range_call_counter:
        env._command_range_call_counter[attribute] = 0

    env._command_range_call_counter[attribute] += 1
    call_count = env._command_range_call_counter[attribute]
    progress = env._command_range_progress[attribute]

    scaled_increment_every = increment_every * 27
    if call_count // scaled_increment_every > progress and progress < num_steps:
        progress += 1
        env._command_range_progress[attribute] = progress

    new_min = range_start[0] + (range_end[0] - range_start[0]) * progress / num_steps
    new_max = range_start[1] + (range_end[1] - range_start[1]) * progress / num_steps

    # Set the new range directly using env.command_manager.cfg
    cmd_cfg = getattr(env.command_manager.cfg, command_name)
    attr_parts = attribute.split('.')

    # Navigate to the parent of the target attribute
    obj = cmd_cfg
    for part in attr_parts[:-1]:
        obj = getattr(obj, part)

    # Set the new value
    setattr(obj, attr_parts[-1], (new_min, new_max))

    return {attribute: (new_min, new_max), "progress": progress}


#     env.command_manager.cfg.base_velocity.ranges.lin_vel_x=(-3.0, 3.0)
#     env.command_manager.cfg.base_velocity.ranges.lin_vel_y=(-2.0, 2.0)
