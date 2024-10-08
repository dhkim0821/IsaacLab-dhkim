# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils.math import wrap_to_pi

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv

# define the reward function taking the target joint positions
# as an argument and return the error between and the current joint positions
def joint_pos_target(env: ManagerBasedRLEnv, target: torch.Tensor,
                        asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # wrap the joint positions to (-pi, pi)
    joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
    # print("------------------")
    # print(f"joint pos: \n {joint_pos}")
    # print(f"target pos: \n {target}")
    # print(torch.norm(joint_pos - target, dim=1))
    # print("------------------")

    # compute the reward
    # return torch.sum(torch.square(joint_pos - target), dim=1)
    jpos_err = torch.norm(joint_pos - target, dim=1)
    return 1 - torch.tanh(jpos_err/ 3.14)

def joint_vel_suppression(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize joint velocity."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene["robot"]
    # compute the reward
    return torch.sum(torch.square(asset.data.joint_vel), dim=1)