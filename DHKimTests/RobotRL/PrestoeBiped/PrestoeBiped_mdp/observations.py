# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms
import numpy as np
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def phase_obs(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """getting the phase"""
    robot: RigidObject = env.scene[robot_cfg.name]
    breakpoint()
    cycle = 0.8
    if hasattr(env, "episode_length_buf"):
        phase = env.episode_length_buf * env.step_dt % cycle / cycle
    else:
        phase = torch.zeros(env.num_envs, device=env.device)
    sin_phase = torch.sin(2 * np.pi * phase ).unsqueeze(1)
    cos_phase = torch.cos(2 * np.pi * phase ).unsqueeze(1)

    return torch.cat([sin_phase, cos_phase], dim=1)

