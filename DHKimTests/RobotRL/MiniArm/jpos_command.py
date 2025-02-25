from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from MiniArmCmdCfg import UniformJPosCommandCfg 


class UniformJPosCommand(CommandTerm):
    cfg: UniformJPosCommandCfg 
    """Configuration for the command term."""

    def __init__(self, cfg: UniformJPosCommandCfg, env: ManagerBasedRLEnv):
        """Initialize the command term class.

        Args:
            cfg: The configuration parameters for the command term.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.num_j = cfg.num_joints
        self.jpos_target = torch.zeros(self.num_envs, self.num_j, device=self.device)

    def __str__(self) -> str:
        msg = "UniformJPosCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired goal pose in the environment frame. Shape is (num_envs, 6)."""
        return self.jpos_target 

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        return

    def _resample_command(self, env_ids: Sequence[int]):
        # sample new orientation targets
        self.jpos_target[env_ids, :] = (torch.rand((len(env_ids), self.num_j), device=self.device)-0.5) * torch.pi
        # print("UniformJPosCommand")
        # print(env_ids)
        # print(len(env_ids), self.num_j)
        # print(self.jpos_target)
        # print(self.jpos_target.shape)

    def _update_command(self):
        return

    def _set_debug_vis_impl(self, debug_vis: TYPE_CHECKING):
        return

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self.robot.is_initialized:
            return
