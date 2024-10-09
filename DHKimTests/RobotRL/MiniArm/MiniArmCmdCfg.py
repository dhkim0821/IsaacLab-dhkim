import math
from dataclasses import MISSING

from omni.isaac.lab.managers import CommandTermCfg
from omni.isaac.lab.utils import configclass

from DHKimTests.RobotRL.MiniArm.jpos_command import UniformJPosCommand

@configclass
class UniformJPosCommandCfg(CommandTermCfg):
    """Configuration for uniform pose command generator."""

    class_type: type = UniformJPosCommand

    asset_name: str = MISSING
    num_joints: int = MISSING
    """Name of the asset in the environment for which the commands are generated."""
    