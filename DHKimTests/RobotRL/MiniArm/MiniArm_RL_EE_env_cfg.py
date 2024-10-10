import math

import torch
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import DHKimTests.RobotRL.MiniArm.MiniArm_mdp as mdp

from DHKimTests.RobotRL.MiniArm.MiniArm_cfg import MINIARM_CFG # isort:skip

@configclass
class MiniArmSceneCfg(InteractiveSceneCfg):
    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    robot: ArticulationCfg = MINIARM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )
    distant_light = AssetBaseCfg(
        prim_path="/World/DistantLight",
        spawn=sim_utils.DistantLightCfg(color=(0.9, 0.9, 0.9), intensity=2500.0),
        init_state=AssetBaseCfg.InitialStateCfg(rot=(0.738, 0.477, 0.477, 0.0)),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    ee_target = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="ee_link",
        resampling_time_range=(4.0, 4.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.45, 0.85),
            pos_y=(-0.2, 0.2),
            pos_z=(0.15, 0.5),
            roll=(-0.2, 0.2),
            pitch= (-1.5, 1.5),  # depends on end-effector axis
            yaw=(-0.5, 0.5),
        ),
    )

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    joint_effort = mdp.JointEffortActionCfg(asset_name="robot", 
                                            joint_names=["miniarm_joint.*"], 
                                            scale=5.0)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_pos = ObsTerm(func=mdp.joint_pos)
        joint_vel = ObsTerm(func=mdp.joint_vel)
        actions = ObsTerm(func=mdp.last_action)

        ee_pose_target = ObsTerm(func=mdp.generated_commands, 
                                params={"command_name": "ee_target"})

        # def __post_init__(self) -> None:
        #     self.enable_corruption = False
        #     self.concatenate_terms = True
        def __post_init__(self):
            self.enable_corruption = True 
            self.concatenate_terms = True


    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""
    # reset
    reset_joint_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["miniarm_joint.*"]),
            "position_range": (-3.1, 3.1),
            "velocity_range": (-5.1, 5.1),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # (1) Constant running reward
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    # (2) Failure penalty
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)
    # (3) Primary task: EE Pose Ctrl
    ee_pos_error = RewTerm(
        func=mdp.position_command_error,
        weight= -0.3,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["ee_link"]),
                                            "command_name": "ee_target"}
    )
    ee_pos_error_fine = RewTerm(
        func=mdp.position_command_error_tanh,
        weight= 1.5,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["ee_link"]),
                                            "command_name": "ee_target"}
    )
    ee_ori_error = RewTerm(
        func=mdp.orientation_command_error,
        weight= -0.5,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["ee_link"]),
                                            "command_name": "ee_target"}
    )
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # (2) Cart out of bounds
    jpos_out_of_bounds = DoneTerm(
        func=mdp.joint_pos_out_of_manual_limit,
        params={"asset_cfg": SceneEntityCfg("robot", 
                                            joint_names=["miniarm_joint.*"]), 
                                            "bounds": (-3.1, 3.1)},
    )


@configclass
class CurriculumCfg:
    """Configuration for the curriculum."""

    pass


@configclass
class MiniArmEE_EnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: MiniArmSceneCfg = MiniArmSceneCfg(num_envs=2048, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    curriculum: CurriculumCfg = CurriculumCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    # No command generator
    commands: CommandsCfg = CommandsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 12
        # viewer settings
        self.viewer.eye = (2.5, 2.5, 2.5)
        # simulation settings
        self.sim.dt = 1 / 200
        self.sim.render_interval = self.decimation