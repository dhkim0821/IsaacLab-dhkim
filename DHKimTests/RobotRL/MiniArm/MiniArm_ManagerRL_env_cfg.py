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

import omni.isaac.lab_tasks.manager_based.classic.cartpole.mdp as mdp_cartpole
# import DHKimTests.RobotRL.MiniArm.mdp as mdp
# import DHKimTests.RobotRL.MiniArm.MiniArm_mdp as mdp
import DHKimTests.RobotRL.MiniArm.MiniArm_mdp as mdp
# print("-------------------")
# print("dir of mdp cartpole")
# print(dir(mdp_cartpole))
# print("dir of mdp MiniArm")
# print(dir(mdp))
# print("-------------------")
# import sys
# print('mdp' in sys.modules)

from DHKimTests.RobotRL.MiniArm.MiniArm_cfg import MINIARM_CFG # isort:skip
from DHKimTests.RobotRL.MiniArm.MiniArmCmdCfg import UniformJPosCommandCfg

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
    """Command terms for the MDP."""

    # no commands for this MDP
    # null = mdp.NullCommandCfg()

    # sample joint position target from uniform distribution ranged by [-pi, pi]
    jpos_target = UniformJPosCommandCfg(
        asset_name = "robot",
        num_joints = 6,
        resampling_time_range=(10.0, 10.0),
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
        joint_pos_rel = ObsTerm(func=mdp.joint_pos)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel)
        actions = ObsTerm(func=mdp.last_action)

        jpos_commands = ObsTerm(func=mdp.generated_commands, 
                                params={"command_name": "jpos_target"})

        print(jpos_commands)
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
    # (3) Primary task: jpos control
    # jpos_error = RewTerm(
    #     func=mdp.joint_pos_target,
    #     weight= 5.0,
    #     params={"asset_cfg": SceneEntityCfg("robot", 
    #                                         joint_names=["miniarm_joint.*"]), 
    #                                         "target": torch.tensor([0.0, -0.5, 0.0, 0.5, 0.0, 0.0], 
    #                                                                device="cuda")},
    # )
    jpos_error = RewTerm(
        func=mdp.joint_pos_target_cmd,
        weight= 5.0,
        params={"asset_cfg": SceneEntityCfg("robot", 
                                            joint_names=["miniarm_joint.*"]), 
                                            "command_name": "jpos_target"}
    )
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # jvel_suppression = RewTerm(
    #     func=mdp.joint_vel_suppression,
    #     weight = -0.1)

    


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


##
# Environment configuration
##


@configclass
class MiniArmEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the cartpole environment."""

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
        self.episode_length_s = 7 
        # viewer settings
        self.viewer.eye = (2.5, 2.5, 2.5)
        # simulation settings
        self.sim.dt = 1 / 200
        self.sim.render_interval = self.decimation