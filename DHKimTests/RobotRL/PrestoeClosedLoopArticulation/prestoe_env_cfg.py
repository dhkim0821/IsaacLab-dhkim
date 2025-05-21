import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from dataclasses import MISSING


from . import prestoe_mdp as mdp
# import prestoe_mdp as mdp

from .prestoe_config import PRESTOE_CFG
# from prestoe_config import PRESTOE_CFG

@configclass
class PrestoeSceneCfg(InteractiveSceneCfg):
    # add our usual stuff...
    ground_cfg = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    robot = PRESTOE_CFG.replace(prim_path="{ENV_REGEX_NS}/robot") # type: ignore
    # contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/robot/.*", history_length=3, track_air_time=True)

    light_cfg = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=1000),
    )

    
@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), 
            ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
        ),
    )


@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", 
                                           joint_names=[".*"], scale=0.5, use_default_offset=True)
    
    joint_effort = mdp.JointEffortActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.5)
    
@configclass
class ObservationsCfg:

    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True


    policy = PolicyCfg()


@configclass
class EventCfg:
    # I took the code form PrestoeBiped_env_cfg.py
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "mass_distribution_params": (-1, 1),
            "operation": "add",
        },
    )

    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_robot_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "position_range": (-0.1, 0.1),
            "velocity_range": (-0.5, 0.5),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.005, 0.005), "y": (-0.005, 0.005), "yaw": (0, 0)},
            "velocity_range": {
                "x": (-0.005, 0.005),
                "y": (-0.005, 0.005),
                "z": (0, 0.005),
                "roll": (-0.005, 0.005),
                "pitch": (-0.005, 0.005),
                "yaw": (-0.005, 0.005),
            },
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (0, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )

    reset_scene = EventTerm(func=mdp.reset_scene_to_default, mode='reset')
    # pass

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    ## Velocity Environment's reward
    # track_lin_vel_xy_exp = RewTerm(
    #     func=mdp.track_lin_vel_xy_exp,
    #     weight=2.0,
    #     params={"command_name": "base_velocity", "std": 0.25},
    # )
    # track_ang_vel_z_exp = RewTerm(
    #     func=mdp.track_ang_vel_z_exp, 
    #     weight=0.5, 
    #     params={"command_name": "base_velocity", "std": 0.25}
    # )

    alive = RewTerm(func=mdp.is_alive, weight=5.0)
    # progress = RewTerm(func=mdp.progress_reward, weight=1.0, params={"target_pos": (1000.0, 0.0, 0.0)})


    # -- penalties
    lin_vel_z_l2   = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2  = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    dof_acc_l2     = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-8)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    # energy_penalty = RewTerm(func=mdp.energy_penalty, weight=-2.0e-6)
    # feet_air_time = RewTerm(
    #     func=mdp.feet_air_time_positive_biped,
    #     weight=0,
    #     params={
    #         "command_name": "base_velocity",
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*foot_link"),
    #         "threshold": 0.4,
    #     },
    # )


    # undesired_contacts = RewTerm(
    #     func=mdp.undesired_contacts,
    #     weight=-1.0,
    #     params={"sensor_cfg": 
    #             SceneEntityCfg("contact_forces", body_names=[".*hippitch_link", ".*shank_link"]), "threshold": 1.0},
    # )

    # -- optional penalties
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    ## Prestoe specific
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-2.0)
    # lin_vel_z_l2 = None


    # Penalize ankle joint limits
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits, weight=-1.0, 
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_anklepitch")}
    )
    # Penalize deviation from default of the joints that are not essential for locomotion
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hipyaw", ".*_hiproll"])},
    )
    joint_deviation_torso = RewTerm(
        func=mdp.joint_deviation_l1, weight=-0.4, 
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="torsoyaw")}
    )
    joint_deviation_ankle_roll = RewTerm(
        func=mdp.joint_deviation_l1, weight=-0.2, 
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_ankleroll")}
    )
    joint_deviation_ankle_pitch = RewTerm(
        func=mdp.joint_deviation_l1, weight=-0.02, 
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_anklepitch")}
    )
    joint_deviation_toe_pitch = RewTerm(
        func=mdp.joint_deviation_l1, weight=-0.1, 
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_toepitch")}
    )
    

@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # base_contact = DoneTerm(
    #     func=mdp.illegal_contact,
    #     params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*torso_link"), "threshold": 1.0},
    # )
    # thigh_contact = DoneTerm(
    #     func=mdp.illegal_contact,
    #     params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*hippitch_link"), "threshold": 1.0},
    # )
    # shank_contact = DoneTerm(
    #     func=mdp.illegal_contact,
    #     params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*shank_link"), "threshold": 1.0},
    # )
    torso_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.8})

@configclass
class PrestoeEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the cartpole environment."""

    # Scene settings
    scene: PrestoeSceneCfg = PrestoeSceneCfg(num_envs=4096, env_spacing=4.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
 
    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 16.0
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity = 0.2
        # default friction material
        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.restitution = 0.0
