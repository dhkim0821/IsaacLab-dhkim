# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg, DeformableObjectCfg, RigidObjectCfg
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.managers import CurriculumTermCfg as CurrTerm
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from omni.isaac.lab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from omni.isaac.lab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg

from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from omni.isaac.lab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import DHKimTests.RobotRL.PrestoeBox.PrestoeBox_mdp as mdp
from DHKimTests.RobotRL.PrestoeBox.Prestoe_locomanip_cfg import Prestoe_LocoManip_CFG 

min_height = 0.95

@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    """Configuration for the lift scene with a robot and a object.
    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the target object, robot and end-effector frames
    """

    # robots: will be populated by agent env cfg
    robot: ArticulationCfg = MISSING
    # end-effector sensor: will be populated by agent env cfg
    ee_frame: FrameTransformerCfg = MISSING

    # Table (Height: 0.6)
    table =  RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Table",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.6, 0, 0.55], rot=[1, 0, 0, 0]),
            spawn = sim_utils.MeshCuboidCfg(
            size = (0.8, 1.2, 0.1),
            rigid_props = RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_depenetration_velocity=5.0,
                disable_gravity=True,
                kinematic_enabled = True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.01, 0.1, 0.01)),
        ),
    )

    # target object: will be populated by agent env cfg
    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        # init_state=RigidObjectCfg.InitialStateCfg(pos=[0.45, 0.05, 0.85], rot=[0.966, 0, 0, 0.259]),
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.45, 0.05, 0.85], rot=[1.0, 0, 0, 0.1]),
        spawn = sim_utils.MeshCuboidCfg(
            size = (0.35, 0.3, 0.5),
            rigid_props = RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.02, 0.02)),
            mass_props=sim_utils.MassPropertiesCfg(mass=5.5),
        ),
    )


    # plane
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        terrain_generator=None,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
 
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class CommandsCfg:
    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="table",
        body_name="Table",  # will be set by agent env cfg
        resampling_time_range=(10.0, 10.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(-0.2, -0.1), pos_y=(-0.1, 0.1), pos_z=(0.4, 0.60), 
            roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0),
        ),
    )


@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", 
                                           joint_names=[".*"], scale=0.5, use_default_offset=True)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.05, n_max=0.05))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.0, n_max=1.0))
 
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        object_orientation = ObsTerm(func=mdp.object_orientation_in_robot_root_frame)
        target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={ # from initial state
            "pose_range": {"x": (-0.05, 0.05), "y": (-0.25, 0.25), "z": (0.0, 0.0), "yaw": (-0.2, 0.2)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names="Object"),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # reaching_object = RewTerm(func=mdp.object_ee_distance, params={"std": 0.1}, weight=1.0)
    # lifting_object = RewTerm(func=mdp.object_is_lifted, params={"minimal_height": 0.13}, weight=15.0)

    reaching_object = RewTerm(func=mdp.object_ee_distance, params={"std": 0.1}, weight=1.0e-4)
    lifting_object = RewTerm(func=mdp.object_is_lifted, params={"minimal_height": min_height}, weight=1.0e-4)


    object_goal_tracking = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.5, "minimal_height": min_height, "command_name": "object_pose"},
        weight=20.0,
    )

    object_goal_tracking_fine_grained = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.05, "minimal_height": min_height, "command_name": "object_pose"},
        weight=10.0,
    )

    # action penalty
    # action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)

    # joint_vel = RewTerm(
    #     func=mdp.joint_vel_l2,
    #     weight=-1e-4,
    #     params={"asset_cfg": SceneEntityCfg("robot")},
    # )

    ## From Locomotion
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-150.0)

    # Penalize ankle joint limits
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits, weight=-0.5, 
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_anklepitch")}
    )
    # Penalize deviation from default of the joints that are not essential for locomotion
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-3.8,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hipyaw", ".*_hiproll"])},
    )
    joint_deviation_torso = RewTerm(
        func=mdp.joint_deviation_l1, weight=-3.8, 
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="torsoyaw")}
    )
    # feet_slide = RewTerm(
    #     weight=-0.25,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*foot_link"),
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*foot_link"),
    #     },
    # )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": 0.8, "asset_cfg": SceneEntityCfg("object")}
    )
    # base_contact = DoneTerm(
    #     func=mdp.illegal_contact,
    #     params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*torso_link"), "threshold": 1.0},
    # )
    falliq = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": 0.73, "asset_cfg": SceneEntityCfg("robot")}
    )
 

@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    # action_rate = CurrTerm(
    #     func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -0.001, "num_steps": 20000}
    # )

    # joint_vel = CurrTerm(
    #     func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -0.001, "num_steps": 20000}
    # )
    reaching_object = CurrTerm(func=mdp.modify_reward_weight, 
                               params={"term_name": "reaching_object", "weight": 20.0, "num_steps": 5000})
    lifting_object = CurrTerm(func=mdp.modify_reward_weight, 
                              params={"term_name": "lifting_object", "weight":35.0, "num_steps": 5000})

    # termination_penalty = CurrTerm(func=mdp.modify_reward_weight, 
    #                           params={"term_name": "termination_penalty", "weight":-5.0, "num_steps": 10000})



##
# Environment configuration
##


@configclass
class TerrainBoxEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the lifting environment."""

    # Scene settings
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=1500, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        self.scene.robot = Prestoe_LocoManip_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # general settings
        self.decimation = 4 
        self.episode_length_s = 7.0
        # simulation settings
        self.sim.dt = 0.005  # 200Hz
        self.sim.render_interval = self.decimation
        self.scene.contact_forces.update_period = self.sim.dt

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
