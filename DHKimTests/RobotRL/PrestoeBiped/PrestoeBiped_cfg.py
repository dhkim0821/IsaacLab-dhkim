import isaaclab.sim as sim_utils
from isaaclab.actuators import ActuatorNetMLPCfg, DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
# from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
usd_relative_path = os.path.join(BASE_DIR, "../../RobotTests", "prestoebiped_minimal.usd")
PrestoeBiped_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=usd_relative_path,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.00),
        joint_pos={
            ".*_hipyaw": 0.0,
            ".*_hiproll": 0.0,
            ".*_hippitch": -0.05,  # -16 degrees
            ".*_knee": 0.1,
            ".*_anklepitch": -0.1,  # -30 degrees
            ".*_ankleroll": 0.0,  # 0 degrees
            ".*_toepitch": 0.02,  # -30 degrees
            "torsoyaw": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "hips": ImplicitActuatorCfg(
            joint_names_expr=[".*_hipyaw", ".*_hiproll", ".*_hippitch", "torsoyaw"],
            effort_limit=70,
            velocity_limit=30.0,
            stiffness={
                ".*_hipyaw": 150.0,
                ".*_hiproll": 150.0,
                ".*_hippitch": 200.0,
                "torsoyaw": 200.0,
            },
            damping={
                ".*_hipyaw": 5.0,
                ".*_hiproll": 5.0,
                ".*_hippitch": 5.0,
                "torsoyaw": 5.0,
            },
        ),
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_knee"],
            effort_limit=200,
            velocity_limit=50.0,
            stiffness={ ".*_knee": 200.0 },
            damping={ ".*_knee": 5.0 },
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_anklepitch"],
            effort_limit=100,
            velocity_limit=30.0,
            stiffness={".*_anklepitch": 20.0 },
            damping={".*_anklepitch": 4.0},
        ),
        "toe": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankleroll", ".*_toepitch"],
            effort_limit=40,
            velocity_limit=20.0,
            stiffness={".*_ankleroll": 20.0, ".*_toepitch": 20.0},
            damping={".*_ankleroll": 4.0, ".*_toepitch": 4.0},
        ),
    },
)
"""Configuration for the Prestoe Humanoid robot."""


PrestoeBiped_MINIMAL_CFG = PrestoeBiped_CFG.copy()
PrestoeBiped_MINIMAL_CFG.spawn.usd_path = usd_relative_path
"""Configuration for the PrestoeBiped robot with fewer collision meshes.
This configuration removes most collision meshes to speed up simulation.
"""

