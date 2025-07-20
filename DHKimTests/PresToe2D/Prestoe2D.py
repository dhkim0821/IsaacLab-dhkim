import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

Prestoe2D_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"/home/daros/Repositories/IsaacLab-dhkim/DHKimTests/PresToe2D/model/Prestoe2D_simple.usd",
        # activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=0
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "rootx": 0.0,
            "rootz": 1.15,
            "rooty": 0.0,
            "r_hip": -0.310,
            "r_knee": 0.2,
            "r_ankle": -0.37,
            "l_hip": 0.310,
            "l_knee": 0.0,
            "l_ankle": -0.337,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "r_hip": ImplicitActuatorCfg(
            joint_names_expr=["r_hip"],
            effort_limit_sim=87.0,
            velocity_limit_sim=2.175,
            stiffness=80.0,
            damping=4.0,
        ),
        "r_knee": ImplicitActuatorCfg(
            joint_names_expr=["r_knee"],
            effort_limit_sim=87.0,
            velocity_limit_sim=2.175,
            stiffness=80.0,
            damping=4.0,
        ),
        "r_ankle": ImplicitActuatorCfg(
            joint_names_expr=["r_ankle"],
            effort_limit_sim=10.0,
            velocity_limit_sim=2.175,
            stiffness=80.0,
            damping=4.0,
        ),
        "l_hip": ImplicitActuatorCfg(
            joint_names_expr=["l_hip"],
            effort_limit_sim=87.0,
            velocity_limit_sim=2.175,
            stiffness=80.0,
            damping=4.0,
        ),
        "l_knee": ImplicitActuatorCfg(
            joint_names_expr=["l_knee"],
            effort_limit_sim=87.0,
            velocity_limit_sim=2.175,
            stiffness=80.0,
            damping=4.0,
        ),
        "l_ankle": ImplicitActuatorCfg(
            joint_names_expr=["l_ankle"],
            effort_limit_sim=10.0,
            velocity_limit_sim=2.175,
            stiffness=80.0,
            damping=4.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of Prestoe2D robot."""

