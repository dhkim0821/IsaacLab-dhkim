import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets.articulation import ArticulationCfg

MINI_ARM_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"/home/daros/Repositories/IsaacLab/DHKimTests/RobotTests/mini_arm.usd",
        # usd_path=f"./DHKimTests/RobotTests/mini_arm.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "miniarm_joint1": 0.7,
            "miniarm_joint2": 0.0,
            "miniarm_joint3": 0.0,
            "miniarm_joint4": -2.810,
            "miniarm_joint5": 0.0,
            "miniarm_joint6": 3.037,
        },
    ),
    actuators={
        "miniarm_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["miniarm_joint[1-4]"],
            effort_limit=87.0,
            velocity_limit=2.175,
            stiffness=80.0,
            damping=4.0,
        ),
        "miniarm_forearm": ImplicitActuatorCfg(
            joint_names_expr=["miniarm_joint[5-7]"],
            effort_limit=12.0,
            velocity_limit=2.61,
            stiffness=80.0,
            damping=4.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of Mini Arm robot."""


MINI_ARM_HIGH_PD_CFG = MINI_ARM_CFG.copy()
MINI_ARM_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True
MINI_ARM_HIGH_PD_CFG.actuators["miniarm_shoulder"].stiffness = 400.0
MINI_ARM_HIGH_PD_CFG.actuators["miniarm_shoulder"].damping = 80.0
MINI_ARM_HIGH_PD_CFG.actuators["miniarm_forearm"].stiffness = 400.0
MINI_ARM_HIGH_PD_CFG.actuators["miniarm_forearm"].damping = 80.0
"""Configuration of Mini Arm robot with stiffer PD control.
This configuration is useful for task-space control using differential IK.
"""
