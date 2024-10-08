import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets.articulation import ArticulationCfg

MINIARM_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"/home/daros/Repositories/IsaacLab-dhkim/DHKimTests/RobotRL/MiniArm.usd",
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
            "miniarm_joint1": 0.0,
            "miniarm_joint2": -0.5,
            "miniarm_joint3": 0.0,
            "miniarm_joint4": 0.5,
            "miniarm_joint5": 0.0,
            "miniarm_joint6": 0.5,
        },
    ),
    actuators={
        "miniarm_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["miniarm_joint[1-4]"],
            effort_limit=87.0,
            velocity_limit=2.175,
            # stiffness=80.0,
            # damping=4.0,
            stiffness=0.0,
            damping=0.3,
         ),
        "miniarm_forearm": ImplicitActuatorCfg(
            joint_names_expr=["miniarm_joint[5-6]"],
            effort_limit=12.0,
            velocity_limit=2.61,
            # stiffness=80.0,
            # damping=4.0,
            stiffness=0.0,
            damping=0.3,
         ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of Mini Arm robot."""


MINIARM_HIGH_PD_CFG = MINIARM_CFG.copy()
MINIARM_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True
MINIARM_HIGH_PD_CFG.actuators["miniarm_shoulder"].stiffness = 400.0
MINIARM_HIGH_PD_CFG.actuators["miniarm_shoulder"].damping = 80.0
MINIARM_HIGH_PD_CFG.actuators["miniarm_forearm"].stiffness = 400.0
MINIARM_HIGH_PD_CFG.actuators["miniarm_forearm"].damping = 80.0
"""Configuration of Mini Arm robot with stiffer PD control.
This configuration is useful for task-space control using differential IK.
"""
