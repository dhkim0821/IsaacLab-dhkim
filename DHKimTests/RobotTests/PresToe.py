import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets.articulation import ArticulationCfg

PRESTOE_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"/home/daros/Repositories/IsaacLab/DHKimTests/RobotTests/prestoe.usd",
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
            enabled_self_collisions=False, 
            solver_position_iteration_count=4, 
            solver_velocity_iteration_count=0
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.05),
        joint_pos={
            ".*_hipyaw": 0.0,
            ".*_hiproll": 0.0,
            ".*_hippitch": 0.0,  # -16 degrees
            ".*_knee": 0.0,  # 45 degrees
            ".*_anklepitch": 0.0,  # -30 degrees
            ".*_ankleroll": 0.0,  # 0 degrees
            ".*_toepitch": 0.0,  # -30 degrees
            "torsoyaw": 0.0,
            ".*_shoulderpitch": 0.2,
            ".*_shoulderroll": 0.0,
            ".*_shoulderyaw": 0.0,
            ".*_elbowpitch": -0.5,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_hipyaw", ".*_hiproll", ".*_hippitch", 
                              ".*_knee",
                              "torsoyaw"],
            effort_limit=300,
            velocity_limit=100.0,
            stiffness={
                ".*_hipyaw": 150.0,
                ".*_hiproll": 150.0,
                ".*_hippitch": 200.0,
                ".*_knee": 200.0,
                "torsoyaw": 200.0,
            },
            damping={
                ".*_hipyaw": 5.0,
                ".*_hiproll": 5.0,
                ".*_hippitch": 5.0,
                ".*_knee": 5.0,
                "torsoyaw": 5.0,
            },
        ),
        "ankle": ImplicitActuatorCfg(
            joint_names_expr=[".*_anklepitch", ".*_ankleroll", ".*_toepitch"],
            effort_limit=100,
            velocity_limit=100.0,
            stiffness={".*_anklepitch": 20.0,
                       ".*_ankleroll": 20.0},
            damping={".*_anklepitch": 4.0,
                     ".*_ankleroll": 4.0},
        ),
        "toe": ImplicitActuatorCfg(
            joint_names_expr=[".*_toepitch"],
            effort_limit=50,
            velocity_limit=50.0,
            stiffness={".*_toepitch": 20.0},
            damping={".*_toepitch": 4.0},
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[".*_shoulderpitch", ".*_shoulderroll", ".*_shoulderyaw", ".*_elbowpitch"],
            effort_limit=300,
            velocity_limit=100.0,
            stiffness={
                ".*_shoulderpitch": 40.0,
                ".*_shoulderroll": 40.0,
                ".*_shoulderyaw": 40.0,
                ".*_elbowpitch": 40.0,
            },
            damping={
                ".*_shoulderpitch": 10.0,
                ".*_shoulderroll": 10.0,
                ".*_shoulderyaw": 10.0,
                ".*_elbowpitch": 10.0,
            },
        ),
    },
)
"""Configuration of PresToe robot."""
