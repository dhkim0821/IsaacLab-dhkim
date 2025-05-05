import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from stable_baselines3 import PPO

FOURBAR_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/isaac/Documents/Github/IsaacLab-dhkim/scripts/fourbarlinkage/Assembly_4.usd",  
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
        )
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),

        joint_pos={
            "Revolute_1": 0.0,
            "Revolute_2": 0.0,
            "Revolute_3": 0.0,
            #"Revolute_4": 0.0,
            # "Revolute_4": 0.0, It seems like I cannot add this here given the fact that i've removed it from articulation
        },
        
    ),
    actuators={
        "motor_1": ImplicitActuatorCfg(
            joint_names_expr=["Revolute_1"],
            effort_limit=400.0,
            velocity_limit=100.0,
            stiffness=0.0,
            damping=10.0,
        )
    },
)
