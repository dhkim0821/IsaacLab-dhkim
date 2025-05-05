import isaaclab.sim as sim_utils
from isaaclab.actuators import ActuatorNetMLPCfg, DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR




VIVO_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        # usd_path=f"/home/simon/Repositories/RL_experiment/IsaacLab-dhkim/DHKimTests/RobotRL/Vivo/Vivo_Assets/v1/robot.usd", currently is still hardcoded in the code
        #usd_path=f"/home/simon/Repositories/RL_experiment/IsaacLab-dhkim/DHKimTests/RobotRL/Vivo/Vivo_Assets/v1/robot.usd",
        usd_path=f"D:\CodeProjects\Isaac\IsaacLab-dhkim\DHKimTests\RobotRL\Vivo\Vivo_Assets\v1\robot.usd",

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
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.4),
        joint_pos={
            # Front left leg joints
            ".*fl_roll": 0.0,     # Hip roll (abduction/adduction)
            ".*fl_pitch": -0.61,    # Hip pitch (flexion/extension) 
            ".*fl_fe": 0.35,   # Knee (flexion/extension)
            
            # Front right leg joints
            ".*fr_roll": 0.0,      # Hip roll
            ".*fr_pitch": -0.61,    # Hip pitch
            ".*fr_fe": 0.35,      # Knee (note this uses 'fe' instead of shank)
            
            # Back left leg joints
            ".*bl_roll": 0.0,     # Hip roll
            ".*bl_pitch": -0.61,   # Hip pitch
            ".*bl_fe": 0.35,       # Knee
            
            # Back right leg joints
            ".*br_roll": 0.0,     # Hip roll
            ".*br_pitch": -0.61,   # Hip pitch
            ".*br_fe": 0.35     # Knee (using shank naming)
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "base_legs": DCMotorCfg(
            joint_names_expr=[".*_roll", ".*_pitch", ".*_fe"],
            effort_limit=23.5,
            saturation_effort=23.5,
            velocity_limit=30.0,
            stiffness=25.0,
            damping=0.5,
            friction=0.0,
        ),
    },
)







'''
VIVO_ACTUATOR_CFG = ActuatorNetMLPCfg(
    joint_names_expr=[".*_roll", ".*_pitch", ".*_fe"],
    network_file=f"{ISAACLAB_NUCLEUS_DIR}/ActuatorNets/Unitree/unitree_go1.pt",
    pos_scale=-1.0,
    vel_scale=1.0,
    torque_scale=1.0,
    input_order="pos_vel",
    input_idx=[0, 1, 2],
    effort_limit=23.7,  # taken from spec sheet
    velocity_limit=30.0,  # taken from spec sheet
    saturation_effort=23.7,  # same as effort limit
)

VIVO_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="D:/CodeProjects/Isaac/Vivo/v1/robot.usd",
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
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.4),
        joint_pos={
            # Front left leg joints
            "fl_roll": 0.0,     # Hip roll (abduction/adduction)
            "fl_pitch": -0.61,    # Hip pitch (flexion/extension) 
            "fl_fe": 0.35,   # Knee (flexion/extension)
            
            # Front right leg joints
            "fr_roll": 0.0,      # Hip roll
            "fr_pitch": -0.61,    # Hip pitch
            "fr_fe": 0.35,      # Knee (note this uses 'fe' instead of shank)
            
            # Back left leg joints
            "bl_roll": 0.0,     # Hip roll
            "bl_pitch": -0.61,   # Hip pitch
            "bl_fe": 0.35,       # Knee
            
            # Back right leg joints
            "br_roll": 0.0,     # Hip roll
            "br_pitch": -0.61,   # Hip pitch
            "br_fe": 0.35     # Knee (using shank naming)
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "base_legs": DCMotorCfg(
            joint_names_expr=[".*_roll", ".*_pitch", ".*_fe"],
            effort_limit=23.5,
            saturation_effort=23.5,
            velocity_limit=30.0,
            stiffness=25.0,
            damping=0.5,
            friction=0.0,
        ),
    },
)
'''