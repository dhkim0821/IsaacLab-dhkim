import isaaclab.sim as sim_utils 
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

import os

# USD_PATH = "/home/isaac/Documents/Github/IsaacLab-dhkim/DHKimTests/RobotRL/PrestoeClosedLoopArticulation/newprestoe/prestoeCC2/prestoe_CC2.usd"
USD_PATH = "/home/isaac/Documents/Github/IsaacLab-dhkim/DHKimTests/RobotRL/PrestoeClosedLoopArticulation/newprestoe_simplified_v3/prestoeCC2/prestoe_CC2.usd"

#BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#USD_PATH = os.path.join(BASE_DIR, "../../RobotTests", "prestoebiped_minimal.usd")

PRESTOE_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=USD_PATH,
        activate_contact_sensors=True,
        rigid_props = sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        # all articulation props set to default
        articulation_props= sim_utils.ArticulationRootPropertiesCfg(
            articulation_enabled=True,
            enabled_self_collisions=True,
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    soft_joint_pos_limit_factor=0.9,
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
    # actuators= {
    #     "hips": ImplicitActuatorCfg(
    #         joint_names_expr=[".*_hipyaw", ".*_hiproll", ".*_hippitch", "torsoyaw"],
    #         effort_limit=70,
    #         velocity_limit=30.0,
    #         stiffness={
    #             ".*_hipyaw": 150.0,
    #             ".*_hiproll": 150.0,
    #             ".*_hippitch": 200.0,
    #             "torsoyaw": 200.0,
    #         },
    #         damping={
    #             ".*_hipyaw": 5.0,
    #             ".*_hiproll": 5.0,
    #             ".*_hippitch": 5.0,
    #             "torsoyaw": 5.0,
    #         },
    #         armature = 0.0049167
    #     ),
    #     "legs": ImplicitActuatorCfg(
    #         joint_names_expr=[".*_knee"],
    #         effort_limit=200,
    #         velocity_limit=50.0,
    #         stiffness={ ".*_knee": 200.0 },
    #         damping={ ".*_knee": 5.0 },
    #         armature = 0.09712
    #     ),
    #     "feet": ImplicitActuatorCfg(
    #         joint_names_expr=[".*_anklepitch"],
    #         effort_limit=100,
    #         velocity_limit=30.0,
    #         stiffness={".*_anklepitch": 20.0 },
    #         damping={".*_anklepitch": 4.0},
    #         armature = 0.3
    #     ),
    #     "toe": ImplicitActuatorCfg(
    #         joint_names_expr=[".*_ankleroll", ".*_toepitch"],
    #         effort_limit=40,
    #         velocity_limit=20.0,
    #         stiffness={".*_ankleroll": 20.0, ".*_toepitch": 20.0},
    #         damping={".*_ankleroll": 4.0, ".*_toepitch": 4.0},
    #         armature = 0.3
    #     ),
    # }
    actuators={
        "R_AL_pulley_pitch_acc" : ImplicitActuatorCfg(
            joint_names_expr= ["R_AL_pulley_pitch"],
            stiffness=25, damping=0.5,
            friction=0,
            effort_limit_sim=120.0,
            velocity_limit_sim=80.0, armature = 0.3,
        ),
        # "R_toe_pulley_pitch_acc" : ImplicitActuatorCfg(
        #     joint_names_expr= ["R_toe_pulley_pitch"],
        #     stiffness=25, damping=0.5,
        #     friction=0,
        #     effort_limit_sim=120.0,
        #     velocity_limit_sim=80.0, armature = 0.3,
        # ),
        "R_toepitch_acc" : ImplicitActuatorCfg(
            joint_names_expr= ["R_toepitch"],
            stiffness=25, damping=0.5,
            friction=0,
            effort_limit_sim=120.0,
            velocity_limit_sim=80.0, armature = 0.3,
        ),
        "R_AR_pulley_pitch_acc" : ImplicitActuatorCfg(
            joint_names_expr= ["R_AR_pulley_pitch"],
            stiffness=25, damping=0.5,
            friction=0,
            effort_limit_sim=120.0,
            velocity_limit_sim=80.0, armature = 0.3,
        ),
        "L_AL_pulley_pitch_acc" : ImplicitActuatorCfg(
            joint_names_expr= ["L_AL_pulley_pitch"],
            stiffness=25, damping=0.5,
            friction=0,
            effort_limit_sim=120.0,
            velocity_limit_sim=80.0, armature = 0.3,
        ),
        # "L_toe_pulley_pitch_acc" : ImplicitActuatorCfg(
        #     joint_names_expr= ["L_toe_pulley_pitch"],
        #     stiffness=25, damping=0.5,
        #     friction=0,
        #     effort_limit_sim=120.0,
        #     velocity_limit_sim=80.0, armature = 0.3,
        # ),
        "L_toepitch_acc" : ImplicitActuatorCfg(
            joint_names_expr= ["L_toepitch"],
            stiffness=25, damping=0.5,
            friction=0,
            effort_limit_sim=120.0,
            velocity_limit_sim=80.0, armature = 0.3,
        ),
        "L_AR_pulley_pitch_acc" : ImplicitActuatorCfg(
            joint_names_expr= ["L_AR_pulley_pitch"],
            stiffness=25, damping=0.5,
            friction=0,
            effort_limit_sim=120.0,
            velocity_limit_sim=80.0, armature = 0.3,
        ),
        "torsoyaw_acc" : ImplicitActuatorCfg(
            joint_names_expr= ["torsoyaw"],
            stiffness=25, damping=0.5,
            friction=0,
            effort_limit_sim=40.0,
            velocity_limit_sim=30.0, armature = 0.3,
        ),
        "R_hipyaw_acc" : ImplicitActuatorCfg(
            joint_names_expr= ["R_hipyaw"],
            stiffness=25, damping=0.5,
            friction=0,
            effort_limit_sim=40.0,
            velocity_limit_sim=30.0, armature = 0.3,
        ),
        "L_hipyaw_acc" : ImplicitActuatorCfg(
            joint_names_expr= ["L_hipyaw"],
            stiffness=25, damping=0.5,
            friction=0,
            effort_limit_sim=40.0,
            velocity_limit_sim=30.0, armature = 0.3,
        ),
        "R_hiproll_acc" : ImplicitActuatorCfg(
            joint_names_expr=["R_hiproll"],
            stiffness=25, damping=0.5,
            friction=0,
            effort_limit_sim=40.0,
            velocity_limit_sim=30.0, armature = 0.3,
        ),
        "L_hiproll_acc" : ImplicitActuatorCfg(
            joint_names_expr=["L_hiproll"],
            stiffness=25, damping=0.5,
            friction=0,
            effort_limit_sim=40.0,
            velocity_limit_sim=30.0, armature = 0.3,
        ),
        "R_hippitch_acc" : ImplicitActuatorCfg(
            joint_names_expr=["R_hippitch"],
            stiffness=25, damping=0.5,
            friction=0,
            effort_limit_sim=40.0,
            velocity_limit_sim=30.0, armature = 0.3,
        ),
        "L_hippitch_acc" : ImplicitActuatorCfg(
            joint_names_expr=["L_hippitch"],
            stiffness=25, damping=0.5,
            friction=0,
            effort_limit_sim=40.0,
            velocity_limit_sim=30.0, armature = 0.3,
        ),
        "R_shoulderpitch_acc" : ImplicitActuatorCfg(
            joint_names_expr=["R_shoulderpitch"],
            stiffness=25, damping=0.5,
            friction=0,
            effort_limit_sim=40.0,
            velocity_limit_sim=30.0, armature = 0.3,
        ),
        "R_shoulderroll_acc" : ImplicitActuatorCfg(
            joint_names_expr=["R_shoulderroll"],
            stiffness=25, damping=0.5,
            friction=0,
            effort_limit_sim=40.0,
            velocity_limit_sim=30.0, armature = 0.3,
        ),
        "R_shoulderyaw_acc" : ImplicitActuatorCfg(
            joint_names_expr=["R_shoulderyaw"],
            stiffness=25, damping=0.5,
            friction=0,
            effort_limit_sim=40.0,
            velocity_limit_sim=30.0, armature = 0.3,
        ),
        "R_elbowpitch_acc" : ImplicitActuatorCfg(
            joint_names_expr=["R_elbowpitch"],
            stiffness=25, damping=0.5,
            friction=0,
            effort_limit_sim=40.0,
            velocity_limit_sim=30.0, armature = 0.3,
        ),
        "L_shoulderpitch_acc" : ImplicitActuatorCfg(
            joint_names_expr=["L_shoulderpitch"],
            stiffness=25, damping=0.5,
            friction=0,
            effort_limit_sim=40.0,
            velocity_limit_sim=30.0, armature = 0.3,
        ),
        "L_shoulderroll_acc" : ImplicitActuatorCfg(
            joint_names_expr=["L_shoulderroll"],
            stiffness=25, damping=0.5,
            friction=0,
            effort_limit_sim=40.0,
            velocity_limit_sim=30.0, armature = 0.3,
        ),
        "L_shoulderyaw_acc" : ImplicitActuatorCfg(
            joint_names_expr=["L_shoulderyaw"],
            stiffness=25, damping=0.5,
            friction=0,
            effort_limit_sim=40.0,
            velocity_limit_sim=30.0, armature = 0.3,
        ),
        "L_elbowpitch_acc" : ImplicitActuatorCfg(
            joint_names_expr=["L_elbowpitch"],
            stiffness=25, damping=0.5,
            friction=0,
            effort_limit_sim=40.0,
            velocity_limit_sim=30.0, armature = 0.3,
        ),
        "R_knee" : ImplicitActuatorCfg(
            joint_names_expr=["R_knee"],
            stiffness=25, damping=0.5,
            friction=0,
            effort_limit_sim=40.0,
            velocity_limit_sim=30.0, armature = 0.3,
        ),
        "L_knee" : ImplicitActuatorCfg(
            joint_names_expr=["L_knee"],
            stiffness=25, damping=0.5,
            friction=0,
            effort_limit_sim=40.0,
            velocity_limit_sim=30.0, armature = 0.3,
        ),
    }
)

PRESTOE_CFG = PRESTOE_CFG.copy()