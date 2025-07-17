import argparse

from isaaclab.app import AppLauncher
import numpy as np

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on using the differential IK controller.")
parser.add_argument("--robot", type=str, default="prestoe2d", help="Name of the robot.")
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

# from isaaclab.markers.config import FRAME_MARKER_CFG

##
# Pre-defined configs
##
from Prestoe2D import Prestoe2D_CFG  # isort:skip


@configclass
class TableTopSceneCfg(InteractiveSceneCfg):

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # articulation
    robot = Prestoe2D_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability.
    robot = scene["robot"]

    robot_entity_cfg = SceneEntityCfg("robot", joint_names=[".*_hip", ".*_knee", ".*_ankle"])
    # robot_entity_cfg = SceneEntityCfg("robot", joint_names=["l_hip", "l_knee", "l_ankle"])
    robot_entity_cfg.resolve(scene)

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0
    # Simulation loop
    while simulation_app.is_running():
        # reset
        if count % 550 == 0:
            # reset time
            count = 0
            # reset joint state
            joint_pos = robot.data.default_joint_pos.clone()
            joint_vel = robot.data.default_joint_vel.clone()
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            robot.reset()

        # # apply actions 
        curr_time = count * sim_dt
        # 0, 1, 2: x, y, pitch
        # 3, 4: hip (left, right)
        # 5, 6: knee (left, right)
        # 7, 8: ankle (left, right)

        # print(f"joint id: {robot_entity_cfg.joint_ids}")
        # print(f"joint pos: {robot.data.joint_pos[:, robot_entity_cfg.joint_ids]}")

        joint_pos_des = 0.5 + 0.5 * np.sin(2*np.pi*curr_time)
        robot.set_joint_position_target(joint_pos_des, joint_ids=[7, 8])

        # # set joint torque
        # jtorque_des = 50. * np.sin(2*np.pi*curr_time)
        # robot.set_joint_effort_target(jtorque_des, joint_ids=robot_entity_cfg.joint_ids)
 
        # robot.set_joint_position_target(joint_pos_des, joint_ids=robot_entity_cfg.joint_ids)
        scene.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        count += 1
        # update buffers
        scene.update(sim_dt)

def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(dt=0.01)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
    # Design scene
    scene_cfg = TableTopSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
