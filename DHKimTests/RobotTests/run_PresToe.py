import argparse

from isaaclab.app import AppLauncher
import numpy as np

# add argparse arguments
parser = argparse.ArgumentParser(description="PresToe")
parser.add_argument("--robot", type=str, default="franka_panda", help="Name of the robot.")
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

##
# Pre-defined configs
##
from PresToe import PRESTOE_CFG  # isort:skip

@configclass
class TableTopSceneCfg(InteractiveSceneCfg):

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.85)),
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # articulation
    robot = PRESTOE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability.
    robot = scene["robot"]

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0

    robot_entity_cfg = SceneEntityCfg("robot", joint_names=[".*_knee"])
    robot_entity_cfg.resolve(scene)


    # Simulation loop
    while simulation_app.is_running():
        # reset
        if count % 550 == 0:
            # reset time
            count = 0
            # reset joint state
            joint_pos = robot.data.default_joint_pos.clone()
            joint_vel = robot.data.default_joint_vel.clone()
            print(joint_pos)
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            robot.reset()

        # apply actions
        # robot.set_joint_position_target(joint_pos_des, joint_ids=robot_entity_cfg.joint_ids)

        # swing robot's knee 
        curr_time = count * sim_dt
        joint_pos_des = 0.5 + 0.5 * np.sin(2*np.pi*curr_time)
        robot.set_joint_position_target(joint_pos_des, joint_ids=robot_entity_cfg.joint_ids)

        # set joint torque
        jtorque_des = 50. * np.sin(2*np.pi*curr_time)
        # robot.set_joint_effort_target(jtorque_des, joint_ids=robot_entity_cfg.joint_ids)
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
