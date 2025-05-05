
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Four Bar Linkage simulation")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch 

import isaacsim.core.utils.prims as prim_utils

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext


from fourbar import FOURBAR_CFG

def create_scene():
    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func('/World/defaultGroundPlane', cfg_ground)

    cfg_light = sim_utils.DomeLightCfg(intensity=2000, color=(0.8, 0.8, 0.8))
    cfg_light.func('/World/defaultLight', cfg_light)

    origins = [[0,0,0]]

    origin = prim_utils.create_prim("/World/Origin", "Xform", translation=origins[0])

    fb_cfg = FOURBAR_CFG.copy() # type: ignore
    fb_cfg.prim_path = "/World/Origin/FourBar"
    fb = Articulation(cfg = fb_cfg)

    entities = {'fb' : fb}
    return entities, origins

def addforce(robot):
    ef = torch.ones_like(robot.data.joint_pos) * 0.9
    robot.set_joint_effort_target(ef)
    robot.write_data_to_sim()

# import keyboard as keyboard
# from pynput import keyboard
# from isaaclab.controllers import OperationalSpaceController
from isaaclab.devices import Se2Keyboard
    



# def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, Articulation], origins: torch.Tensor):
#     robot = entities["fb"]
#     sim_dt = sim.get_physics_dt()
#     count = 0
#     cap = 300
#     #ji, jn = robot.find_joints('Revolute_1')
#     #ji = ji[0]
#     while simulation_app.is_running():

        #kb  = Se2Keyboard()
        #kb.reset()

        # kb.add_callback("SPACE", lambda: addforce(robot))
        
        # robot.set_joint_velocity_target(torch.tensor([[1000000]], dtype=torch.float32), joint_ids=[ji])

        # if count == cap:

        #     count =0

        #     root_state = robot.data.default_root_state.clone()
        #     root_state[:, :3] += origins
        #     robot.write_root_pose_to_sim(root_state[:, :7])
        #     robot.write_root_velocity_to_sim(root_state[:, 7:])
        #     # set joint positions with some noise
        #     joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
        #     robot.write_joint_state_to_sim(joint_pos, joint_vel)
        #     # clear internal buffers
        #     robot.reset()
        #     print("[INFO]: Resetting robot state...")

        #keyboard setup 


        # The event listener will be running in this block
        # with keyboard.Events() as events:
        #     for event in events:
        #         if event.key == keyboard.Key.space:
                #     addforce(robot)
                # else:
                #     print('Received event {}'.format(event))
     
        # How did i break this?????
        #addforce(robot)

        # ef = torch.rand_like(robot.data.joint_pos) * 1.4
        # ef = torch.ones_like(robot.data.joint_pos) * 0.87
        # robot.set_joint_effort_target(ef)
        # robot.write_data_to_sim()
    
        #efforts = torch.randn_like(robot.data.joint_pos) * 5.0
        # robot.set_joint_effort_target(efforts)
        # robot.write_data_to_sim()
        # sim.step()
        # count += 1
        # robot.update(sim_dt)

def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, Articulation], origins: torch.Tensor):
    robot = entities["fb"]
    sim_dt = sim.get_physics_dt()
    count = 0
    while simulation_app.is_running():
        count += 1
        sim.step()
        robot.update(sim_dt)

        

def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)

    sim.set_camera_view([0.0, 1.0, 2.0], [0.0, 0.0, 0.0]) # type: ignore
    scene_entities, scene_origins = create_scene()
    scene_origins = torch.tensor(scene_origins, device=sim.device)
 
    sim.reset()
    print("[INFO]: Setup complete...")
    # Run the simulator

    # Dummy code for testing the environment setup
    #while simulation_app.is_running():
     #   sim.step()

    run_simulator(sim, scene_entities, scene_origins)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()



