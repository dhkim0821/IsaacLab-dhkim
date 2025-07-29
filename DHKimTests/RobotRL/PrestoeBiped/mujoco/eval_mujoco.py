from mjenv import MujocoBipedalEnv
import os
import numpy as np
import time
import torch
import mujoco
import mujoco.viewer 
# get current file path
file_path = os.path.dirname(os.path.abspath(__file__)) + "/robot/prestoebiped_scene.xml"

model_dir = "/home/simon/Repositories/RL_experiment/IsaacLab-dhkim/DHKimTests/RobotRL/PrestoeBiped/saved_policy/policy.pt"

env = MujocoBipedalEnv(model_path=file_path)


model = torch.jit.load(model_dir)

cur_time = time.time()
obs, _, _ = env.reset()
with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
    root_body = env.model.body('torso_link').id# or whatever your XML calls it
    with viewer.lock():                     # lock while you tweak cam parameters
        viewer.cam.trackbodyid = root_body
        viewer.cam.type        = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.fixedcamid  = -1        # disable fixed cams
        # optional: set how far/how high/angle you want to sit behind
        viewer.cam.distance    = 3.0       # meters behind the body
        viewer.cam.elevation   = 0     # degrees above/below
        viewer.cam.azimuth     = 90     # degrees around body
        viewer.cam.lookat[2]  = 0
    while True:
        # with viewer.lock():
        #     # data.xpos is [nbody×3], so grab the root position
        #     viewer.cam.lookat[:] = env.data.xpos[root_body]
        viewer.sync()
        obs = torch.tensor(obs, dtype=torch.float32).to('cpu')  # Convert observations to tensor
        with torch.no_grad():
            action = model(obs.unsqueeze(0)).squeeze(0).numpy()  # Get action from the model
            
        obs, dis, tau_squareOverDist = env.step(action)  # Take a step with zero action
        print("dis:", dis, "tau_squareOverDist:", tau_squareOverDist)
        speed  = np.linalg.norm(env.data.qvel[:2])
        print("speed: ", speed) 
        # print("Observations:", obs)
        time_elapsed = time.time() - cur_time
        time.sleep(max(0, 0.05 - time_elapsed))  # Sleep to maintain the desired timestep
        cur_time = time.time()  # Update current time
    
    
    
