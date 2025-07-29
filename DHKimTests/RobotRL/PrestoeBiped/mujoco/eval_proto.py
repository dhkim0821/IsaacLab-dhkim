from protoenv import MujocoBipedalEnv
import os
import numpy as np
import time
import torch
from protomotions.agents.ppo.agent import PPO 
# get current file path
file_path = os.path.dirname(os.path.abspath(__file__)) + "/robot/prestoebiped_scene.xml"

model_dir = '/home/simon/Repositories/RL_experiment/ProtoMotions/results/prestoe_walk/lightning_logs/version_0/last.ckpt'
state_dict = torch.load(model_dir, map_location =  torch.device("cuda:0"))


env = MujocoBipedalEnv(model_path=file_path)


model = torch.jit.load(model_dir)

cur_time = time.time()
obs = env.reset()
while True:
    env.render()  # Render the environment
    obs = torch.tensor(obs, dtype=torch.float32).to('cpu')  # Convert observations to tensor
    with torch.no_grad():
        action = model(obs.unsqueeze(0)).squeeze(0).numpy()  # Get action from the model
        
    obs = env.step(action)  # Take a step with zero action
    # print("Observations:", obs)
    time_elapsed = time.time() - cur_time
    time.sleep(max(0, 0.05 - time_elapsed))  # Sleep to maintain the desired timestep
    cur_time = time.time()  # Update current time
    
    
    
