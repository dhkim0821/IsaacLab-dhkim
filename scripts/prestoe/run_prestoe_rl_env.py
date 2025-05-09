import argparse
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Running the PrestoeBiped RL environment.")
parser.add_argument("--num_envs", type=int, default=1, 
                    help="Number of environments to spawn.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

from isaaclab.envs import ManagerBasedRLEnv

from prestoe_env_cfg import PrestoeEnvCfg

def main():
    """Main function."""
    # create environment configuration
    env_cfg = PrestoeEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    # setup RL environment
    env = ManagerBasedRLEnv(cfg=env_cfg)

    # simulate physics
    count = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            # reset
            if count % 1000000 == 0:
                count = 0
                env.reset()
                print("-" * 80)
                print("[INFO]: Resetting environment...")
                # set joint effort to zero
                joint_cmd = torch.zeros_like(env.action_manager.action)
                # print("[INFO]: Joint efforts: ", joint_cmd)
                obs, rew, terminated, truncated, info = env.step(joint_cmd)

                print(terminated, rew)
                pause = input("Press enter to continue...")
                count += 1
            else:
                joint_cmd = torch.zeros_like(env.action_manager.action)
                # print("[INFO]: Joint efforts: ", joint_cmd)
                obs, rew, terminated, truncated, info = env.step(joint_cmd)
                print(terminated, rew)
                # print current orientation of pole
                # print("[Env 0]: Pole joint: ", obs["policy"][0][:])
                # update counter
                count += 1
                # print(count)

    # # close the environment
    # env.close()


if __name__ == "__main__":
    
    main()

    simulation_app.close()