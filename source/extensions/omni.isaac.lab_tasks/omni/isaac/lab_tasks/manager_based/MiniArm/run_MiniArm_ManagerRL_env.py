import argparse
from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Running the MiniArm RL environment.")
parser.add_argument("--num_envs", type=int, default=2, 
                    help="Number of environments to spawn.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
import torch
from omni.isaac.lab.envs import ManagerBasedRLEnv
from omni.isaac.lab_tasks.manager_based.classic.cartpole.cartpole_env_cfg import CartpoleEnvCfg
print("-------------------")
print("dir of cartpole")
print(dir(CartpoleEnvCfg))

# from omni.isaac.lab_tasks.manager_based.MiniArm.MiniArm_ManagerRL_env_cfg import MiniArmEnvCfg
from omni.isaac.lab_tasks.manager_based.MiniArm.MiniArm_ManagerRL_env_cfg import MiniArmEnvCfg
# from omni.isaac.lab_tasks.manager_based.MiniArm import MiniArmEnvCfg
# from omni.isaac.lab_tasks.manager_based.classic.cartpole.MiniArm_ManagerRL_env_cfg import MiniArmEnvCfg

print("dir of MiniArm")
print(dir(MiniArmEnvCfg))
print("-------------------")
#

def main():
    """Main function."""
    # create environment configuration
    env_cfg = MiniArmEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    # setup RL environment
    env = ManagerBasedRLEnv(cfg=env_cfg)

    # simulate physics
    count = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            # reset
            if count % 300 == 0:
                count = 0
                env.reset()
                print("-" * 80)
                print("[INFO]: Resetting environment...")
            # sample random actions
            joint_efforts = torch.randn_like(env.action_manager.action)
            print("[INFO]: Joint efforts: ", joint_efforts)
            # step the environment
            obs, rew, terminated, truncated, info = env.step(joint_efforts)
            # print current orientation of pole
            # print("[Env 0]: Pole joint: ", obs["policy"][0][:])
            # update counter
            count += 1

    # close the environment
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()