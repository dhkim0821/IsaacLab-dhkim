from isaaclab.app import AppLauncher
from isaaclab.scene import Scene
from isaaclab.utils import configclass
from isaaclab.assets import ArticulationCfg
from isaaclab.sim.spawners import GridSpawnerCfg
from isaaclab.sim.spawners.from_files import UsdFileCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sim import SimulationContext

from isaaclab.envs import BaseEnvCfg, VecEnv
import torch
from Prestoe2D import Prestoe2D_CFG  # isort:skip

# --- Launch Isaac Sim ---
app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

# --- Configuration ---
@configclass
class RobotCfg(ArticulationCfg):
    meta_info.urdf_path = "/home/daros/Repositories/IsaacLab-dhkim/DHKimTests/PresToe2D/model/Prestoe2D.urdf",
    init_state.pos = (0.0, 0.0, 0.5)

@configclass
class MySceneCfg(BaseEnvCfg):
    scene: Scene.Config = Scene.Config(
        num_envs=16,
        env_spacing=2.0,
        replicate_physics=True,
        replication_type="grid",
        spawn_config=GridSpawnerCfg(spacing=2.0, pattern="grid"),
        assets={"robot": RobotCfg()},
    )

# --- Setup VecEnv ---
env = VecEnv(MySceneCfg())

# --- Reset environment to spawn all robots ---
env.reset()

# --- Main loop ---
sim = SimulationContext()
for _ in range(1000):
    obs = env.step(torch.zeros_like(env.action_space.sample()))  # send dummy zero actions
    env.render()

# --- Close simulation ---
simulation_app.close()
