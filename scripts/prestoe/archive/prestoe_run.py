import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Prestoe")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import torch

import isaacsim.core.utils.prims as prim_utils

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

from prestoe_config import PRESTOE_CFG


def create_environemnt():
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)

    light_cfg = sim_utils.DomeLightCfg(intensity=1000)
    light_cfg.func("/World/defaultDomeLight", light_cfg)

    origins = [[0,0,0]]

    for idx in range(len(origins)):
        prim_utils.create_prim(f"/World/Origin{idx + 1}", "Xform", translation=origins[idx])

    usd_cfg = PRESTOE_CFG.copy() # type: ignore
    usd_cfg.prim_path = "/World/Origin1/Prestoe"
    prestoe = Articulation(usd_cfg)

    scene_entities = {
        "prestoe" : prestoe,
    }

    return scene_entities, origins 

def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, Articulation], origins: torch.tensor):
    prestoe = entities["prestoe"]

    sim_dt = sim.get_physics_dt()

    while simulation_app.is_running():
        prestoe.write_data_to_sim()
        sim.step()
        prestoe.update(sim_dt)


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([0.0, 2.0, 2.0], [0.0, 0.0, 0.0]) # type: ignore
    entities, origins = create_environemnt()
    sim.reset()

    run_simulator(sim, entities, origins)


if __name__ == "__main__":
    main()

    simulation_app.close()