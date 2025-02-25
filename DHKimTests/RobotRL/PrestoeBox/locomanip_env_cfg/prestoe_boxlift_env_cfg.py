import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

import DHKimTests.RobotRL.PrestoeBox.PrestoeBox_mdp as mdp
from DHKimTests.RobotRL.PrestoeBox.locomanip_env_cfg.terrain_box_env_cfg import TerrainBoxEnvCfg 
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip

@configclass
class PrestoeLiftEnvCfg(TerrainBoxEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.num_envs = 900

        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/torso_link",
            debug_vis=False,
            # debug_vis=True,
            visualizer_cfg=marker_cfg,
            target_frames= [
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/R_elbowpitch_link",
                    name="robot_right_hand",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, -0.3],
                    )
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/L_elbowpitch_link",
                    name="robot_left_hand",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, -0.3],
                    ),
                ),
            ]
        )


@configclass
class PrestoeLiftEnvCfg_PLAY(PrestoeLiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 30
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
