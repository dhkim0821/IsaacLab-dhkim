# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform

from .prestoe2d_env_cfg import Prestoe2dEnvCfg
import numpy as np

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    # Add the current directory to the system path
    sys.path.append(current_dir)

class Prestoe2dEnv(DirectRLEnv):
    cfg: Prestoe2dEnvCfg

    def __init__(self, cfg: Prestoe2dEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._joint_idx, _ = self.robot.find_joints(".*")
        self._act_joint_idx = self._joint_idx[3:].copy()  # exclude root joint 
        self._num_act_joints = len(self._act_joint_idx)

        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel
        # read data
        filename = current_dir + "/walking_2D.npz"
        np_qpos_ref = np.load(filename, allow_pickle=True)["qpos"]
        self.qpos_ref = torch.tensor(np_qpos_ref, dtype=torch.float32, device=self.device)
        self.ref_seq_scale = 4.0 # maybe 2?

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            print("** Warning: CPU simulation does not support collision filtering, this may lead to performance issues **")
            self.scene.filter_collisions(global_prim_paths=[])
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        # print(f"Actions: {self.actions}, Action shape: {self.actions.shape}")
        self.robot.set_joint_effort_target(self.actions[:, :self._num_act_joints] * self.cfg.action_scale, joint_ids=self._act_joint_idx)

    def _get_observations(self) -> dict:
        obs = torch.cat(
            (
                # self.joint_pos.unsqueeze(dim=1),
                # self.joint_vel.unsqueeze(dim=1),
                self.joint_pos,
                self.joint_vel,
             ),
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        # reset_terminated = self.reset_buf.clone()
        self._get_dones() 
        # print(f"terminate buf: {self.reset_buf[:]}")  # Debugging line
        # print(f"[get reward] reset terminated: {self.reset_terminated[:]}")  # Debugging line
        total_reward = compute_rewards(
            self.cfg.rew_scale_alive,
            self.cfg.rew_scale_terminated,
            self.cfg.rew_scale_xvel,
            self.cfg.rew_scale_height_maintenance,
            self.cfg.rew_scale_jvel,
            self.cfg.rew_scale_imitation,
            self.qpos_ref,
            self.episode_length_buf,
            self.ref_seq_scale,
            self.joint_pos,
            self.joint_vel,
            self.reset_terminated,
        )
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # height check
        root_z = self.joint_pos[:, 1]
        self.reset_terminated = root_z < self.cfg.min_zpos
        # print(f"Root z position: {root_z[:]}")
        # print(f"Reset terminated: {self.reset_terminated[:]}")

        # joint position check
        non_root_joints = self.joint_pos[:, 3:]  # exclude root joint
        out_of_bounds = torch.any(torch.abs(non_root_joints) > self.cfg.max_jpos, dim=1)

        return out_of_bounds, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_pos[:, 3:] += sample_uniform(
            lower=torch.tensor(
                [
                    self.cfg.initial_hip_range[0],
                    self.cfg.initial_hip_range[0],
                    self.cfg.initial_knee_range[0],
                    self.cfg.initial_knee_range[0],
                    self.cfg.initial_ankle_range[0],
                    self.cfg.initial_ankle_range[0],
                ],
                device=joint_pos.device,
            ),
            upper=torch.tensor(
                [
                    self.cfg.initial_hip_range[1],
                    self.cfg.initial_hip_range[1],
                    self.cfg.initial_knee_range[1],
                    self.cfg.initial_knee_range[1],
                    self.cfg.initial_ankle_range[1],
                    self.cfg.initial_ankle_range[1],
                ],
                device=joint_pos.device,
            ),
            size=joint_pos[:, 3:].shape,
            device=joint_pos.device,
        )

        joint_vel = self.robot.data.default_joint_vel[env_ids]

        # joint_pos[:, 0] += self.scene.env_origins[env_ids][0]
        # joint_pos[:, 1] += self.scene.env_origins[env_ids][2]

        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel

        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)


@torch.jit.script
def compute_rewards(
    rew_scale_alive: float,
    rew_scale_terminated: float,
    rew_scale_xvel: float,
    rew_scale_height_maintenance: float,
    rew_scale_joint_vel: float,
    rew_scale_imitation: float,
    qpos_ref: torch.Tensor,
    episode_length_buf: torch.Tensor,
    ref_seq_scale: float,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    reset_terminated: torch.Tensor,
):
    mask = (reset_terminated.float() < 0.5)
    # print(f"reset_terminated: {reset_terminated[:]}")  # Debugging line
    # print(f"Mask: {mask[:]}")

    rew_alive = torch.zeros(episode_length_buf.shape, device=episode_length_buf.device, dtype=torch.float32)
    rew_forward_vel = torch.zeros(episode_length_buf.shape, device=episode_length_buf.device, dtype=torch.float32)
    rew_height = torch.zeros(episode_length_buf.shape, device=episode_length_buf.device, dtype=torch.float32)

    rew_alive[mask] = rew_scale_alive
    rew_termination = rew_scale_terminated * reset_terminated.float()


    rew_forward_vel[mask] = rew_scale_xvel * torch.clamp(joint_vel[mask, 0], min=0, max=1.3)  # clamp to [0, 1.3] 
    rew_height[mask] = rew_scale_height_maintenance * torch.clamp(joint_pos[mask, 1] - 0.6, min=0.0)
    
    rew_joint_vel = rew_scale_joint_vel * torch.sum(torch.abs(joint_vel[:, 3:]), dim=-1)

    # imitation reward
    ref_seq_len = qpos_ref.size(0)
    phase = (episode_length_buf / ref_seq_scale) % ref_seq_len
    ref_idx = torch.floor(phase).long()
    ref_idx = torch.clamp(ref_idx, min=0, max=ref_seq_len - 1)
    # print(f"Reference index: {ref_idx[0]}, Phase: {phase[0]}, Episode length: {episode_length_buf[0]}, Ref seq scale: {ref_seq_scale}, ref seq len: {ref_seq_len}")
    rew_imitation = rew_scale_imitation * torch.sum(torch.abs(joint_pos[:, 3:] - qpos_ref[ref_idx, 3:]), dim=-1)

    total_reward = rew_alive + rew_termination + rew_forward_vel + rew_height + rew_joint_vel + rew_imitation
    # for i in range (3):
    #     print(f"Rewards: Alive: {rew_alive[i]}, Termination: {rew_termination[i]}, Forward Vel: {rew_forward_vel[i]}, Height: {rew_height[i]}, Joint Vel: {rew_joint_vel[i]}, Imitation: {rew_imitation[i]}")

    return total_reward
