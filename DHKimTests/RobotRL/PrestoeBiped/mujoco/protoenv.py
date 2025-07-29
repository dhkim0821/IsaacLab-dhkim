import mujoco
import mujoco.viewer
import numpy as np
import torch
from torch import Tensor

def wxyz_to_xyzw(quat: Tensor):
    shape = quat.shape
    flat_quat = quat.view(-1, 4)
    flat_quat = flat_quat[:, [1, 2, 3, 0]]
    return flat_quat.view(shape)

@torch.jit.script
def quat_rotate(q: Tensor, v: Tensor, w_last: bool) -> Tensor:
    shape = q.shape
    flat_q = q.reshape(-1, shape[-1])
    flat_v = v.reshape(-1, v.shape[-1])
    if w_last:
        q_w = flat_q[:, -1]
        q_vec = flat_q[:, :3]
    else:
        q_w = flat_q[:, 0]
        q_vec = flat_q[:, 1:]
    a = flat_v * (2.0 * q_w**2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, flat_v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = (
        q_vec
        * torch.bmm(
            q_vec.reshape(flat_q.shape[0], 1, 3), flat_v.reshape(flat_q.shape[0], 3, 1)
        ).squeeze(-1)
        * 2.0
    )
    return (a + b + c).reshape(v.shape)

@torch.jit.script
def calc_heading(q: Tensor, w_last: bool) -> Tensor:
    # calculate heading direction from quaternion
    # the heading is the direction on the xy plane
    # q must be normalized
    ref_dir = torch.zeros_like(q[..., 0:3])
    ref_dir[..., 0] = 1
    rot_dir = quat_rotate(q, ref_dir, w_last)

    heading = torch.atan2(rot_dir[..., 1], rot_dir[..., 0])
    return heading

@torch.jit.script
def normalize(x, eps: float = 1e-9):
    return x / x.norm(p=2, dim=-1).clamp(min=eps, max=None).unsqueeze(-1)

@torch.jit.script
def quat_unit(a):
    return normalize(a)

@torch.jit.script
def quat_from_angle_axis(angle: Tensor, axis: Tensor, w_last: bool) -> Tensor:
    theta = (angle / 2).unsqueeze(-1)
    xyz = normalize(axis) * theta.sin()
    w = theta.cos()
    if w_last:
        return quat_unit(torch.cat([xyz, w], dim=-1))
    else:
        return quat_unit(torch.cat([w, xyz], dim=-1))

@torch.jit.script
def calc_heading_quat_inv(q: Tensor, w_last: bool = False) -> Tensor:
    # calculate heading rotation from quaternion
    # the heading is the direction on the xy plane
    # q must be normalized
    heading = calc_heading(q, w_last)
    axis = torch.zeros_like(q[..., 0:3])
    axis[..., 2] = 1

    heading_q = quat_from_angle_axis(-heading, axis, w_last)
    return heading_q

@torch.jit.script
def quat_mul(a, b, w_last: bool):
    assert a.shape == b.shape
    shape = a.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 4)

    if w_last:
        x1, y1, z1, w1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
        x2, y2, z2, w2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    else:
        w1, x1, y1, z1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
        w2, x2, y2, z2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    if w_last:
        quat = torch.stack([x, y, z, w], dim=-1).view(shape)
    else:
        quat = torch.stack([w, x, y, z], dim=-1).view(shape)

    return quat

@torch.jit.script
def quat_to_tan_norm(q: Tensor, w_last: bool) -> Tensor:
    # represents a rotation using the tangent and normal vectors
    ref_tan = torch.zeros_like(q[..., 0:3])
    ref_tan[..., 0] = 1
    tan = quat_rotate(q, ref_tan, w_last)
    
    ref_norm = torch.zeros_like(q[..., 0:3])
    ref_norm[..., -1] = 1
    norm = quat_rotate(q, ref_norm, w_last)
    
    norm_tan = torch.cat([tan, norm], dim=len(tan.shape) - 1)
    return norm_tan

class MujocoBipedalEnv():
    def __init__(self, model_path: str):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.step_dt = 0.005  # Time step for simulation
        self.control_dt = 0.02
        self.decimation_factor = int(self.control_dt / self.step_dt)
        self._viewer = None
        self._action_scale = 0.5
        self._joint_names_isaac = ['torsoyaw', 'L_hipyaw', 'R_hipyaw', 'L_hiproll', 'R_hiproll', 'L_hippitch', 'R_hippitch', 'L_knee', 'R_knee', 'L_anklepitch', 'R_anklepitch', 'L_ankleroll', 'R_ankleroll', 'L_toepitch', 'R_toepitch']
        self._joint_names_mujoco = ['torsoyaw', 'R_hipyaw', 'R_hiproll', 'R_hippitch', 'R_knee', 'R_anklepitch', 'R_ankleroll', 'R_toepitch', 'L_hipyaw', 'L_hiproll', 'L_hippitch', 'L_knee', 'L_anklepitch', 'L_ankleroll', 'L_toepitch']
        isaac_idx   = {n: i for i, n in enumerate(self._joint_names_isaac)}
        mujoco_idx  = {n: i for i, n in enumerate(self._joint_names_mujoco)}
        self.mujoco_to_isaac = np.array([mujoco_idx[n] for n in self._joint_names_isaac])
        self.isaac_to_mujoco  = np.array([isaac_idx[n]  for n in self._joint_names_mujoco])
        self._prev_action = np.zeros(len(self._joint_names_isaac), dtype=np.float32)
        
        # This is under the isaaclab convention
        self._default_joint_pos = np.array([ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000, -0.0500, -0.0500,  0.1000,
          0.1000, -0.1000, -0.1000,  0.0000,  0.0000,  0.0200,  0.0200], dtype=np.float32)
        self._stiffness = np.array([200, 150, 150, 150, 150, 200, 200, 200, 200, 20, 20, 20, 20, 20, 20], dtype=np.float32)
        self._damping = np.array([5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4], dtype=np.float32)
        self._effort_limit = np.array([70, 70, 70, 70, 70, 70, 70, 200, 200, 100, 100, 40, 40, 40, 40], dtype=np.float32)
        self._episode_length = 0
        
        self._bodynames = ['torso_link', 'pelvis', 'head', 'L_hipyaw_link', 'R_hipyaw_link', 'L_hiproll_link', 'R_hiproll_link', 'L_hippitch_link', 'R_hippitch_link', 'L_shank_link', 'R_shank_link', 'L_ankle_link', 'R_ankle_link', 'L_foot_link', 'R_foot_link', 'L_toe_link', 'R_toe_link']
        self.body_convert_to_common = torch.tensor([ 0,  2,  1,  4,  6,  8, 10, 12, 14, 16,  3,  5,  7,  9, 11, 13, 15], device='cuda:0')
        self.dof_convert_to_common = torch.tensor([ 0,  2,  4,  6,  8, 10, 12, 14,  1,  3,  5,  7,  9, 11, 13], device='cuda:0')
        self._common_pd_action_scale = torch.tensor([2.1991, 2.1980, 2.1980, 2.1980, 1.8326, 2.1980, 1.0990, 0.7330, 2.1991, 2.1991, 2.1991, 1.8326, 2.1991, 0.7330, 0.7330], device='cuda:0')
        self._common_pd_action_offset = torch.tensor([0.0000, 0.0000, 0.0000, 0.0000, 1.3090, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.3090, 0.0000, 0.0000, 0.0000], device='cuda:0')
        self._common_p_gains = torch.tensor([100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100.], device='cuda:0')
        self._common_d_gains = torch.tensor([10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10.], device='cuda:0')
        self._torque_limits_common = torch.tensor([ 88.,  88.,  88., 139.,  50.,  50.,  88.,  88.,  88., 139.,  50.,  50., 88.,  50.,  50.], device='cuda:0')
        self.dof_convert_to_sim = torch.tensor([ 0,  8,  1,  9,  2, 10,  3, 11,  4, 12,  5, 13,  6, 14,  7], device='cuda:0')
        
    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)  # Ensure the model is in a valid state
        self.data.qpos[7:] = self._default_joint_pos[self.isaac_to_mujoco]
        return self.get_observations()
    
    def step(self, action):
        processed_action = self._process_action(action)
        self._episode_length += 1
        # print(f"Action: {action}")
        for _ in range(self.decimation_factor):
            effort = self._compute_effort(processed_action)
            self.data.ctrl[:] = effort[self.isaac_to_mujoco] 
            mujoco.mj_step(self.model, self.data)
        

        return self.get_observations()
    
    def get_observations(self):

        base_ang_vel = self._get_base_ang_vel()
        projected_gravity = self._get_projected_gravity()
        velocity_commands = self._get_velocity_commands()
        joint_pos = self._get_joint_positions()
        joint_vel = self._get_joint_velocities()
        phase_obs = self.phase_obs()
        bodies_positions, bodies_roations, bodies_velocities, bodies_angular_velocities = self._get_simulator_bodies_state()
        bodies_positions, bodies_roations, bodies_velocities, bodies_angular_velocities = self._convert_to_common(
            bodies_positions, bodies_roations, bodies_velocities, bodies_angular_velocities
        )
        ground_heights = torch.tensor([[0.]], device='cuda:0')
        obs = self.compute_humanoid_observations_max(
                                                    bodies_positions,
                                                    bodies_roations,
                                                    bodies_velocities,
                                                    bodies_angular_velocities,
                                                    ground_heights,
                                                    True,
                                                    True,
                                                    True, 
                                                    )
                                                    
                                                 
        print("phase_obs:", phase_obs)
        obs = np.concatenate([
            base_ang_vel, 
            projected_gravity, 
            velocity_commands, 
            joint_pos, 
            joint_vel, 
            self._prev_action,  # Previous action
            phase_obs
        ])
        
        return obs
    
    def _process_action(self, action):
        self._prev_action = action
        clamp_actions = 1.0

        actions = torch.clamp(action, -clamp_actions, clamp_actions)
        
        return actions
    
    def _compute_effort(self, actions):
        isaacsim_dof_pos = self.data.qpos[7:][self.mujoco_to_isaac]
        isaacsim_dof_vel = self.data.qvel[6:][self.mujoco_to_isaac]
        new_dof_pos = isaacsim_dof_pos[:, self.dof_convert_to_common]
        new_dof_vel = isaacsim_dof_vel[:, self.dof_convert_to_common]
        
        pd_tar = self._common_pd_action_offset + self._common_pd_action_scale * actions
        
        torques =   self._common_p_gains * (pd_tar - new_dof_pos)  - self._common_d_gains * new_dof_vel
        torques = torch.clamp(torques, -self._torque_limits_common, self._torque_limits_common)
        isaaclab_torques = torques[:, self.dof_convert_to_sim]
        return isaaclab_torques
    
    def render(self):
        if self._viewer is None:
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self._viewer.sync()
        
        
    def _get_projected_gravity(self):
        """Returns the projected gravity vector."""
        quat = self.data.qpos[3:7]
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, quat)
        R = R.reshape(3, 3)
        return np.dot(R.T, np.array([0, 0, -1]))
    
    def _get_base_ang_vel(self):
        """Returns the angular velocity of the base."""
        return self.data.qvel[3:6]
    
    def _get_velocity_commands(self):
        return np.array([1, 0.0, 0.0])
    
    def _get_joint_positions(self):
        """Returns the joint positions."""
        return self.data.qpos[7:][self.mujoco_to_isaac] - self._default_joint_pos
    
    def _get_joint_velocities(self):
        """Returns the joint velocities."""
        return self.data.qvel[6:][self.mujoco_to_isaac]
    
    def phase_obs(self):
        cycle = 0.8
        phase = self._episode_length * self.control_dt % cycle / cycle
        sin_phase = np.sin(2 * np.pi * phase)
        cos_phase = np.cos(2 * np.pi * phase)
        return np.array([sin_phase, cos_phase], dtype=np.float32)
    
    def _get_simulator_bodies_state(self):
        bodies_positions  = torch.tensor([self.data.body(b).xpos for b in self._bodynames], device='cuda:0').unsqueeze(0)
        bodies_roations   = torch.tensor([self.data.body(b).xquat for b in self._bodynames], device='cuda:0').unsqueeze(0)
        bodies_velocities = torch.tensor([self.data.body(b).cvel[3:] for b in self._bodynames], device='cuda:0').unsqueeze(0)
        bodies_angular_velocities = torch.tensor([self.data.body(b).cvel[:3] for b in self._bodynames], device='cuda:0').unsqueeze(0)
        return bodies_positions, bodies_roations, bodies_velocities, bodies_angular_velocities

    def _convert_to_common(self, bodies_positions, bodies_roations, bodies_velocities, bodies_angular_velocities):
        new_rigid_body_pos = bodies_positions[:, self.body_convert_to_common]
        rb_rot = wxyz_to_xyzw(bodies_roations)
        new_rigid_body_rot = rb_rot[:, self.body_convert_to_common]
        new_rigid_body_vel = bodies_velocities[:, self.body_convert_to_common]
        new_rigid_body_ang_vel = bodies_angular_velocities[:, self.body_convert_to_common]
        return new_rigid_body_pos, new_rigid_body_rot, new_rigid_body_vel, new_rigid_body_ang_vel
        

    def compute_humanoid_observations_max(
        self,
        body_pos: Tensor,
        body_rot: Tensor,
        body_vel: Tensor,
        body_ang_vel: Tensor,
        ground_height: Tensor,
        local_root_obs: bool,
        root_height_obs: bool,
        w_last: bool,
    ) -> Tensor:
        root_pos = body_pos[:, 0, :]
        root_rot = body_rot[:, 0, :]

        root_h = root_pos[:, 2:3]
        heading_rot = calc_heading_quat_inv(root_rot, w_last)

        if not root_height_obs:
            root_h_obs = torch.zeros_like(root_h)
        else:
            root_h_obs = root_h - ground_height

        heading_rot_expand = heading_rot.unsqueeze(-2)
        heading_rot_expand = heading_rot_expand.repeat((1, body_pos.shape[1], 1))
        flat_heading_rot = heading_rot_expand.reshape(
            heading_rot_expand.shape[0] * heading_rot_expand.shape[1],
            heading_rot_expand.shape[2],
        )

        root_pos_expand = root_pos.unsqueeze(-2)
        local_body_pos = body_pos - root_pos_expand
        flat_local_body_pos = local_body_pos.reshape(
            local_body_pos.shape[0] * local_body_pos.shape[1], local_body_pos.shape[2]
        )
        flat_local_body_pos = quat_rotate(
            flat_heading_rot, flat_local_body_pos, w_last
        )
        local_body_pos = flat_local_body_pos.reshape(
            local_body_pos.shape[0], local_body_pos.shape[1] * local_body_pos.shape[2]
        )
        local_body_pos = local_body_pos[..., 3:]  # remove root pos

        flat_body_rot = body_rot.reshape(
            body_rot.shape[0] * body_rot.shape[1], body_rot.shape[2]
        )
        flat_local_body_rot = quat_mul(flat_heading_rot, flat_body_rot, w_last)
        flat_local_body_rot_obs = quat_to_tan_norm(flat_local_body_rot, w_last)
        local_body_rot_obs = flat_local_body_rot_obs.reshape(
            body_rot.shape[0], body_rot.shape[1] * flat_local_body_rot_obs.shape[1]
        )

        if not local_root_obs:
            root_rot_obs = quat_to_tan_norm(root_rot, w_last)
            local_body_rot_obs[..., 0:6] = root_rot_obs

        flat_body_vel = body_vel.reshape(
            body_vel.shape[0] * body_vel.shape[1], body_vel.shape[2]
        )
        flat_local_body_vel = quat_rotate(flat_heading_rot, flat_body_vel, w_last)
        local_body_vel = flat_local_body_vel.reshape(
            body_vel.shape[0], body_vel.shape[1] * body_vel.shape[2]
        )

        flat_body_ang_vel = body_ang_vel.reshape(
            body_ang_vel.shape[0] * body_ang_vel.shape[1], body_ang_vel.shape[2]
        )
        flat_local_body_ang_vel = quat_rotate(
            flat_heading_rot, flat_body_ang_vel, w_last
        )
        local_body_ang_vel = flat_local_body_ang_vel.reshape(
            body_ang_vel.shape[0], body_ang_vel.shape[1] * body_ang_vel.shape[2]
        )

        obs = torch.cat(
            (
                root_h_obs,
                local_body_pos,
                local_body_rot_obs,
                local_body_vel,
                local_body_ang_vel,
            ),
            dim=-1,
        )
        return obs