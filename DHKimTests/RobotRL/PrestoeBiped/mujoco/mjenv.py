import mujoco
import mujoco.viewer
import numpy as np


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
        self._total_effort = 0.0
        self._init_2dpose =  None
        
    
    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[7:] = self._default_joint_pos[self.isaac_to_mujoco]
        self._init_2dpose = self.data.qpos[0:2].copy()
        return self.get_observations()
    
    def step(self, action):
        processed_action = self._process_action(action)
        self._episode_length += 1
        # print(f"Action: {action}")
        for _ in range(self.decimation_factor):
            effort = self._compute_effort(processed_action)
            self.data.ctrl[:] = effort[self.isaac_to_mujoco] 
            self._total_effort += np.sum(np.square(effort))
            mujoco.mj_step(self.model, self.data)
        

        return self.get_observations()
    
    def get_observations(self):

        base_ang_vel = self._get_base_ang_vel()
        projected_gravity = self._get_projected_gravity()
        velocity_commands = self._get_velocity_commands()
        joint_pos = self._get_joint_positions()
        joint_vel = self._get_joint_velocities()
        phase_obs = self.phase_obs()
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
        
        distance_traveled = np.linalg.norm(self.data.qpos[0:2] - self._init_2dpose)
        tau_squareToD = self._total_effort  / distance_traveled if distance_traveled > 0 else 0
        
        return obs, distance_traveled, tau_squareToD
    
    def _process_action(self, action):
        self._prev_action = action
        processed_action = action * self._action_scale + self._default_joint_pos
        return processed_action
    
    def _compute_effort(self, action):
        error_pos = action - self.data.qpos[7:][self.mujoco_to_isaac]
        error_vel = -self.data.qvel[6:][self.mujoco_to_isaac]
        effort = self._stiffness * error_pos + self._damping * error_vel
        effort = np.clip(effort, -self._effort_limit, self._effort_limit)
        return effort
    
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
        return np.array([1.2, 0.0, 0.0])
    
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