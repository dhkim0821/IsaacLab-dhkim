# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin
from __future__ import annotations


from collections import namedtuple
import torch
import numpy as np

from rsl_rl.utils import split_and_pad_trajectories
from rsl_rl.utils.collections import is_namedarraytuple
from rsl_rl.utils.buffer import buffer_from_example, buffer_method, buffer_swap, buffer_expand

class RolloutStorage:
    class Transition:
        def __init__(self):
            self.observations = None
            self.privileged_observations = None
            self.actions = None
            self.privileged_actions = None
            self.rewards = None
            self.dones = None
            self.values = None
            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None
            self.hidden_states = None
            self.rnd_state = None
        
        def clear(self):
            self.__init__()

    MiniBatch = namedtuple("MiniBatch", [
        "obs",
        "critic_obs",
        "actions",
        "values",
        "advantages",
        "returns",
        "old_actions_log_prob",
        "old_mu",
        "old_sigma",
        "hidden_states",
        "masks",
    ])

    def __init__(
        self,
        training_type,
        num_envs,
        num_transitions_per_env,
        obs_shape,
        privileged_obs_shape,
        actions_shape,
        rnd_state_shape=None,
        device="cpu",
    ):
        # store inputs
        self.training_type = training_type
        self.device = device
        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs
        self.obs_shape = obs_shape
        self.privileged_obs_shape = privileged_obs_shape
        self.rnd_state_shape = rnd_state_shape
        self.actions_shape = actions_shape

        # Core
        self.observations = torch.zeros(num_transitions_per_env, num_envs, *obs_shape, device=self.device)
        if privileged_obs_shape is not None:
            self.privileged_observations = torch.zeros(
                num_transitions_per_env, num_envs, *privileged_obs_shape, device=self.device
            )
        else:
            self.privileged_observations = None
        self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()

        # for distillation
        if training_type == "distillation":
            self.privileged_actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)

        # for reinforcement learning
        if training_type == "rl":
            self.values = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
            self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
            self.mu = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
            self.sigma = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
            self.returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
            self.advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)

        # For RND
        if rnd_state_shape is not None:
            self.rnd_state = torch.zeros(num_transitions_per_env, num_envs, *rnd_state_shape, device=self.device)

        # For RNN networks
        self.saved_hidden_states_a = None
        self.saved_hidden_states_c = None

        # counter for the number of transitions stored
        self.step = 0

    def add_transitions(self, transition: Transition):
        # check if the transition is valid
        if self.step >= self.num_transitions_per_env:
            raise OverflowError("Rollout buffer overflow! You should call clear() before adding new transitions.")

        # Core
        self.observations[self.step].copy_(transition.observations)
        if self.privileged_observations is not None:
            self.privileged_observations[self.step].copy_(transition.privileged_observations)
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))

        # for distillation
        if self.training_type == "distillation":
            self.privileged_actions[self.step].copy_(transition.privileged_actions)

        # for reinforcement learning
        if self.training_type == "rl":
            self.values[self.step].copy_(transition.values)
            self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
            self.mu[self.step].copy_(transition.action_mean)
            self.sigma[self.step].copy_(transition.action_sigma)

        # For RND
        if self.rnd_state_shape is not None:
            self.rnd_state[self.step].copy_(transition.rnd_state)

        # For RNN networks
        self._save_hidden_states(transition.hidden_states)

        # increment the counter
        self.step += 1

    def _save_hidden_states(self, hidden_states):
        if hidden_states is None or hidden_states == (None, None):
            return
        # make a tuple out of GRU hidden state sto match the LSTM format
        hid_a = hidden_states[0] if isinstance(hidden_states[0], tuple) else (hidden_states[0],)
        hid_c = hidden_states[1] if isinstance(hidden_states[1], tuple) else (hidden_states[1],)
        # initialize if needed
        if self.saved_hidden_states_a is None:
            self.saved_hidden_states_a = [
                torch.zeros(self.observations.shape[0], *hid_a[i].shape, device=self.device) for i in range(len(hid_a))
            ]
            self.saved_hidden_states_c = [
                torch.zeros(self.observations.shape[0], *hid_c[i].shape, device=self.device) for i in range(len(hid_c))
            ]
        # copy the states
        for i in range(len(hid_a)):
            self.saved_hidden_states_a[i][self.step].copy_(hid_a[i])
            self.saved_hidden_states_c[i][self.step].copy_(hid_c[i])

    def clear(self):
        self.step = 0

    def compute_returns(self, last_values, gamma, lam, normalize_advantage: bool = True):
        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            # if we are at the last step, bootstrap the return value
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]
            # 1 if we are not in a terminal state, 0 otherwise
            next_is_not_terminal = 1.0 - self.dones[step].float()
            # TD error: r_t + gamma * V(s_{t+1}) - V(s_t)
            delta = self.rewards[step] + next_is_not_terminal * gamma * next_values - self.values[step]
            # Advantage: A(s_t, a_t) = delta_t + gamma * lambda * A(s_{t+1}, a_{t+1})
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            # Return: R_t = A(s_t, a_t) + V(s_t)
            self.returns[step] = advantage + self.values[step]

        # Compute the advantages
        self.advantages = self.returns - self.values
        # Normalize the advantages if flag is set
        # This is to prevent double normalization (i.e. if per minibatch normalization is used)
        if normalize_advantage:
            self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    # for distillation
    def generator(self):
        if self.training_type != "distillation":
            raise ValueError("This function is only available for distillation training.")

        for i in range(self.num_transitions_per_env):
            if self.privileged_observations is not None:
                privileged_observations = self.privileged_observations[i]
            else:
                privileged_observations = self.observations[i]
            yield self.observations[i], privileged_observations, self.actions[i], self.privileged_actions[
                i
            ], self.dones[i]

    def get_statistics(self):
        done = self.dones
        done[-1] = 1
        flat_dones = done.permute(1, 0, 2).reshape(-1, 1)
        done_indices = torch.cat((flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero(as_tuple=False)[:, 0]))
        trajectory_lengths = (done_indices[1:] - done_indices[:-1])
        return trajectory_lengths.float().mean(), self.rewards.mean()

    # for reinforcement learning with feedforward networks
    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        if self.training_type != "rl":
            raise ValueError("This function is only available for reinforcement learning training.")
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)

        # Core
        observations = self.observations.flatten(0, 1)
        if self.privileged_observations is not None:
            privileged_observations = self.privileged_observations.flatten(0, 1)
        else:
            privileged_observations = observations

        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)

        # For PPO
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)

        # For RND
        if self.rnd_state_shape is not None:
            rnd_state = self.rnd_state.flatten(0, 1)

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):
                # Select the indices for the mini-batch
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]

                # Create the mini-batch
                # -- Core
                obs_batch = observations[batch_idx]
                privileged_observations_batch = privileged_observations[batch_idx]
                actions_batch = actions[batch_idx]

                # -- For PPO
                target_values_batch = values[batch_idx]
                returns_batch = returns[batch_idx]
                old_actions_log_prob_batch = old_actions_log_prob[batch_idx]
                advantages_batch = advantages[batch_idx]
                old_mu_batch = old_mu[batch_idx]
                old_sigma_batch = old_sigma[batch_idx]

                # -- For RND
                if self.rnd_state_shape is not None:
                    rnd_state_batch = rnd_state[batch_idx]
                else:
                    rnd_state_batch = None

                # yield the mini-batch
                yield obs_batch, privileged_observations_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, (
                    None,
                    None,
                ), None, rnd_state_batch

    # for reinfrocement learning with recurrent networks
    def recurrent_mini_batch_generator(self, num_mini_batches, num_epochs=8):
        if self.training_type != "rl":
            raise ValueError("This function is only available for reinforcement learning training.")
        padded_obs_trajectories, trajectory_masks = split_and_pad_trajectories(self.observations, self.dones)
        if self.privileged_observations is not None:
            padded_privileged_obs_trajectories, _ = split_and_pad_trajectories(self.privileged_observations, self.dones)
        else:
            padded_privileged_obs_trajectories = padded_obs_trajectories

        if self.rnd_state_shape is not None:
            padded_rnd_state_trajectories, _ = split_and_pad_trajectories(self.rnd_state, self.dones)
        else:
            padded_rnd_state_trajectories = None

        mini_batch_size = self.num_envs // num_mini_batches
        for ep in range(num_epochs):
            first_traj = 0
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                stop = (i + 1) * mini_batch_size

                dones = self.dones.squeeze(-1)
                last_was_done = torch.zeros_like(dones, dtype=torch.bool)
                last_was_done[1:] = dones[:-1]
                last_was_done[0] = True
                trajectories_batch_size = torch.sum(last_was_done[:, start:stop])
                last_traj = first_traj + trajectories_batch_size

                masks_batch = trajectory_masks[:, first_traj:last_traj]
                obs_batch = padded_obs_trajectories[:, first_traj:last_traj]
                privileged_obs_batch = padded_privileged_obs_trajectories[:, first_traj:last_traj]

                if padded_rnd_state_trajectories is not None:
                    rnd_state_batch = padded_rnd_state_trajectories[:, first_traj:last_traj]
                else:
                    rnd_state_batch = None

                actions_batch = self.actions[:, start:stop]
                old_mu_batch = self.mu[:, start:stop]
                old_sigma_batch = self.sigma[:, start:stop]
                returns_batch = self.returns[:, start:stop]
                advantages_batch = self.advantages[:, start:stop]
                values_batch = self.values[:, start:stop]
                old_actions_log_prob_batch = self.actions_log_prob[:, start:stop]

                # reshape to [num_envs, time, num layers, hidden dim] (original shape: [time, num_layers, num_envs, hidden_dim])
                # then take only time steps after dones (flattens num envs and time dimensions),
                # take a batch of trajectories and finally reshape back to [num_layers, batch, hidden_dim]
                last_was_done = last_was_done.permute(1, 0)
                hid_a_batch = [
                    saved_hidden_states.permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj]
                    .transpose(1, 0)
                    .contiguous()
                    for saved_hidden_states in self.saved_hidden_states_a
                ]
                hid_c_batch = [
                    saved_hidden_states.permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj]
                    .transpose(1, 0)
                    .contiguous()
                    for saved_hidden_states in self.saved_hidden_states_c
                ]
                # remove the tuple for GRU
                hid_a_batch = hid_a_batch[0] if len(hid_a_batch) == 1 else hid_a_batch
                hid_c_batch = hid_c_batch[0] if len(hid_c_batch) == 1 else hid_c_batch

                yield obs_batch, privileged_obs_batch, actions_batch, values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, (
                    hid_a_batch,
                    hid_c_batch,
                ), masks_batch, rnd_state_batch

                first_traj = last_traj


    def get_minibatch_from_indices(self, T_slice, B_slice, padded_B_slice= None, prev_done_mask= None):
        """ Extract minibatch based on selected indices/slice.
        An independent method allows override by subclasses.
        Args:
            - padded_B_slice: For recurrent trajectories, the observations are already expanded and padded with zeros.
            - prev_done_mask: For recurrent trajectories, 
        Outputs:
            - MiniBatch:
                only batch dimension if not padded_B_slice (non-recurrent case)
                with time, batch dimension if padded_B_slice (recurrent case)
        """
        if padded_B_slice is None:
            obs_batch = self.observations[T_slice, B_slice]
            critic_obs_batch = obs_batch if self.privileged_observations is None else self.privileged_observations[T_slice, B_slice]
            hid_batch = None
            obs_mask_batch = None
        else:
            obs_batch = self._padded_obs_trajectories[T_slice, padded_B_slice]
            critic_obs_batch = obs_batch if self.privileged_observations is None else self._padded_critic_obs_trajectories[T_slice, padded_B_slice]
            
            # reshape to [num_envs, time, num layers, hidden dim] (original shape: [time, num_layers, num_envs, hidden_dim])
            # then take only time steps after dones (flattens num envs and time dimensions),
            # take a batch of trajectories and finally reshape back to [num_layers, batch, hidden_dim]
            prev_done_mask = prev_done_mask.permute(1, 0) # (T, B) -> (B, T)
            hid_batch = buffer_method(
                buffer_method(
                    buffer_method(self.saved_hidden_states, "permute", 2, 0, 1, 3)[prev_done_mask][padded_B_slice],
                    "transpose", 1, 0
                ),
                "contiguous",
            )
            obs_mask_batch = self._trajectory_masks[T_slice, padded_B_slice]
 
        action_batch = self.actions[T_slice, B_slice]
        target_value_batch = self.values[T_slice, B_slice]
        return_batch = self.returns[T_slice, B_slice]
        old_action_log_prob_batch = self.actions_log_prob[T_slice, B_slice]
        advantage_batch = self.advantages[T_slice, B_slice]
        old_mu_batch = self.mu[T_slice, B_slice]
        old_sigma_batch = self.sigma[T_slice, B_slice]

        if padded_B_slice is None:
            # flatten the trajectory sample if not recurrent
            obs_batch = obs_batch.flatten(0, 1)
            critic_obs_batch = critic_obs_batch.flatten(0, 1)
            
            action_batch = action_batch.flatten(0, 1)
            target_value_batch = target_value_batch.flatten(0, 1)
            return_batch = return_batch.flatten(0, 1)
            old_action_log_prob_batch = old_action_log_prob_batch.flatten(0, 1)
            advantage_batch = advantage_batch.flatten(0, 1)
            old_mu_batch = old_mu_batch.flatten(0, 1)
            old_sigma_batch = old_sigma_batch.flatten(0, 1)

        return RolloutStorage.MiniBatch(
            obs_batch, critic_obs_batch,
            action_batch,
            target_value_batch, advantage_batch, return_batch,
            old_action_log_prob_batch, old_mu_batch, old_sigma_batch,
            hid_batch, obs_mask_batch,
        )

class QueueRolloutStorage(RolloutStorage):
    def __init__(self,
            num_envs,
            num_transitions_per_env,
            *args,
            buffer_dilation_ratio= 1.0,
            **kwargs,
        ):
        """ This rollout storage allows the buffer to be larger than the rollout length.
        NOTE: num_transitions_per_env is no longer a constant representing the buffer temporal length.
        
        Args:
            size_dilation_ratio: float, for the size of buffer bigger than num_transitions_per_env
        """
        self.num_timesteps_each_rollout = num_transitions_per_env
        self.buffer_dilation_ratio = buffer_dilation_ratio
        self.buffer_full = False
        super().__init__(
            num_envs,
            num_transitions_per_env,
            *args,
            **kwargs,
        )

    @torch.no_grad()
    def expand_buffer_once(self):
        """ Expand the buffer size in this way so that the mini_batch_generator will not output
        the buffer where no data has been stored
        """
        expand_size = int(self.buffer_dilation_ratio * self.num_timesteps_each_rollout - self.num_transitions_per_env)
        expand_size = min(expand_size, self.num_timesteps_each_rollout)
        self.num_transitions_per_env += expand_size

        # expand the buffer by concatenating
        # Core
        self.observations = torch.cat([
            self.observations,
            torch.zeros(expand_size, self.num_envs, *self.obs_shape, device=self.device),
        ], dim= 0).contiguous()
        if self.privileged_obs_shape[0] is not None:
            self.privileged_observations = torch.cat([
                self.privileged_observations,
                torch.zeros(expand_size, self.num_envs, *self.privileged_obs_shape, device=self.device),
            ], dim= 0).contiguous()
        self.rewards = torch.cat([
            self.rewards,
            torch.zeros(expand_size, self.num_envs, 1, device=self.device),
        ], dim= 0).contiguous()
        self.actions = torch.cat([
            self.actions,
            torch.zeros(expand_size, self.num_envs, *self.actions_shape, device=self.device),
        ], dim= 0).contiguous()
        self.dones = torch.cat([
            self.dones,
            torch.zeros(expand_size, self.num_envs, 1, device=self.device).byte(),
        ], dim= 0).contiguous()

        # For PPO
        self.actions_log_prob = torch.cat([
            self.actions_log_prob,
            torch.zeros(expand_size, self.num_envs, 1, device=self.device),
        ], dim= 0).contiguous()
        self.values = torch.cat([
            self.values,
            torch.zeros(expand_size, self.num_envs, 1, device=self.device),
        ], dim= 0).contiguous()
        self.returns = torch.cat([
            self.returns,
            torch.zeros(expand_size, self.num_envs, 1, device=self.device),
        ], dim= 0).contiguous()
        self.advantages = torch.cat([
            self.advantages,
            torch.zeros(expand_size, self.num_envs, 1, device=self.device),
        ], dim= 0).contiguous()
        self.mu = torch.cat([
            self.mu,
            torch.zeros(expand_size, self.num_envs, *self.actions_shape, device=self.device),
        ], dim= 0).contiguous()
        self.sigma = torch.cat([
            self.sigma,
            torch.zeros(expand_size, self.num_envs, *self.actions_shape, device=self.device),
        ], dim= 0).contiguous()

        # For hidden_states
        if not self.saved_hidden_states is None:
            self.saved_hidden_states = buffer_expand(
                self.saved_hidden_states,
                expand_size,
                dim= 0,
                contiguous= True,
            )

        return expand_size

    def add_transitions(self, transition: RolloutStorage.Transition):
        return_ = super().add_transitions(transition)
        if self.step >= self.num_transitions_per_env:
            self.buffer_full = self.num_transitions_per_env >= int(self.buffer_dilation_ratio * self.num_timesteps_each_rollout)
            if self.buffer_full:
                self.step = self.step % self.num_transitions_per_env
        return return_

    def clear(self):
        """ Not return the self.step to 0 but check whether it needs to expaned the buffer.
        """
        if self.step >= self.num_transitions_per_env and not self.buffer_full:
            _ = self.expand_buffer_once() # Then self.num_transitions_per_env is updated
            print("QueueRolloutStorage: rollout storage expanded.")

    @torch.no_grad()
    def swap_from_cursor(self, buffer):
        """ This returns a new buffer (not necessarily new memory) """
        if self.step == buffer.shape[0] or self.step == 0:
            return buffer
        return torch.cat([
            buffer[self.step:],
            buffer[:self.step],
        ], dim= 0).detach().contiguous()

    def untie_buffer_loop(self):
        self.observations = self.swap_from_cursor(self.observations)
        if self.privileged_observations is not None: self.privileged_observations = self.swap_from_cursor(self.privileged_observations)
        self.actions = self.swap_from_cursor(self.actions)
        self.rewards = self.swap_from_cursor(self.rewards)
        self.dones = self.swap_from_cursor(self.dones)
        self.values = self.swap_from_cursor(self.values)
        self.actions_log_prob = self.swap_from_cursor(self.actions_log_prob)
        self.mu = self.swap_from_cursor(self.mu)
        self.sigma = self.swap_from_cursor(self.sigma)
        if not self.saved_hidden_states is None:
            with torch.no_grad():
                self.saved_hidden_states = buffer_swap(self.saved_hidden_states, self.step, contiguous= True)
        self.step = 0

    def reccurent_mini_batch_generator(self, num_mini_batches, num_epochs=8):
        """ Re-align all the buffer to make the transitions continuous before the sampling.
        [5,6,7,8,9,0,1,2,3,4] -> [0,1,2,3,4,5,6,7,8,9] where 9 is where the latest transition stored.
        """
        if self.buffer_dilation_ratio > 1.0 and self.buffer_full:
            self.untie_buffer_loop()
        return super().reccurent_mini_batch_generator(num_mini_batches, num_epochs)

class ActionLabelRollout(QueueRolloutStorage):
    class Transition(QueueRolloutStorage.Transition):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.action_labels = None
    
    MiniBatch = namedtuple("MiniBatch", [
        *RolloutStorage.MiniBatch._fields,
        "action_labels",
    ])

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_labels = torch.zeros_like(self.actions)

    def expand_buffer_once(self):
        expand_size = super().expand_buffer_once()
        self.action_labels = torch.cat([
            self.action_labels,
            torch.zeros(expand_size, self.num_envs, *self.actions_shape, device=self.device),
        ], dim= 0).contiguous()
        return expand_size

    def add_transitions(self, transition: Transition):
        self.action_labels[self.step] = transition.action_labels
        return super().add_transitions(transition)

    def untie_buffer_loop(self):
        self.action_labels = self.swap_from_cursor(self.action_labels)
        return super().untie_buffer_loop()
    
    def get_minibatch_from_indices(self, T_slice, B_slice, padded_B_slice=None, prev_done_mask=None):
        minibatch = super().get_minibatch_from_indices(T_slice, B_slice, padded_B_slice, prev_done_mask)
        action_label_batch = self.action_labels[T_slice, B_slice]

        if padded_B_slice is None:
            action_label_batch = action_label_batch.flatten(0, 1)
        
        return ActionLabelRollout.MiniBatch(*minibatch, action_label_batch)
            
class SarsaRolloutStorage(RolloutStorage):
    """ The rollout storage for SARSA algorithm and those who need the next state """
    class Transition(RolloutStorage.Transition):
        def __init__(self):
            super().__init__()
            self.next_observations = None
            self.next_privileged_observations = None

    MiniBatch = namedtuple("MiniBatch", [
        *RolloutStorage.MiniBatch._fields,
        "next_obs",
        "next_critic_obs",
    ])

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Use a longer buffer to store the next observation and perform the safety checking
        self.all_observations = torch.cat([
            self.observations,
            torch.zeros_like(self.observations[:1]),
        ]).contiguous()
        self.observations = self.all_observations[:-1]
        self.next_observations = self.all_observations[1:]
        if not self.privileged_observations is None:
            self.all_privileged_observations = torch.cat([
                self.privileged_observations,
                torch.zeros_like(self.privileged_observations[:1]),
            ]).contiguous()
            self.privileged_observations = self.all_privileged_observations[:-1]
            self.next_privileged_observations = self.all_privileged_observations[1:]

    def add_transitions(self, transition: Transition):
        if False:
            # For the running efficiency, will not check obs[step] == next_obs[step-1]
            assert (transition.observations == self.next_observations[self.step-1]).all(), \
            "It is the user's responsibility to make sure that that the next_obs[step-1] == obs[step] (error in observation) "
            assert (transition.privileged_observations == self.next_privileged_observations[self.step-1]).all(), \
            "It is the user's responsibility to make sure that that the next_obs[step-1] == obs[step] (error in privileged observation) "
        if self.step == (self.num_transitions_per_env - 1):
            # For the running efficiency, only copy next_observation at the end of the rollout.
            # Because next_obs and obs shares the same memory.
            # Also assuming each rollout (by the runner) fills the rollout storage.
            self.next_observations[self.step].copy_(transition.next_observations)
            self.next_privileged_observations[self.step].copy_(transition.next_privileged_observations)
        return super().add_transitions(transition)
    
    def reccurent_mini_batch_generator(self, num_mini_batches, num_epochs=8):
        self._padded_next_obs_trajectories, _ = split_and_pad_trajectories(
           self.next_observations,
           self.dones,
        )
        if not self.privileged_observations is None:
            self._padded_next_critic_obs_trajectories, _ = split_and_pad_trajectories(
                self.next_privileged_observations,
                self.dones,
            )
        return super().reccurent_mini_batch_generator(num_mini_batches, num_epochs)
    
    def get_minibatch_from_indices(self, T_slice, B_slice, padded_B_slice=None, prev_done_mask=None):
        minibatch = super().get_minibatch_from_indices(T_slice, B_slice, padded_B_slice, prev_done_mask)

        if padded_B_slice is None:
            next_obs_batch = self.next_observations[T_slice, B_slice]
            next_critic_obs_batch = next_obs_batch if self.privileged_observations is None else self.next_privileged_observations[T_slice, B_slice]
        else:
            next_obs_batch = self._padded_next_obs_trajectories[T_slice, B_slice]
            next_critic_obs_batch = next_obs_batch if self.privileged_observations is None else self._padded_next_critic_obs_trajectories[T_slice, B_slice]
        
        return SarsaRolloutStorage.MiniBatch(
            *minibatch,
            next_obs_batch,
            next_critic_obs_batch,
        )
