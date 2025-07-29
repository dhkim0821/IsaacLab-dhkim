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

import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn
from rsl_rl.modules.conv2d import Conv2dHeadModel  # Added import for visual encoder

class ActorCriticVisual(nn.Module):
    is_recurrent = False
    def __init__(self,  num_actor_obs,
                        num_critic_obs,
                        num_actions,
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        activation='elu',
                        init_noise_std=1.0,
                        mu_activation= None,
                        obs_segments= None,
                        privileged_obs_segments= None,
                        visual_latent_size=256,
                        **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(ActorCriticVisual, self).__init__()

        if privileged_obs_segments is None:
            privileged_obs_segments = obs_segments
        self.obs_segments = obs_segments
        self.privileged_obs_segments = privileged_obs_segments

        activation = get_activation(activation)

        # Visual encoder
        self.depth_image_size = (1, 48, 64)
        self.depth_image_flatten = 3072
        self.visual_latent_size = visual_latent_size
        visual_kwargs= dict()
        self.visual_kwargs = dict(
            channels= [64, 64],
            kernel_sizes= [3, 3],
            strides= [1, 1],
            hidden_sizes= [256],
        ); self.visual_kwargs.update(visual_kwargs)

        self.visual_encoder = Conv2dHeadModel(
            image_shape= self.depth_image_size,
            output_size= self.visual_latent_size,
            **self.visual_kwargs,
        )

        mlp_input_dim_a = num_actor_obs - self.depth_image_flatten + self.visual_latent_size
        mlp_input_dim_c = num_critic_obs - self.depth_image_flatten + self.visual_latent_size

        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
                if mu_activation:
                    actor_layers.append(get_activation(mu_activation))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        Normal.set_default_validate_args = False

    @staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    def _embed_visual_latent(self, observations):
        obs = observations
        leading_dims = obs.shape[:-1]
        obs_flat = obs.view(-1, obs.shape[-1])

        non_visual = obs_flat[:, :-self.depth_image_flatten]
        visual = obs_flat[:, -self.depth_image_flatten:]
        visual = visual.reshape(-1, *self.depth_image_size)
        visual_latent = self.visual_encoder(visual)
        obs_latent = torch.cat([non_visual, visual_latent], dim=-1)
        obs_latent = obs_latent.view(*leading_dims, -1)
        return obs_latent

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        obs_latent = self._embed_visual_latent(observations)
        mean = self.actor(obs_latent)
        self.distribution = Normal(mean, mean * 0. + self.std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        obs_latent = self._embed_visual_latent(observations)
        actions_mean = self.actor(obs_latent)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        obs_latent = self._embed_visual_latent(critic_observations)
        value = self.critic(obs_latent)
        return value

    @torch.no_grad()
    def clip_std(self, min= None, max= None):
        self.std.copy_(self.std.clip(min= min, max= max))

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
