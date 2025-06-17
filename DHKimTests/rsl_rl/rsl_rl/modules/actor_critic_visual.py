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

# Import Conv2dHeadModel and get_activation from codebase
# Here's a basic placeholder
class Conv2dHeadModel(nn.Module):
    def __init__(
        self,
        image_shape,
        output_size,
        channels=[64, 64],
        kernel_sizes=[3, 3],
        strides=[1, 1],
        hidden_sizes=[256],
    ):
        super().__init__()
        c, h, w = image_shape
        conv_layers = []
        in_c = c
        for out_c, k, s in zip(channels, kernel_sizes, strides):
            conv_layers.append(nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=k // 2))
            conv_layers.append(nn.ReLU())
            in_c = out_c
        self.conv = nn.Sequential(*conv_layers)

        # Compute the size after conv layers
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            conv_out = self.conv(dummy)
            flat_size = conv_out.view(1, -1).shape[1]

        mlp = []
        last_size = flat_size
        for hidden in hidden_sizes:
            mlp.append(nn.Linear(last_size, hidden))
            mlp.append(nn.ReLU())
            last_size = hidden
        mlp.append(nn.Linear(last_size, output_size))
        self.mlp = nn.Sequential(*mlp)

    def forward(self, x):
        # x: (N, C, H, W)
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        h = self.mlp(h)
        return h

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

class ActorCriticVisual(nn.Module):
    is_recurrent = False
    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation='elu',
        init_noise_std=1.0,
        mu_activation=None,
        obs_segments=None,
        privileged_obs_segments=None,  # Not used
        visual_latent_size=256,
        # You may add more kwargs for visual features if desired
        visual_channels=[64, 64],
        visual_kernel_sizes=[3, 3],
        visual_strides=[1, 1],
        visual_hidden_sizes=[256],
        **kwargs
    ):
        if kwargs:
            print("ActorCriticVisual.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(ActorCriticVisual, self).__init__()

        # The observation is assumed to have the last 3072 entries as depth image (1, 64, 48)
        self.depth_image_size = (1, 48, 64)  # (channels, height, width)
        self.depth_image_flatten = 3072      # 1*64*48
        self.visual_latent_size = visual_latent_size

        activation_fn = get_activation(activation)

        # Set up visual encoder
        self.visual_encoder = Conv2dHeadModel(
            image_shape=self.depth_image_size,
            output_size=self.visual_latent_size,
            channels=visual_channels,
            kernel_sizes=visual_kernel_sizes,
            strides=visual_strides,
            hidden_sizes=visual_hidden_sizes,
        )

        # Determine MLP input sizes (subtract image obs, add latent)
        mlp_input_dim_a = num_actor_obs - self.depth_image_flatten + self.visual_latent_size
        mlp_input_dim_c = num_critic_obs - self.depth_image_flatten + self.visual_latent_size

        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation_fn)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
                if mu_activation:
                    actor_layers.append(get_activation(mu_activation))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation_fn)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation_fn)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation_fn)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    def _embed_visual_latent(self, observations):
        """
        observations: (..., obs_dim)
        Returns same shape but with image replaced by visual latent
        """
        obs = observations
        leading_dims = obs.shape[:-1]
        obs_flat = obs.view(-1, obs.shape[-1])

        # Split: non-visual | visual
        non_visual = obs_flat[:, :-self.depth_image_flatten]
        visual = obs_flat[:, -self.depth_image_flatten:]
        # Reshape visual to (N, C, H, W)
        visual = visual.reshape(-1, *self.depth_image_size)
        visual_latent = self.visual_encoder(visual)
        # Concatenate
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
        obs_latent = self._embed_visual_latent(observations)
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
    def clip_std(self, min=None, max=None):
        self.std.copy_(self.std.clip(min=min, max=max))