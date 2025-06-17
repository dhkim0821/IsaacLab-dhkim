# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal
from rsl_rl.modules.conv2d import Conv2dHeadModel

# If there is utility use it; otherwise use get_activation below.
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

# class Conv2dHeadModel(nn.Module):
#     def __init__(
#         self,
#         image_shape,
#         output_size,
#         channels=[64, 64],
#         kernel_sizes=[3, 3],
#         strides=[1, 1],
#         hidden_sizes=[256],
#     ):
#         super().__init__()
#         c, h, w = image_shape
#         conv_layers = []
#         in_c = c
#         for out_c, k, s in zip(channels, kernel_sizes, strides):
#             conv_layers.append(nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=k // 2))
#             conv_layers.append(nn.ReLU())
#             in_c = out_c
#         self.conv = nn.Sequential(*conv_layers)

#         # Compute the size after conv layers
#         with torch.no_grad():
#             dummy = torch.zeros(1, c, h, w)
#             conv_out = self.conv(dummy)
#             flat_size = conv_out.view(1, -1).shape[1]

#         mlp = []
#         last_size = flat_size
#         for hidden in hidden_sizes:
#             mlp.append(nn.Linear(last_size, hidden))
#             mlp.append(nn.ReLU())
#             last_size = hidden
#         mlp.append(nn.Linear(last_size, output_size))
#         self.mlp = nn.Sequential(*mlp)

    def forward(self, x):
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        h = self.mlp(h)
        return h

class StudentTeacherVisual(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_student_obs,
        num_teacher_obs,  # keep for API compatibility, but we'll use 44+191 for teacher
        num_actions,
        student_hidden_dims=[256, 256, 256],
        teacher_hidden_dims=[512, 256, 128],  # match your PPO teacher!
        activation="elu",
        init_noise_std=0.1,
        # Visual encoder options for student
        visual_latent_size=256,
        visual_kwargs= dict(),
        #visual_channels=[64, 64],
        #visual_kernel_sizes=[3, 3],
        #visual_strides=[1, 1],
        #visual_hidden_sizes=[256],
        **kwargs,
    ):
        if kwargs:
            print(
                "StudentTeacherVisual.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        activation_fn = get_activation(activation)
        self.loaded_teacher = False  # indicates if teacher has been loaded

        # Assume obs: [proprio (44)], [depth (3072)], [height (191)]
        self.proprio_size = 44
        self.depth_image_flatten = 3072
        self.depth_image_size = (1, 48, 64)
        self.height_flatten = 191
        self.visual_latent_size = visual_latent_size

        self.visual_kwargs = dict(
            channels= [64, 64],
            kernel_sizes= [3, 3],
            strides= [1, 1],
            hidden_sizes= [256],
        ); self.visual_kwargs.update(visual_kwargs)

        # Visual encoder for student (depth image)
        # self.visual_encoder = Conv2dHeadModel(
        #     image_shape=self.depth_image_size,
        #     output_size=self.visual_latent_size,
        #     channels=visual_channels,
        #     kernel_sizes=visual_kernel_sizes,
        #     strides=visual_strides,
        #     hidden_sizes=visual_hidden_sizes,
        # )

        self.visual_encoder = Conv2dHeadModel(
            image_shape= self.depth_image_size,
            output_size= self.visual_latent_size,
            **self.visual_kwargs,
        )


        # Student MLP input: (proprio + height) + visual latent
        mlp_input_dim_s = self.proprio_size + self.height_flatten + self.visual_latent_size

        self.teacher_obs_dim = self.proprio_size + self.height_flatten  # 44+191=235
        mlp_input_dim_t = self.teacher_obs_dim

        # student
        if isinstance(student_hidden_dims, int):
            student_hidden_dims = [student_hidden_dims]
        student_layers = []
        student_layers.append(nn.Linear(mlp_input_dim_s, student_hidden_dims[0]))
        student_layers.append(activation_fn)
        for layer_index in range(len(student_hidden_dims)):
            if layer_index == len(student_hidden_dims) - 1:
                student_layers.append(nn.Linear(student_hidden_dims[layer_index], num_actions))
            else:
                student_layers.append(nn.Linear(student_hidden_dims[layer_index], student_hidden_dims[layer_index + 1]))
                student_layers.append(activation_fn)
        self.student = nn.Sequential(*student_layers)

        # teacher
        if isinstance(teacher_hidden_dims, int):
            teacher_hidden_dims = [teacher_hidden_dims]
        teacher_layers = []
        teacher_layers.append(nn.Linear(mlp_input_dim_t, teacher_hidden_dims[0]))
        teacher_layers.append(activation_fn)
        for layer_index in range(len(teacher_hidden_dims)):
            if layer_index == len(teacher_hidden_dims) - 1:
                teacher_layers.append(nn.Linear(teacher_hidden_dims[layer_index], num_actions))
            else:
                teacher_layers.append(nn.Linear(teacher_hidden_dims[layer_index], teacher_hidden_dims[layer_index + 1]))
                teacher_layers.append(activation_fn)
        self.teacher = nn.Sequential(*teacher_layers)
        self.teacher.eval()

        print(f"Student MLP: {self.student}")
        print(f"Teacher MLP: {self.teacher}")

        # action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

    def reset(self, dones=None, hidden_states=None):
        pass

    def forward(self):
        raise NotImplementedError

    def _split_obs(self, obs):
        # obs shape: (..., 44+3072+191)
        proprio = obs[..., :self.proprio_size]
        depth = obs[..., self.proprio_size:self.proprio_size + self.depth_image_flatten]
        height = obs[..., self.proprio_size + self.depth_image_flatten:self.proprio_size + self.depth_image_flatten + self.height_flatten]
        return proprio, depth, height

    def _embed_student_latent(self, obs):
        # obs: (..., 3307)
        proprio, depth, height = self._split_obs(obs)
        depth = depth.view(-1, *self.depth_image_size)
        visual_latent = self.visual_encoder(depth)
        # Always use [proprio, height, visual_latent] for student
        if proprio.dim() == 1:
            proprio = proprio.unsqueeze(0)
            height = height.unsqueeze(0)
        obs_latent = torch.cat([proprio, height, visual_latent], dim=-1)
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

    def update_distribution(self, obs):
        obs_latent = self._embed_student_latent(obs)
        mean = self.student(obs_latent)
        std = self.std.expand_as(mean)
        self.distribution = Normal(mean, std)

    def act(self, obs):
        self.update_distribution(obs)
        return self.distribution.sample()

    def act_inference(self, obs):
        obs_latent = self._embed_student_latent(obs)
        actions_mean = self.student(obs_latent)
        return actions_mean

    def evaluate(self, obs):
        """Evaluate teacher: always use first 44 and last 191 dims (proprio+height)"""
        with torch.no_grad():
            obs_first44 = obs[..., :self.proprio_size]
            obs_last191 = obs[..., -self.height_flatten:]
            teacher_input = torch.cat([obs_first44, obs_last191], dim=-1)
            actions = self.teacher(teacher_input)
        return actions

    def load_state_dict(self, state_dict, strict=True):
        """Load the parameters of the student and teacher networks.

        Args:
            state_dict (dict): State dictionary of the model.
            strict (bool): Whether to strictly enforce that the keys in state_dict match the keys returned by this
                           module's state_dict() function.

        Returns:
            bool: Whether this training resumes a previous training. This flag is used by the `load()` function of
                  `OnPolicyRunner` to determine how to load further parameters.
        """

        # check if state_dict contains teacher and student or just teacher parameters
        if any("actor" in key for key in state_dict.keys()):  # loading parameters from rl training
            # rename keys to match teacher and remove critic parameters
            teacher_state_dict = {}
            for key, value in state_dict.items():
                if "actor." in key:
                    teacher_state_dict[key.replace("actor.", "")] = value
            load_result = self.teacher.load_state_dict(teacher_state_dict, strict=strict)
            self.loaded_teacher = True
            self.teacher.eval()
            return False
        elif any("student" in key for key in state_dict.keys()):  # loading parameters from distillation training
            super().load_state_dict(state_dict, strict=strict)
            self.loaded_teacher = True
            self.teacher.eval()
            return True
        else:
            raise ValueError("state_dict does not contain student or teacher parameters")

    def get_hidden_states(self):
        return None

    def detach_hidden_states(self, dones=None):
        pass