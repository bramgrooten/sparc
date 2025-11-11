import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.constants import LOG_STD_MIN, LOG_STD_MAX


class ExpertPolicy(nn.Module):
    def __init__(self, env, obs_shape=None, context_shape=None, action_shape=None):
        super().__init__()
        if obs_shape is not None:
            obs_dim = np.prod(obs_shape)
            context_dim = np.prod(context_shape)
            action_dim = np.prod(action_shape)
        else:
            obs_dim = np.array(env.single_observation_space.shape).prod()
            context_dim = 2  # for wind envs: context is speed in x and z directions
            action_dim = np.prod(env.single_action_space.shape)
        # observation encoder
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)

        # context encoder
        self.encoder = ContextEncoder(context_dim)

        # decision layers
        self.fc3 = nn.Linear(256 + 32, 256)
        self.fc_mean = nn.Linear(256, action_dim)
        self.fc_logstd = nn.Linear(256, action_dim)
        # action rescaling, to the environment's bounds
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (env.single_action_space.high - env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (env.single_action_space.high + env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, obs, context):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))

        z = self.encoder(context)

        # concatenate encodings of observation and context
        x = torch.cat([x, z], dim=1)
        x = F.relu(self.fc3(x))

        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mean, log_std

    def get_action(self, obs, context):
        mean, log_std = self(obs, context)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # enforcing action bounds
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean_action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean_action


class AdapterPolicy(nn.Module):
    def __init__(self, env, history_length=50, obs_shape=None, action_shape=None):
        super().__init__()
        if obs_shape is not None:
            obs_dim = np.prod(obs_shape)
            action_dim = np.prod(action_shape)
        else:
            obs_dim = np.array(env.single_observation_space.shape).prod()
            action_dim = np.prod(env.single_action_space.shape)
        # observation encoder
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)

        # history adapter
        self.adapter = HistoryAdapter(history_length, obs_dim, action_dim)

        # decision layers
        self.fc3 = nn.Linear(256 + 32, 256)
        self.fc_mean = nn.Linear(256, action_dim)
        self.fc_logstd = nn.Linear(256, action_dim)
        # action rescaling, to the environment's bounds
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (env.single_action_space.high - env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (env.single_action_space.high + env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, obs, history):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))

        z = self.adapter(history)

        # concatenate encodings of observation and history
        x = torch.cat([x, z], dim=1)
        x = F.relu(self.fc3(x))

        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mean, log_std

    def get_action(self, obs, history):
        mean, log_std = self(obs, history)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # enforcing action bounds
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean_action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean_action


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        obs_dim = np.array(env.single_observation_space.shape).prod()
        action_dim = np.prod(env.single_action_space.shape)
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, action_dim)
        self.fc_logstd = nn.Linear(256, action_dim)
        # action rescaling, to the environment's bounds
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (env.single_action_space.high - env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (env.single_action_space.high + env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # enforcing action bounds
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean_action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean_action


class Critic(nn.Module):
    def __init__(self, env, num_quantiles=1):
        super().__init__()
        obs_dim = np.array(env.single_observation_space.shape).prod()
        action_dim = np.prod(env.single_action_space.shape)
        self.fc1 = nn.Linear(obs_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, num_quantiles)
        self.num_quantiles = num_quantiles

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q = self.fc4(x)
        return q


class CriticContext(nn.Module):
    def __init__(self, env=None, num_quantiles=1, obs_shape=None, action_shape=None, context_shape=None):
        super().__init__()
        if env is None:
            obs_dim = np.prod(obs_shape)
            action_dim = np.prod(action_shape)
            context_dim = np.prod(context_shape)
        else:
            obs_dim = np.array(env.single_observation_space.shape).prod()
            action_dim = np.prod(env.single_action_space.shape)
            context_dim = 2  # for wind envs: context is speed in x and z directions
        # observation encoder
        self.fc1 = nn.Linear(obs_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        # context encoder
        self.enc1 = nn.Linear(context_dim, 32)
        self.enc2 = nn.Linear(32, 32)
        # decision layers
        self.fc3 = nn.Linear(256 + 32, 256)
        self.fc4 = nn.Linear(256, num_quantiles)
        self.num_quantiles = num_quantiles

    def forward(self, obs, act, context):
        x = torch.cat([obs, act], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        z = F.relu(self.enc1(context))
        z = F.relu(self.enc2(z))
        # concatenate encodings of (observation,action) and context
        x = torch.cat([x, z], dim=1)

        x = F.relu(self.fc3(x))
        q = self.fc4(x)
        return q


class HistoryAdapter(nn.Module):
    def __init__(self, history_length, obs_dim, action_dim):
        super().__init__()
        self.history_length = history_length

        self.fc1 = nn.Linear(obs_dim + action_dim, 32)
        big_stride = 4 if history_length >= 10 else 1
        big_pad = 3 if history_length >= 10 else 1
        self.conv1 = nn.Conv1d(32, 32, kernel_size=8, stride=big_stride, padding=big_pad, padding_mode='replicate')
        self.conv2 = nn.Conv1d(32, 32, kernel_size=5, stride=1, padding=2, padding_mode='replicate')
        self.conv3 = nn.Conv1d(32, 32, kernel_size=5, stride=1, padding=2, padding_mode='replicate')
        flat_dim = 32 * math.ceil(history_length / big_stride)
        self.flat = nn.Flatten()
        self.fc2 = nn.Linear(flat_dim, 32)

    def forward(self, history):
        """
        Forward pass through the history adapter.
        The first fc layer processes each history entry separately. (done automatically by a Linear layer)
        :param history: tensor of shape (batch_size, history_length, obs_dim + action_dim)
        :return: encoding of shape (batch_size, 32)
        """
        x = F.relu(self.fc1(history))  # (B, 50, 32)  for H=50
        x = x.permute(0, 2, 1)         # (B, 32, 50)
        x = F.relu(self.conv1(x))      # (B, 32, 13)
        x = F.relu(self.conv2(x))      # (B, 32, 13)
        x = F.relu(self.conv3(x))      # (B, 32, 13)
        x = self.flat(x)               # (B, 416)
        x = F.relu(self.fc2(x))        # (B, 32)
        return x


class ContextEncoder(nn.Module):
    def __init__(self, context_dim):
        super().__init__()
        self.enc1 = nn.Linear(context_dim, 32)
        self.enc2 = nn.Linear(32, 32)

    def forward(self, context):
        x = F.relu(self.enc1(context))
        z = F.relu(self.enc2(x))
        return z
